"""Fine-tune KeybedSegNet2 on synthetic renders and real frames, on a GPU or locally."""

import argparse
import json
import math
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from pianocv.export import export_seg2_onnx
from pianocv.model import MASK_SIZE
from pianocv.segnet2 import SEG2_INPUT_SIZE, KeybedSegNet2

try:
    import mlflow
    import mlflow.data
except ImportError:  # pragma: no cover - exercised only where mlflow is absent
    mlflow = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SYNTHETIC_DIR = _REPO_ROOT / "data" / "corpus"
DEFAULT_REAL_DIR = _REPO_ROOT / "data" / "real-seg2"
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "models" / "seg2-tuned"

_RASTER = 576
_DEFAULT_EPOCHS = 25
_DEFAULT_BATCH = 48
_DEFAULT_LR = 3e-4
_DEFAULT_WEIGHT_DECAY = 1e-4
_DEFAULT_SEED = 0
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    synthetic_iou: float
    real_val_iou: float
    held_out_iou: float


def mask_from_quad(
    corners: Sequence[Sequence[float]], mask_size: int = MASK_SIZE, raster: int = _RASTER
) -> np.ndarray:
    big = np.zeros((raster, raster), dtype=np.uint8)
    cv2.fillPoly(big, [(np.asarray(corners, dtype=np.float64) * raster).astype(np.int32)], 255)
    small = cv2.resize(big, (mask_size, mask_size), interpolation=cv2.INTER_AREA)
    return np.asarray(small, dtype=np.uint8)


def validate_real_splits(splits: dict[str, list[str]], labels: dict[str, object]) -> None:
    grouped = {name: set(members) for name, members in splits.items()}
    covered: set[str] = set().union(*grouped.values()) if grouped else set()
    if covered != set(labels) or sum(len(members) for members in grouped.values()) != len(labels):
        raise ValueError("real splits must be disjoint and cover every label")
    if not all(grouped.get(name) for name in ("train", "validation", "held_out")):
        raise ValueError("real splits must include train, validation, held_out")


def _load_synthetic(
    synthetic_dir: Path, mask_size: int, raster: int, colour: bool
) -> tuple[np.ndarray, np.ndarray]:
    corners = json.loads((synthetic_dir / "corners.json").read_text())
    read_flag = cv2.IMREAD_COLOR if colour else cv2.IMREAD_GRAYSCALE
    conversion = cv2.COLOR_BGR2RGB if colour else cv2.COLOR_GRAY2RGB

    def load_one(stem: str) -> tuple[np.ndarray, np.ndarray] | None:
        image = cv2.imread(str(synthetic_dir / "frames" / f"{stem}.png"), read_flag)
        if image is None:
            return None
        resized = cv2.resize(
            cv2.cvtColor(image, conversion),
            (SEG2_INPUT_SIZE, SEG2_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=np.uint8), mask_from_quad(corners[stem], mask_size, raster)

    with ThreadPoolExecutor(max_workers=8) as workers:
        pairs = [pair for pair in workers.map(load_one, sorted(corners)) if pair is not None]
    if not pairs:
        raise ValueError(f"no synthetic frames found in {synthetic_dir}")
    return np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])


def _load_real(real_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    splits: dict[str, list[str]] = json.loads((real_dir / "splits.json").read_text())
    labels: dict[str, object] = json.loads((real_dir / "labels.json").read_text())
    validate_real_splits(splits, labels)
    frame_root, mask_root = real_dir / "frames", real_dir / "masks"

    def load_one(name: str) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(frame_root / name), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_root / name), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"missing real frame or mask: {name}")
        resized_image = cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            (SEG2_INPUT_SIZE, SEG2_INPUT_SIZE),
            interpolation=cv2.INTER_AREA,
        )
        resized_mask = cv2.resize(mask, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA)
        return np.asarray(resized_image, dtype=np.uint8), np.asarray(resized_mask, dtype=np.uint8)

    pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, names in splits.items():
        with ThreadPoolExecutor(max_workers=8) as workers:
            loaded = list(workers.map(load_one, sorted(names)))
        images = np.stack([item[0] for item in loaded])
        masks = np.stack([item[1] for item in loaded])
        pairs[split] = (images, masks)
    return pairs


def train_seg2(
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    real_dir: Path = DEFAULT_REAL_DIR,
    epochs: int = _DEFAULT_EPOCHS,
    batch: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    seed: int = _DEFAULT_SEED,
    pretrained: bool = True,
    synthetic_colour: bool = True,
    on_dataset_loaded: Callable[[np.ndarray], None] | None = None,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> tuple[KeybedSegNet2, list[EpochMetrics]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device {device}", flush=True)
    synthetic_images, synthetic_masks = _load_synthetic(
        synthetic_dir, MASK_SIZE, _RASTER, synthetic_colour
    )
    if on_dataset_loaded is not None:
        on_dataset_loaded(synthetic_images)
    real_pairs = _load_real(real_dir)
    train_images, train_masks = real_pairs["train"]
    val_images, val_masks = real_pairs["validation"]
    held_images, held_masks = real_pairs["held_out"]
    stores: dict[str, tuple[torch.Tensor, torch.Tensor]] = {
        "synthetic": (torch.from_numpy(synthetic_images), torch.from_numpy(synthetic_masks)),
        "real_train": (torch.from_numpy(train_images), torch.from_numpy(train_masks)),
        "real_val": (torch.from_numpy(val_images), torch.from_numpy(val_masks)),
        "real_held": (torch.from_numpy(held_images), torch.from_numpy(held_masks)),
    }
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def batch_images(name: str, index: np.ndarray, jitter: bool) -> torch.Tensor:
        picked = stores[name][0][torch.from_numpy(np.ascontiguousarray(index))]
        out = picked.to(device).permute(0, 3, 1, 2).float().div_(255)
        if jitter:
            gain = torch.empty(out.shape[0], 1, 1, 1, device=device).uniform_(0.7, 1.4)
            bias = torch.empty(out.shape[0], 1, 1, 1, device=device).uniform_(-0.16, 0.16)
            out = (out * gain + bias).clamp_(0, 1)
        return out.sub_(mean).div_(std)

    def truth(name: str, index: np.ndarray) -> torch.Tensor:
        picked = stores[name][1][torch.from_numpy(np.ascontiguousarray(index))]
        return picked.float().div_(255).unsqueeze(1).to(device)

    def evaluate(model: KeybedSegNet2, name: str) -> float:
        inter = union = 0.0
        model.eval()
        with torch.no_grad():
            count = stores[name][0].shape[0]
            for start in range(0, count, batch):
                pick = np.arange(start, min(start + batch, count))
                predicted = torch.sigmoid(model(batch_images(name, pick, False))) > 0.5
                actual = truth(name, pick) > 0.5
                inter += float((predicted & actual).sum())
                union += float((predicted | actual).sum())
        return inter / max(union, 1.0)

    torch.manual_seed(seed)
    model = KeybedSegNet2(pretrained=pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=_DEFAULT_WEIGHT_DECAY)
    order = np.random.default_rng(seed).permutation(len(synthetic_images))
    synthetic_train = order[max(1, int(len(order) * 0.08)) :]
    rng = np.random.default_rng(seed)
    half = batch // 2
    real_train_count = stores["real_train"][0].shape[0]
    step_count = max(math.ceil(len(synthetic_train) / half), math.ceil(real_train_count / half))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=step_count * epochs, eta_min=1e-6
    )
    history: list[EpochMetrics] = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(step_count):
            synthetic_index = rng.choice(synthetic_train, half, replace=True)
            real_index = rng.choice(real_train_count, half, replace=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                torch.cat(
                    [
                        batch_images("synthetic", synthetic_index, True),
                        batch_images("real_train", real_index, True),
                    ]
                )
            )
            target = torch.cat(
                [truth("synthetic", synthetic_index), truth("real_train", real_index)]
            )
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += float(loss.detach())
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metrics = EpochMetrics(
            epoch=epoch + 1,
            loss=epoch_loss / step_count,
            synthetic_iou=evaluate(model, "synthetic"),
            real_val_iou=evaluate(model, "real_val"),
            held_out_iou=evaluate(model, "real_held"),
        )
        history.append(metrics)
        print(
            f"epoch {metrics.epoch}/{epochs} synthetic_iou {metrics.synthetic_iou:.4f} "
            f"real_val_iou {metrics.real_val_iou:.4f} held_out_iou {metrics.held_out_iou:.4f}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(metrics)
    return model, history


def _dataset_source_uri(bucket: str, key: str, real_dir: Path) -> str:
    return f"s3://{bucket}/{key}" if key else real_dir.resolve().as_uri()


def _require_mlflow() -> ModuleType:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed; add it to the tools dev dependency group")
    return mlflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="fine-tune KeybedSegNet2 on synthetic renders and real labelled frames"
    )
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC_DIR)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--synthetic-colour",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read synthetic frames as colour, or grayscale like the notebook did",
    )
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics and artifacts")
    parser.add_argument(
        "--mlflow-experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "keybed-seg2")
    )
    parser.add_argument("--dataset-bucket", default=os.environ.get("DATASET_BUCKET", "datasets"))
    parser.add_argument("--dataset-key", default=os.environ.get("DATASET_KEY", ""))
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    tracker = _require_mlflow() if args.mlflow else None
    if tracker is not None:
        tracker.set_experiment(args.mlflow_experiment)
    run_context: AbstractContextManager[object] = (
        tracker.start_run() if tracker is not None else nullcontext()
    )
    with run_context:
        if tracker is not None:
            tracker.log_params(
                {
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "lr": args.lr,
                    "seed": args.seed,
                    "synthetic_colour": args.synthetic_colour,
                }
            )

        def log_dataset(synthetic_images: np.ndarray) -> None:
            if tracker is None:
                return
            source = _dataset_source_uri(args.dataset_bucket, args.dataset_key, args.real_dir)
            dataset = tracker.data.from_numpy(synthetic_images, source=source, name="keybed-seg2")
            tracker.log_input(dataset, context="training")

        def log_epoch(metrics: EpochMetrics) -> None:
            if tracker is None:
                return
            tracker.log_metrics(
                {
                    "loss": metrics.loss,
                    "synthetic_iou": metrics.synthetic_iou,
                    "real_val_iou": metrics.real_val_iou,
                    "held_out_iou": metrics.held_out_iou,
                },
                step=metrics.epoch,
            )

        model, history = train_seg2(
            args.synthetic_dir,
            args.real_dir,
            args.epochs,
            args.batch,
            args.lr,
            args.seed,
            synthetic_colour=args.synthetic_colour,
            on_dataset_loaded=log_dataset,
            on_epoch=log_epoch,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = args.out_dir / "keybed_seg2.pt"
        torch.save(model.state_dict(), pt_path)
        onnx_path = export_seg2_onnx(pt_path, args.out_dir / "keybed_seg2.onnx")
        last = history[-1]
        for stem, value in (
            ("synthetic_iou", last.synthetic_iou),
            ("real_val_iou", last.real_val_iou),
            ("held_out_iou", last.held_out_iou),
        ):
            (args.out_dir / f"{stem}.txt").write_text(f"{value:.6f}")
        if tracker is not None:
            tracker.log_artifact(str(pt_path))
            tracker.log_artifact(str(onnx_path))

    print(
        f"saved {pt_path} synthetic_iou {last.synthetic_iou:.4f} "
        f"real_val_iou {last.real_val_iou:.4f} held_out_iou {last.held_out_iou:.4f}"
    )


if __name__ == "__main__":
    main()
