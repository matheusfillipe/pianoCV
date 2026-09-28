"""Train KeySegNet on oriented crops streamed from data/synth-keys, and from real frames the app
labelled itself when given.

Runs on a GPU or locally. `--preview N` writes N crops with their labels drawn over
them and stops, which is how we check the labels before spending GPU time on them.
"""

import argparse
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

from pianocv.keymatch import (
    DEFAULT_DATA_DIR,
    PerturbConfig,
    SynthKeysFrame,
    load_synth_keys,
    perturb_quad,
)
from pianocv.keyseg import (
    BLACK,
    BOUNDARY,
    CLASSES,
    IGNORE,
    WHITE,
    KeySegNet,
    class_iou,
    crop_for,
    crop_image,
    export_keyseg_onnx,
    keyseg_from_onnx,
    label_map,
    preprocess_crop,
    refine_by_brightness,
)
from pianocv.trainkeymatch import _augment_strip

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised only where mlflow is absent
    mlflow = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "models" / "keyseg"

_DEFAULT_EPOCHS = 30
_DEFAULT_BATCH = 16
_DEFAULT_LR = 1e-3
_DEFAULT_SEED = 0
_DEFAULT_STEPS_PER_EPOCH = 300
_VAL_FRACTION = 0.05
_VAL_SEED = 12345
# the boundary and the black keys cover few pixels next to the white keys and the background,
# so we weight them up or the net would learn to paint every key white
_CLASS_WEIGHTS = (1.0, 1.0, 2.0, 4.0)
_OVERLAY = {
    WHITE: (60, 200, 60),
    BLACK: (200, 60, 200),
    BOUNDARY: (0, 0, 255),
    IGNORE: (90, 90, 90),
}


def _sample(
    frame: SynthKeysFrame, rng: np.random.Generator, config: PerturbConfig, augment: bool
) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read frame {frame.image_path}")
    crop = crop_for(perturb_quad(frame.corners_px, rng, config))
    pixels = crop_image(image, crop)
    labels = label_map(frame, crop)
    if frame.real:
        labels = refine_by_brightness(labels, pixels)
    if augment:
        pixels = _augment_strip(pixels, rng)
    return pixels, labels


class KeySegDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Draws a fresh quad perturbation and augmentation of a random frame on every read, from
    the real frames `real_share` of the time when there are any."""

    def __init__(
        self,
        frames: list[SynthKeysFrame],
        length: int,
        seed: int,
        config: PerturbConfig,
        real: list[SynthKeysFrame] | None = None,
        real_share: float = 0.0,
    ) -> None:
        self.frames = frames
        self.length = length
        self.seed = seed
        self.config = config
        self.real = real or []
        self.real_share = real_share if self.real else 0.0
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng((self.seed, self.epoch, worker_id, index))
        pool = self.real if rng.random() < self.real_share else self.frames
        frame = pool[int(rng.integers(len(pool)))]
        pixels, labels = _sample(frame, rng, self.config, augment=True)
        return torch.from_numpy(preprocess_crop(pixels)), torch.from_numpy(labels.astype(np.int64))


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    val_loss: float
    iou: tuple[float, ...]
    real_iou: tuple[float, ...] | None = None


def _validation(
    frames: list[SynthKeysFrame], config: PerturbConfig
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    samples: list[tuple[torch.Tensor, torch.Tensor]] = []
    for index, frame in enumerate(frames):
        pixels, labels = _sample(
            frame, np.random.default_rng((_VAL_SEED, index)), config, augment=False
        )
        samples.append(
            (torch.from_numpy(preprocess_crop(pixels)), torch.from_numpy(labels.astype(np.int64)))
        )
    return samples


def _evaluate(
    model: KeySegNet,
    samples: list[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: nn.Module,
    device: torch.device,
    batch: int,
) -> tuple[float, tuple[float, ...]]:
    model.eval()
    total = 0.0
    ious: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start : start + batch]
            crops = torch.stack([crop for crop, _ in chunk]).to(device)
            labels = torch.stack([label for _, label in chunk]).to(device)
            logits = model(crops)
            total += float(loss_fn(logits, labels)) * len(chunk)
            predicted = logits.argmax(dim=1).cpu().numpy()
            truth = labels.cpu().numpy()
            ious.extend(class_iou(p, t) for p, t in zip(predicted, truth, strict=True))
    mean_iou = tuple(float(np.mean([row[c] for row in ious])) for c in range(CLASSES))
    return total / max(len(samples), 1), mean_iou


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _held_out(frame: SynthKeysFrame, clips: tuple[str, ...]) -> bool:
    return any(clip in frame.image_path.stem for clip in clips)


def train_keyseg(
    data_dir: Path = DEFAULT_DATA_DIR,
    epochs: int = _DEFAULT_EPOCHS,
    batch: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    seed: int = _DEFAULT_SEED,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
    workers: int = 4,
    pretrained: bool = True,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
    *,
    init_onnx: Path | None = None,
    real_dir: Path | None = None,
    real_share: float = 0.0,
    hold_out: tuple[str, ...] = (),
) -> tuple[KeySegNet, list[EpochMetrics]]:
    device = _device()
    print(f"training device {device}", flush=True)
    frames = load_synth_keys(data_dir)
    if not frames:
        raise ValueError(f"no synth-keys frames found in {data_dir}")
    order = np.random.default_rng(seed).permutation(len(frames))
    val_count = max(1, round(len(frames) * _VAL_FRACTION)) if len(frames) > 1 else 0
    val_indices = set(order[:val_count].tolist())
    train_frames = [frame for i, frame in enumerate(frames) if i not in val_indices] or frames
    val_frames = [frame for i, frame in enumerate(frames) if i in val_indices] or frames
    real = load_synth_keys(real_dir) if real_dir is not None else []
    real_train = [frame for frame in real if not _held_out(frame, hold_out)]
    real_val = [frame for frame in real if _held_out(frame, hold_out)]
    print(f"real frames {len(real_train)} to train on, {len(real_val)} held out", flush=True)

    config = PerturbConfig()
    dataset = KeySegDataset(
        train_frames, steps_per_epoch * batch, seed, config, real_train, real_share
    )
    loader = DataLoader(dataset, batch_size=batch, num_workers=workers, shuffle=False)
    validation = _validation(val_frames, config)
    real_validation = _validation(real_val, config)

    torch.manual_seed(seed)
    model = (
        keyseg_from_onnx(str(init_onnx))
        if init_onnx is not None
        else KeySegNet(pretrained=pretrained)
    ).to(device)
    # a model rebuilt from its export carries the folded statistics in its batch norms, and a
    # batch of crops would overwrite them, so those stay in eval mode while the rest trains
    frozen_norms = (
        [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
        if init_onnx is not None
        else []
    )
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(_CLASS_WEIGHTS, device=device), ignore_index=IGNORE
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(steps_per_epoch * epochs, 1)
    )
    history: list[EpochMetrics] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        for norm in frozen_norms:
            norm.eval()
        running = 0.0
        steps = 0
        for crops, labels in loader:
            crops, labels = crops.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(crops), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
            steps += 1
        val_loss, iou = _evaluate(model, validation, loss_fn, device, batch)
        real_iou = (
            _evaluate(model, real_validation, loss_fn, device, batch)[1]
            if real_validation
            else None
        )
        metrics = EpochMetrics(epoch + 1, running / max(steps, 1), val_loss, iou, real_iou)
        history.append(metrics)
        held = (
            f" held out real white {real_iou[1]:.3f} black {real_iou[2]:.3f} "
            f"boundary {real_iou[3]:.3f}"
            if real_iou is not None
            else ""
        )
        print(
            f"epoch {metrics.epoch}/{epochs} loss {metrics.loss:.4f} val_loss {val_loss:.4f} "
            f"iou bg {iou[0]:.3f} white {iou[1]:.3f} black {iou[2]:.3f} boundary {iou[3]:.3f}"
            f"{held}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(metrics)
    return model, history


def write_previews(data_dir: Path, out_dir: Path, count: int) -> list[Path]:
    frames = load_synth_keys(data_dir)
    rng = np.random.default_rng(_VAL_SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(min(count, len(frames))):
        frame = frames[int(rng.integers(len(frames)))]
        pixels, labels = _sample(frame, rng, PerturbConfig(), augment=False)
        shown = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        tint = shown.copy()
        for label, colour in _OVERLAY.items():
            tint[labels == label] = colour
        path = out_dir / f"keyseg-preview-{index}.png"
        cv2.imwrite(str(path), np.vstack([shown, cv2.addWeighted(shown, 0.5, tint, 0.5, 0)]))
        written.append(path)
    return written


def _require_mlflow() -> ModuleType:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed; add it to the tools dev dependency group")
    return mlflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train KeySegNet on oriented keybed crops")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--steps-per-epoch", type=int, default=_DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preview", type=int, default=0, help="write N label previews and stop")
    parser.add_argument("--init-onnx", type=Path, help="start from the weights of this export")
    parser.add_argument("--real-dir", type=Path, help="real frames labelled in the same format")
    parser.add_argument("--real-share", type=float, default=0.5)
    parser.add_argument(
        "--hold-out", default="", help="comma separated clip names kept for validation only"
    )
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics and artifacts")
    parser.add_argument(
        "--mlflow-experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "keyseg")
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.preview > 0:
        for path in write_previews(args.data_dir, args.out_dir, args.preview):
            print(path)
        return
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
                    "steps_per_epoch": args.steps_per_epoch,
                    "init_onnx": str(args.init_onnx or ""),
                    "real_dir": str(args.real_dir or ""),
                    "real_share": args.real_share,
                    "hold_out": args.hold_out,
                }
            )

        def log_epoch(metrics: EpochMetrics) -> None:
            if tracker is None:
                return
            tracker.log_metrics(
                {
                    "loss": metrics.loss,
                    "val_loss": metrics.val_loss,
                    "iou_white": metrics.iou[WHITE],
                    "iou_black": metrics.iou[BLACK],
                    "iou_boundary": metrics.iou[BOUNDARY],
                    **(
                        {
                            "real_iou_white": metrics.real_iou[WHITE],
                            "real_iou_black": metrics.real_iou[BLACK],
                            "real_iou_boundary": metrics.real_iou[BOUNDARY],
                        }
                        if metrics.real_iou is not None
                        else {}
                    ),
                },
                step=metrics.epoch,
            )

        model, history = train_keyseg(
            args.data_dir,
            args.epochs,
            args.batch,
            args.lr,
            args.seed,
            args.steps_per_epoch,
            args.workers,
            on_epoch=log_epoch,
            init_onnx=args.init_onnx,
            real_dir=args.real_dir,
            real_share=args.real_share,
            hold_out=tuple(clip for clip in args.hold_out.split(",") if clip),
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = args.out_dir / "keyseg.pt"
        onnx_path = args.out_dir / "keyseg.onnx"
        torch.save(model.state_dict(), pt_path)
        export_keyseg_onnx(model.cpu(), str(onnx_path))
        last = history[-1]
        for stem, value in (
            ("iou_white", last.iou[WHITE]),
            ("iou_black", last.iou[BLACK]),
            ("iou_boundary", last.iou[BOUNDARY]),
        ):
            (args.out_dir / f"{stem}.txt").write_text(f"{value:.6f}")
        if tracker is not None:
            tracker.log_artifact(str(pt_path))
            tracker.log_artifact(str(onnx_path))
    print(f"saved {pt_path} iou white {last.iou[WHITE]:.3f} black {last.iou[BLACK]:.3f}")


if __name__ == "__main__":
    main()
