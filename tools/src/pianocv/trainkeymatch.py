"""Train KeyMatchNet on rectified strips streamed from data/synth-keys.

Runs on a GPU or locally.
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
from torch.utils.data import DataLoader, Dataset, get_worker_info

from pianocv.keymatch import (
    DEFAULT_DATA_DIR,
    PEAK_TOLERANCE_PX,
    ChannelMetrics,
    KeyMatchNet,
    PerturbConfig,
    SynthKeysFrame,
    compute_targets,
    export_keymatch_onnx,
    find_peaks,
    focal_heatmap_loss,
    load_synth_keys,
    match_peaks,
    perturb_quad,
    preprocess_strip,
    rectify_strip,
)

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised only where mlflow is absent
    mlflow = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "models" / "keymatch"

_DEFAULT_EPOCHS = 25
_DEFAULT_BATCH = 32
_DEFAULT_LR = 3e-4
_DEFAULT_SEED = 0
_DEFAULT_STEPS_PER_EPOCH = 100
_VAL_FRACTION = 0.1


def _require_corners(frame: SynthKeysFrame) -> np.ndarray:
    # data/synth-keys carries positives only, but the type is shared with negatives now
    if frame.corners_px is None:
        raise ValueError(f"frame {frame.image_path} has no keybed corners")
    return frame.corners_px


_VAL_SEED = 12345
_CHANNEL_NAMES = ("white", "black_left", "black_right")


def _draw_blob(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
    axes = (int(rng.integers(6, 40)), int(rng.integers(4, 16)))
    angle = float(rng.uniform(0.0, 180.0))
    dark = rng.random() < 0.5
    shade = int(rng.integers(0, 40))
    color = (
        (shade, shade, shade)
        if dark
        else (
            int(rng.integers(180, 230)),
            int(rng.integers(120, 180)),
            int(rng.integers(100, 150)),
        )
    )
    overlay = image.copy()
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
    alpha = rng.uniform(0.4, 0.85)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _draw_glare(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)
    x0, x1 = int(rng.integers(0, width)), int(rng.integers(0, width))
    thickness = int(rng.integers(2, 8))
    cv2.line(overlay, (x0, 0), (x1, height), (255, 255, 255), thickness)
    alpha = rng.uniform(0.2, 0.5)
    cv2.addWeighted(overlay, alpha, image, 1.0, 0.0, dst=image)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(values, 0.0, 255.0), dtype=np.uint8)


def _augment_strip(strip_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    working = strip_rgb.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20.0, 20.0)
    gamma = rng.uniform(0.7, 1.4)
    working = 255.0 * (np.clip(working, 0.0, 255.0) / 255.0) ** gamma
    working = working + rng.uniform(-12.0, 12.0, size=3)
    image = _to_uint8(working)

    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 5]))
        image = np.asarray(cv2.GaussianBlur(image, (ksize, ksize), 0), dtype=np.uint8)
    if rng.random() < 0.3:
        length = int(rng.integers(4, 12))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        matrix = cv2.getRotationMatrix2D(
            (length / 2, length / 2), float(rng.uniform(0.0, 180.0)), 1.0
        )
        kernel = np.asarray(cv2.warpAffine(kernel, matrix, (length, length)), dtype=np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        image = np.asarray(cv2.filter2D(image, -1, kernel), dtype=np.uint8)

    noise = rng.normal(scale=rng.uniform(2.0, 10.0), size=image.shape)
    image = _to_uint8(image.astype(np.float32) + noise)

    if rng.random() < 0.5:
        quality = int(rng.integers(35, 90))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
        if decoded is not None:
            image = np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    for _ in range(int(rng.integers(0, 3))):
        _draw_blob(image, rng)
    if rng.random() < 0.3:
        _draw_glare(image, rng)
    return image


class KeyMatchDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Draws a fresh perturbation and augmentation of a random frame on every read.

    `set_epoch` folds the epoch into the seed so a second pass over the same indices is not the
    same pass: without it, `shuffle=False` plus a per-worker deterministic seed would replay the
    exact same "random" batches every epoch.
    """

    def __init__(
        self,
        frames: list[SynthKeysFrame],
        length: int,
        seed: int,
        augment: bool,
        config: PerturbConfig | None = None,
    ) -> None:
        self.frames = frames
        self.length = length
        self.seed = seed
        self.augment = augment
        self.config = config or PerturbConfig()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng((self.seed, self.epoch, worker_id, index))
        frame = self.frames[int(rng.integers(len(self.frames)))]
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise FileNotFoundError(f"cannot read frame {frame.image_path}")
        quad = perturb_quad(_require_corners(frame), rng, self.config)
        strip = rectify_strip(image, quad)
        if self.augment:
            strip = _augment_strip(strip, rng)
        targets = compute_targets(frame, quad)
        return torch.from_numpy(preprocess_strip(strip)), torch.from_numpy(targets.heatmaps)


@dataclass(frozen=True)
class _ValidationSample:
    strip_tensor: torch.Tensor
    heatmaps: torch.Tensor
    positions: tuple[list[float], list[float], list[float]]


def _build_validation_samples(
    frames: list[SynthKeysFrame], seed: int, config: PerturbConfig
) -> list[_ValidationSample]:
    samples: list[_ValidationSample] = []
    for index, frame in enumerate(frames):
        rng = np.random.default_rng((seed, index))
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise FileNotFoundError(f"cannot read frame {frame.image_path}")
        quad = perturb_quad(_require_corners(frame), rng, config)
        strip = rectify_strip(image, quad)
        targets = compute_targets(frame, quad)
        samples.append(
            _ValidationSample(
                strip_tensor=torch.from_numpy(preprocess_strip(strip)),
                heatmaps=torch.from_numpy(targets.heatmaps),
                positions=(
                    targets.white_positions,
                    targets.black_left_positions,
                    targets.black_right_positions,
                ),
            )
        )
    return samples


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    val_loss: float
    white: ChannelMetrics
    black_left: ChannelMetrics
    black_right: ChannelMetrics


def _evaluate(
    model: KeyMatchNet, samples: list[_ValidationSample], device: torch.device, batch: int
) -> tuple[float, ChannelMetrics, ChannelMetrics, ChannelMetrics]:
    model.eval()
    total_loss = 0.0
    counts = {name: [0, 0, 0] for name in _CHANNEL_NAMES}
    errors: dict[str, list[float]] = {name: [] for name in _CHANNEL_NAMES}
    with torch.no_grad():
        for start in range(0, len(samples), batch):
            chunk = samples[start : start + batch]
            strips = torch.stack([sample.strip_tensor for sample in chunk]).to(device)
            heatmaps = torch.stack([sample.heatmaps for sample in chunk]).to(device)
            logits = model(strips)
            total_loss += float(focal_heatmap_loss(logits, heatmaps).detach()) * len(chunk)
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for sample_index, sample in enumerate(chunk):
                for channel_index, name in enumerate(_CHANNEL_NAMES):
                    predicted = find_peaks(probabilities[sample_index, channel_index])
                    truth = sample.positions[channel_index]
                    tp, fp, fn, sample_errors = match_peaks(predicted, truth, PEAK_TOLERANCE_PX)
                    counts[name][0] += tp
                    counts[name][1] += fp
                    counts[name][2] += fn
                    errors[name].extend(sample_errors)
    results = []
    for name in _CHANNEL_NAMES:
        true_positives, false_positives, false_negatives = counts[name]
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        mean_error = float(np.mean(errors[name])) if errors[name] else 0.0
        results.append(ChannelMetrics(precision, recall, mean_error))
    return total_loss / max(len(samples), 1), results[0], results[1], results[2]


def train_keymatch(
    data_dir: Path = DEFAULT_DATA_DIR,
    epochs: int = _DEFAULT_EPOCHS,
    batch: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    seed: int = _DEFAULT_SEED,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
    workers: int = 4,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> tuple[KeyMatchNet, list[EpochMetrics]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device {device}", flush=True)
    frames = load_synth_keys(data_dir)
    if not frames:
        raise ValueError(f"no synth-keys frames found in {data_dir}")

    order = np.random.default_rng(seed).permutation(len(frames))
    val_count = max(1, round(len(frames) * _VAL_FRACTION)) if len(frames) > 1 else 0
    val_indices = set(order[:val_count].tolist())
    train_frames = [frame for i, frame in enumerate(frames) if i not in val_indices] or frames
    val_frames = [frame for i, frame in enumerate(frames) if i in val_indices] or frames

    config = PerturbConfig()
    train_dataset = KeyMatchDataset(
        train_frames, steps_per_epoch * batch, seed, augment=True, config=config
    )
    loader = DataLoader(train_dataset, batch_size=batch, num_workers=workers, shuffle=False)
    validation_samples = _build_validation_samples(val_frames, _VAL_SEED, config)

    torch.manual_seed(seed)
    model = KeyMatchNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps_per_epoch * epochs, 1), eta_min=1e-6
    )

    history: list[EpochMetrics] = []
    for epoch in range(epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        step_count = 0
        for strips, targets in loader:
            strips, targets = strips.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(strips)
            loss = focal_heatmap_loss(logits, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.detach())
            step_count += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()
        val_loss, white, black_left, black_right = _evaluate(
            model, validation_samples, device, batch
        )
        metrics = EpochMetrics(
            epoch=epoch + 1,
            loss=running_loss / max(step_count, 1),
            val_loss=val_loss,
            white=white,
            black_left=black_left,
            black_right=black_right,
        )
        history.append(metrics)
        print(
            f"epoch {metrics.epoch}/{epochs} loss {metrics.loss:.4f} "
            f"val_loss {metrics.val_loss:.4f}\n"
            f"  white p{white.precision:.2f} r{white.recall:.2f} e{white.mean_error_px:.2f}px "
            f"black_left p{black_left.precision:.2f} r{black_left.recall:.2f} "
            f"black_right p{black_right.precision:.2f} r{black_right.recall:.2f}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(metrics)
    return model, history


def _require_mlflow() -> ModuleType:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed; add it to the tools dev dependency group")
    return mlflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train KeyMatchNet on rectified keybed strips")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--steps-per-epoch", type=int, default=_DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics and artifacts")
    parser.add_argument(
        "--mlflow-experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "keymatch")
    )
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
                    "steps_per_epoch": args.steps_per_epoch,
                }
            )

        def log_epoch(metrics: EpochMetrics) -> None:
            if tracker is None:
                return
            tracker.log_metrics(
                {
                    "loss": metrics.loss,
                    "val_loss": metrics.val_loss,
                    "white_precision": metrics.white.precision,
                    "white_recall": metrics.white.recall,
                    "white_mean_error_px": metrics.white.mean_error_px,
                    "black_left_precision": metrics.black_left.precision,
                    "black_left_recall": metrics.black_left.recall,
                    "black_left_mean_error_px": metrics.black_left.mean_error_px,
                    "black_right_precision": metrics.black_right.precision,
                    "black_right_recall": metrics.black_right.recall,
                    "black_right_mean_error_px": metrics.black_right.mean_error_px,
                },
                step=metrics.epoch,
            )

        model, history = train_keymatch(
            args.data_dir,
            args.epochs,
            args.batch,
            args.lr,
            args.seed,
            args.steps_per_epoch,
            args.workers,
            on_epoch=log_epoch,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = args.out_dir / "keymatch.pt"
        torch.save(model.state_dict(), pt_path)
        onnx_path = export_keymatch_onnx(pt_path, args.out_dir / "keymatch.onnx")
        last = history[-1]
        for stem, value in (
            ("white_precision", last.white.precision),
            ("white_recall", last.white.recall),
            ("white_mean_error_px", last.white.mean_error_px),
            ("black_left_precision", last.black_left.precision),
            ("black_left_recall", last.black_left.recall),
            ("black_left_mean_error_px", last.black_left.mean_error_px),
            ("black_right_precision", last.black_right.precision),
            ("black_right_recall", last.black_right.recall),
            ("black_right_mean_error_px", last.black_right.mean_error_px),
        ):
            (args.out_dir / f"{stem}.txt").write_text(f"{value:.6f}")
        if tracker is not None:
            tracker.log_artifact(str(pt_path))
            tracker.log_artifact(str(onnx_path))

    print(
        f"saved {pt_path} white_precision {last.white.precision:.4f} "
        f"black_left_precision {last.black_left.precision:.4f} "
        f"black_right_precision {last.black_right.precision:.4f}"
    )


if __name__ == "__main__":
    main()
