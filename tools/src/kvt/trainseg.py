"""Train KeybedSegNet on synthetic keybed masks, then on the real labelled frames."""

import argparse
import math
from collections.abc import Iterator
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from kvt.dataset import DEFAULT_FRAMES_DIR, DEFAULT_SYNTH_DIR, Frame, load_frames, load_synth
from kvt.model import KeybedSegNet, mask_targets, preprocess
from kvt.render import render_sample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEG_PATH = _REPO_ROOT / "data" / "models" / "keybed_seg.pt"

_FRAME_SIZE = (640, 480)
_DEFAULT_TRAIN_SAMPLES = 20_000
_DEFAULT_VAL_SAMPLES = 800
_DEFAULT_EPOCHS = 5
_BATCH_SIZE = 32
_LEARNING_RATE = 1e-3
_MIN_LEARNING_RATE = 1e-5
_REAL_FRACTION = 0.25
_SYNTH_FRACTION = 0.4
_SEED = 0
_RENDER_WORKERS = 4
_CHUNK_SIZE = 8


def _render_item(task: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    seed, epoch, index = task
    width, height = _FRAME_SIZE
    sample = render_sample(np.random.default_rng([seed, epoch, index]), width, height)
    return preprocess(sample.image), sample.mask


def _iter_batches(
    pool: Pool | None, seed: int, epoch: int, count: int, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    tasks = [(seed, epoch, index) for index in range(count)]
    rendered = (
        pool.imap(_render_item, tasks, chunksize=_CHUNK_SIZE)
        if pool is not None
        else map(_render_item, tasks)
    )
    inputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for item_input, item_mask in rendered:
        inputs.append(item_input)
        masks.append(item_mask)
        if len(inputs) == batch_size:
            yield np.stack(inputs), mask_targets(np.stack(masks))
            inputs, masks = [], []
    if inputs:
        yield np.stack(inputs), mask_targets(np.stack(masks))


def frame_items(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray] | None:
    inputs: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for frame in frames:
        if frame.corners_px is None:
            continue
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise ValueError(f"cannot read frame {frame.image_path}")
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [frame.corners_px.astype(np.int32)], 255)
        inputs.append(preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        # frames differ in size, so each mask is resampled to the target grid before stacking
        masks.append(mask_targets(mask[None])[0])
    if not inputs:
        return None
    return np.stack(inputs), np.stack(masks)


def _iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    predicted = torch.sigmoid(logits) > 0.5
    truth = target > 0.5
    union = float((predicted | truth).sum())
    return float((predicted & truth).sum()) / union if union else 1.0


def _evaluate(
    model: KeybedSegNet,
    pool: Pool | None,
    samples: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    seen = 0
    with torch.no_grad():
        for inputs, masks in _iter_batches(pool, seed + 1, 0, samples, batch_size):
            logits = model(torch.from_numpy(inputs).unsqueeze(1).to(device))
            total += _iou(logits, torch.from_numpy(masks).unsqueeze(1).to(device)) * inputs.shape[0]
            seen += inputs.shape[0]
    return total / max(seen, 1)


def train_seg(
    train_samples: int = _DEFAULT_TRAIN_SAMPLES,
    val_samples: int = _DEFAULT_VAL_SAMPLES,
    epochs: int = _DEFAULT_EPOCHS,
    batch_size: int = _BATCH_SIZE,
    seed: int = _SEED,
    workers: int = _RENDER_WORKERS,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    real_fraction: float = _REAL_FRACTION,
    exclude_kinds: tuple[str, ...] = (),
    synth_dir: Path = DEFAULT_SYNTH_DIR,
    synth_fraction: float = _SYNTH_FRACTION,
) -> tuple[KeybedSegNet, float]:
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training device {device}", flush=True)
    model = KeybedSegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    steps = math.ceil(train_samples / batch_size) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=_MIN_LEARNING_RATE
    )
    frames = [f for f in load_frames(frames_dir) if f.kind not in exclude_kinds]
    real = frame_items(frames) if real_fraction > 0.0 else None
    synth = frame_items(load_synth(synth_dir)) if synth_fraction > 0.0 else None
    if synth is not None:
        print(f"mixing {synth[0].shape[0]} rendered frames at {synth_fraction:.0%}", flush=True)
    rng = np.random.default_rng(seed)
    pool = Pool(processes=workers) if workers > 0 else None
    best = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    try:
        for epoch in range(epochs):
            model.train()
            for synthetic_inputs, synthetic_masks in _iter_batches(
                pool, seed, epoch, train_samples, batch_size
            ):
                # real frames ride in every batch so the synthetic prior is never overwritten
                batch_inputs, batch_masks = synthetic_inputs, synthetic_masks
                for pool_items, fraction in ((real, real_fraction), (synth, synth_fraction)):
                    if pool_items is None:
                        continue
                    pick = rng.integers(
                        0, pool_items[0].shape[0], size=max(1, int(batch_size * fraction))
                    )
                    batch_inputs = np.concatenate([batch_inputs, pool_items[0][pick]])
                    batch_masks = np.concatenate([batch_masks, pool_items[1][pick]])
                optimizer.zero_grad()
                inputs = torch.from_numpy(batch_inputs).unsqueeze(1).to(device)
                targets = torch.from_numpy(batch_masks).unsqueeze(1).to(device)
                logits = model(inputs)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
            iou = _evaluate(model, pool, val_samples, batch_size, seed, device)
            print(f"epoch {epoch + 1}/{epochs} val_iou {iou:.4f}", flush=True)
            if iou > best:
                best = iou
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best


def main() -> None:
    parser = argparse.ArgumentParser(description="train the keybed segmentation net")
    parser.add_argument("--train-samples", type=int, default=_DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=_DEFAULT_VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--out", type=Path, default=DEFAULT_SEG_PATH)
    parser.add_argument("--real-fraction", type=float, default=_REAL_FRACTION)
    parser.add_argument("--exclude-kind", action="append", default=[])
    parser.add_argument("--synth-fraction", type=float, default=_SYNTH_FRACTION)
    args = parser.parse_args()
    model, iou = train_seg(
        args.train_samples,
        args.val_samples,
        args.epochs,
        real_fraction=args.real_fraction,
        exclude_kinds=tuple(args.exclude_kind),
        synth_fraction=args.synth_fraction,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out} val_iou {iou:.4f}")


if __name__ == "__main__":
    main()
