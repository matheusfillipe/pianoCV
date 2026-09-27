"""Train KeybedNet on synthetic renders and keep the best-val checkpoint."""

import argparse
import math
from collections.abc import Iterator
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np
import torch

from pianocv.dataset import DEFAULT_FRAMES_DIR, Frame, load_frames
from pianocv.model import (
    KeybedNet,
    corner_loss,
    decode_heatmaps,
    heatmap_targets,
    load_model,
    preprocess,
)
from pianocv.render import render_sample

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_net.pt"
DEFAULT_SYNTHETIC_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_net.synthetic.pt"

_FRAME_SIZE = (640, 480)
_DEFAULT_TRAIN_SAMPLES = 25_000
_DEFAULT_VAL_SAMPLES = 1_000
_DEFAULT_EPOCHS = 6
_BATCH_SIZE = 64
_LEARNING_RATE = 1e-3
_MIN_LEARNING_RATE = 1e-5
_FINE_TUNE_STEPS = 400
_FINE_TUNE_LEARNING_RATE = 1e-4
_FINE_TUNE_SEED_OFFSET = 1_000_000
_SEED = 0
_RENDER_WORKERS = 4
_CHUNK_SIZE = 8


def _render_item(task: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seed, epoch, index = task
    width, height = _FRAME_SIZE
    rng = np.random.default_rng([seed, epoch, index])
    sample = render_sample(rng, width, height)
    corners = (sample.quad_px / np.array([float(width), float(height)])).reshape(8)
    present = np.array(1.0 if sample.present else 0.0)
    return preprocess(sample.image), corners, present


def _iter_batches(
    pool: Pool | None, seed: int, epoch: int, count: int, batch_size: int
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    tasks = [(seed, epoch, index) for index in range(count)]
    rendered = (
        pool.imap(_render_item, tasks, chunksize=_CHUNK_SIZE)
        if pool is not None
        else map(_render_item, tasks)
    )
    inputs: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    present: list[np.ndarray] = []
    for item_inputs, item_corners, item_present in rendered:
        inputs.append(item_inputs)
        corners.append(item_corners)
        present.append(item_present)
        if len(inputs) == batch_size:
            yield np.stack(inputs), np.stack(corners), np.stack(present)
            inputs, corners, present = [], [], []
    if inputs:
        yield np.stack(inputs), np.stack(corners), np.stack(present)


def _train_epoch(
    model: KeybedNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    pool: Pool | None,
    train_samples: int,
    batch_size: int,
    seed: int,
    epoch: int,
) -> float:
    model.train()
    loss_sum = 0.0
    seen = 0
    for inputs, corners, present in _iter_batches(pool, seed, epoch, train_samples, batch_size):
        optimizer.zero_grad()
        corner_tensor = torch.from_numpy(corners)
        pred_heatmaps, present_logits = model(torch.from_numpy(inputs).unsqueeze(1))
        loss = corner_loss(
            pred_heatmaps,
            present_logits,
            heatmap_targets(corner_tensor, torch.from_numpy(present)),
            torch.from_numpy(present),
            corner_tensor,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()
        loss_sum += float(loss.detach()) * inputs.shape[0]
        seen += inputs.shape[0]
    return loss_sum / max(seen, 1)


def _evaluate(
    model: KeybedNet, pool: Pool | None, val_samples: int, batch_size: int, seed: int
) -> tuple[float, float, float, float]:
    model.eval()
    scale = np.array([float(_FRAME_SIZE[0]), float(_FRAME_SIZE[1])])
    error_sum = 0.0
    error_count = 0
    correct = 0
    seen = 0
    predicted_all: list[np.ndarray] = []
    target_all: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, corners, present in _iter_batches(pool, seed + 1, 0, val_samples, batch_size):
            pred_heatmaps, present_logits = model(torch.from_numpy(inputs).unsqueeze(1))
            probabilities = torch.sigmoid(present_logits).numpy()
            predicted = decode_heatmaps(pred_heatmaps).numpy().reshape(-1, 4, 2)
            target = corners.reshape(-1, 4, 2)
            flags = present.astype(bool)
            if flags.any():
                errors = np.linalg.norm((predicted[flags] - target[flags]) * scale, axis=2)
                error_sum += float(errors.mean(axis=1).sum())
                error_count += int(flags.sum())
                predicted_all.append(predicted[flags])
                target_all.append(target[flags])
            correct += int(((probabilities >= 0.5) == flags).sum())
            seen += inputs.shape[0]
    mae = error_sum / error_count if error_count else 0.0
    predicted_std = 0.0
    target_std = 0.0
    if predicted_all and target_all:
        predicted_std = float((np.concatenate(predicted_all).std(axis=0) * scale).mean())
        target_std = float((np.concatenate(target_all).std(axis=0) * scale).mean())
    return mae, correct / max(seen, 1), predicted_std, target_std


def _fine_tune_items(frames: list[Frame]) -> tuple[np.ndarray, np.ndarray] | None:
    inputs: list[np.ndarray] = []
    corners: list[np.ndarray] = []
    for frame in frames:
        if frame.kind != "rec" or frame.corners_px is None:
            continue
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise ValueError(f"cannot read frame {frame.image_path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        scale = np.array([1.0 / float(image.shape[1]), 1.0 / float(image.shape[0])])
        inputs.append(preprocess(rgb))
        corners.append((frame.corners_px * scale).reshape(8))
    if not inputs:
        return None
    return np.stack(inputs), np.stack(corners)


def _fine_tune(
    model: KeybedNet,
    items: tuple[np.ndarray, np.ndarray],
    pool: Pool | None,
    val_samples: int,
    batch_size: int,
    seed: int,
    steps: int,
) -> float:
    inputs, corners = items
    rng = np.random.default_rng([seed, _FINE_TUNE_SEED_OFFSET])
    optimizer = torch.optim.Adam(model.parameters(), lr=_FINE_TUNE_LEARNING_RATE)
    model.train()
    for _ in range(steps):
        pick = rng.integers(0, inputs.shape[0], size=batch_size)
        corner_tensor = torch.from_numpy(corners[pick])
        present = torch.ones(batch_size)
        optimizer.zero_grad()
        pred_heatmaps, present_logits = model(torch.from_numpy(inputs[pick]).unsqueeze(1))
        loss = corner_loss(
            pred_heatmaps,
            present_logits,
            heatmap_targets(corner_tensor, present),
            present,
            corner_tensor,
        )
        loss.backward()
        optimizer.step()
    mae, _, _, _ = _evaluate(model, pool, val_samples, batch_size, seed)
    return mae


def train_model(
    train_samples: int,
    val_samples: int,
    epochs: int,
    batch_size: int = _BATCH_SIZE,
    seed: int = _SEED,
    workers: int = _RENDER_WORKERS,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    fine_tune_steps: int = _FINE_TUNE_STEPS,
    synthetic_path: Path | None = None,
) -> tuple[KeybedNet, float]:
    torch.manual_seed(seed)
    model = KeybedNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=_LEARNING_RATE)
    steps_per_epoch = math.ceil(train_samples / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * epochs, eta_min=_MIN_LEARNING_RATE
    )
    best_mae = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    pool = Pool(processes=workers) if workers > 0 else None
    try:
        for epoch in range(epochs):
            train_loss = _train_epoch(
                model, optimizer, scheduler, pool, train_samples, batch_size, seed, epoch
            )
            mae, presence_accuracy, predicted_std, target_std = _evaluate(
                model, pool, val_samples, batch_size, seed
            )
            print(
                f"epoch {epoch + 1}/{epochs} train_loss {train_loss:.4f} "
                f"val_mae_px {mae:.2f} presence_acc {presence_accuracy:.3f} "
                f"pred_std_px {predicted_std:.1f} tgt_std_px {target_std:.1f}",
                flush=True,
            )
            if mae < best_mae:
                best_mae = mae
                best_state = {
                    name: tensor.detach().clone() for name, tensor in model.state_dict().items()
                }
        final_mae = best_mae
        if best_state is not None:
            model.load_state_dict(best_state)
        # synthetic-only weights separate renderer quality from clip memorisation
        if synthetic_path is not None:
            synthetic_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), synthetic_path)
        items = _fine_tune_items(load_frames(frames_dir))
        if items is not None and fine_tune_steps > 0:
            before, _, _, _ = _evaluate(model, pool, val_samples, batch_size, seed)
            print(
                f"fine-tune {items[0].shape[0]} real rec frames, "
                f"{fine_tune_steps} steps @ lr {_FINE_TUNE_LEARNING_RATE}",
                flush=True,
            )
            final_mae = _fine_tune(
                model, items, pool, val_samples, batch_size, seed, fine_tune_steps
            )
            print(
                f"fine-tune val_mae_px before {before:.2f} after {final_mae:.2f}",
                flush=True,
            )
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return model, final_mae


def fine_tune_from(
    synthetic_path: Path,
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    steps: int = _FINE_TUNE_STEPS,
    val_samples: int = _DEFAULT_VAL_SAMPLES,
    batch_size: int = _BATCH_SIZE,
    seed: int = _SEED,
) -> tuple[KeybedNet, float]:
    model = load_model(synthetic_path)
    items = _fine_tune_items(load_frames(frames_dir))
    if items is None:
        raise ValueError(f"no rec frames to fine-tune on in {frames_dir}")
    before, _, _, _ = _evaluate(model, None, val_samples, batch_size, seed)
    mae = _fine_tune(model, items, None, val_samples, batch_size, seed, steps)
    print(f"fine-tune val_mae_px before {before:.2f} after {mae:.2f}", flush=True)
    return model, mae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="train the keybed corner detector on synthetic renders, "
        "then fine-tune on real rec frames"
    )
    parser.add_argument("--train-samples", type=int, default=_DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=_DEFAULT_VAL_SAMPLES)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument(
        "--fine-tune-from",
        type=Path,
        default=None,
        help="skip synthetic training and fine-tune this checkpoint on the rec frames",
    )
    args = parser.parse_args()
    if args.fine_tune_from is not None:
        model, mae = fine_tune_from(args.fine_tune_from)
        DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
        print(f"saved {DEFAULT_MODEL_PATH} val_mae_px {mae:.2f}")
        return
    model, mae = train_model(
        args.train_samples,
        args.val_samples,
        args.epochs,
        synthetic_path=DEFAULT_SYNTHETIC_MODEL_PATH,
    )
    DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), DEFAULT_MODEL_PATH)
    print(f"saved {DEFAULT_MODEL_PATH} val_mae_px {mae:.2f}")


if __name__ == "__main__":
    main()
