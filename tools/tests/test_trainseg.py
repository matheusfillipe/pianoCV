import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pianocv.dataset import Frame
from pianocv.model import MASK_SIZE, KeybedSegNet, load_seg_model, mask_targets, predict_mask
from pianocv.trainseg import frame_items, train_seg

QUAD = [[16.0, 12.0], [48.0, 12.0], [48.0, 36.0], [16.0, 36.0]]


def _write_frames_dir(frames_dir: Path, count: int = 3) -> None:
    frames_dir.mkdir(parents=True)
    labels: dict[str, dict[str, object]] = {"extracted": {}, "frames": {}}
    for index in range(count):
        name = f"standin.{index:06d}.png"
        image = np.full((48, 64, 3), 60, dtype=np.uint8)
        cv2.fillPoly(image, [np.array(QUAD, dtype=np.int32)], (200, 200, 200))
        cv2.imwrite(str(frames_dir / name), image)
        labels["frames"][name] = {
            "corners_px": QUAD,
            "source_stem": "standin",
            "kind": "rec",
        }
    (frames_dir / "labels.json").write_text(json.dumps(labels))


def test_mask_targets_resamples_to_the_model_grid() -> None:
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:300, 50:600] = 255
    targets = mask_targets(mask[None])
    assert targets.shape == (1, MASK_SIZE, MASK_SIZE)
    assert float(targets.min()) >= 0.0 and float(targets.max()) <= 1.0
    assert float(targets.mean()) > 0.1


def test_real_items_builds_one_target_per_labelled_frame(tmp_path: Path) -> None:
    image_path = tmp_path / "a.png"
    cv2.imwrite(str(image_path), np.zeros((48, 64, 3), dtype=np.uint8))
    frames = [
        Frame(image_path, np.array(QUAD), "a", "rec"),
        Frame(tmp_path / "missing.png", None, "b", "rec"),
    ]
    items = frame_items(frames)
    assert items is not None
    inputs, masks = items
    assert inputs.shape[0] == 1
    assert masks.shape == (1, MASK_SIZE, MASK_SIZE)


def test_real_items_returns_none_without_labels() -> None:
    assert frame_items([Frame(Path("x.png"), None, "x", "rec")]) is None


def test_train_seg_runs_a_small_schedule(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames_dir(frames_dir)
    model, iou = train_seg(
        train_samples=16,
        val_samples=16,
        epochs=1,
        batch_size=8,
        workers=0,
        frames_dir=frames_dir,
    )
    assert 0.0 <= iou <= 1.0
    assert not model.training


def test_train_seg_without_real_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_frames_dir(frames_dir)
    _, iou = train_seg(
        train_samples=16,
        val_samples=16,
        epochs=1,
        batch_size=8,
        workers=0,
        frames_dir=frames_dir,
        real_fraction=0.0,
    )
    assert 0.0 <= iou <= 1.0


def test_predict_mask_matches_the_image_size(tmp_path: Path) -> None:
    path = tmp_path / "seg.pt"
    torch.save(KeybedSegNet().state_dict(), path)
    model = load_seg_model(path)
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    probability = predict_mask(model, image)
    assert probability.shape == (240, 320)
    assert float(probability.min()) >= 0.0 and float(probability.max()) <= 1.0
