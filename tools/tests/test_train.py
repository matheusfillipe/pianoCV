import json
import math
from pathlib import Path

import cv2
import numpy as np

from pianocv.dataset import Frame
from pianocv.train import _fine_tune_items, train_model

STANDIN_QUAD = [[16.0, 12.0], [48.0, 12.0], [48.0, 36.0], [16.0, 36.0]]


def _write_standin_frames_dir(frames_dir: Path, frames: int = 3) -> None:
    frames_dir.mkdir(parents=True)
    labels: dict[str, dict[str, object]] = {"extracted": {}, "frames": {}}
    corners: list[list[float]] = [list(point) for point in STANDIN_QUAD]
    for index in range(frames):
        name = f"standin.{index:06d}.png"
        image = np.full((48, 64, 3), 60, dtype=np.uint8)
        cv2.fillPoly(image, [np.array(corners, dtype=np.int32)], 200)
        cv2.imwrite(str(frames_dir / name), image)
        labels["frames"][name] = {
            "corners_px": corners,
            "source_stem": "standin",
            "kind": "rec",
        }
    (frames_dir / "labels.json").write_text(json.dumps(labels))


def test_train_model_runs_small_schedule() -> None:
    model, mae = train_model(
        train_samples=32,
        val_samples=32,
        epochs=2,
        batch_size=16,
        workers=0,
        fine_tune_steps=0,
    )
    assert math.isfinite(mae)
    assert not model.training


def test_train_model_with_render_workers() -> None:
    _, mae = train_model(
        train_samples=8,
        val_samples=8,
        epochs=1,
        batch_size=8,
        workers=2,
        fine_tune_steps=0,
    )
    assert math.isfinite(mae)


def test_train_model_fine_tunes_on_standin_rec_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _write_standin_frames_dir(frames_dir)
    synthetic_path = tmp_path / "models" / "synthetic.pt"
    model, mae = train_model(
        train_samples=16,
        val_samples=16,
        epochs=1,
        batch_size=8,
        workers=0,
        frames_dir=frames_dir,
        fine_tune_steps=2,
        synthetic_path=synthetic_path,
    )
    assert math.isfinite(mae)
    assert not model.training
    assert synthetic_path.is_file()


def test_train_model_skips_fine_tune_without_rec_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "labels.json").write_text(json.dumps({"extracted": {}, "frames": {}}))
    _, mae = train_model(
        train_samples=8,
        val_samples=8,
        epochs=1,
        batch_size=8,
        workers=0,
        frames_dir=frames_dir,
        fine_tune_steps=2,
    )
    assert math.isfinite(mae)


def test_fine_tune_items_keep_only_rec_frames(tmp_path: Path) -> None:
    image_path = tmp_path / "rec.png"
    cv2.imwrite(str(image_path), np.zeros((48, 64, 3), dtype=np.uint8))
    corners = np.array(STANDIN_QUAD)
    frames = [
        Frame(tmp_path / "missing.png", None, "a", "rec"),
        Frame(tmp_path / "b.png", corners, "b", "snap"),
        Frame(tmp_path / "c.png", corners, "c", "gemini"),
        Frame(image_path, corners, "d", "rec"),
    ]
    items = _fine_tune_items(frames)
    assert items is not None
    inputs, corners = items
    assert inputs.shape == (1, 288, 288)
    assert corners.shape == (1, 8)


def test_fine_tune_items_return_none_without_rec_frames() -> None:
    frames = [Frame(Path("b.png"), np.zeros((4, 2)), "b", "snap")]
    assert _fine_tune_items(frames) is None
