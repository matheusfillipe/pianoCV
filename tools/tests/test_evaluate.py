import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from kvt.dataset import Frame, extract
from kvt.dataset import main as dataset_main
from kvt.evaluate import evaluate_frame, fine_tuned_frames, main, net_detector, run
from kvt.model import KeybedNet
from kvt.render import render_sample

IMAGE_SIZE = (320, 240)
KEYBED = (60, 80, 260, 104)
BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
BLACK_WIDTH = 0.58
WHITE_COUNT = 52
KEYBED_DEPTH = 6.38


def _render_keybed_snap(path: Path) -> None:
    image = np.full((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), 25, dtype=np.uint8)
    x0, y0, x1, y1 = KEYBED
    image[y0:y1, x0:x1] = 235
    width = x1 - x0
    bar_bottom = y0 + round((y1 - y0) * 0.6)
    for octave in range(7):
        for offset in BLACK_OFFSETS:
            u0 = 2.0 + 7 * octave + offset
            u1 = u0 + BLACK_WIDTH
            if u1 > WHITE_COUNT:
                continue
            bx0 = x0 + round(u0 / WHITE_COUNT * width)
            bx1 = x0 + round(u1 / WHITE_COUNT * width)
            image[y0:bar_bottom, bx0:bx1] = 15
    cv2.imwrite(str(path), image)


def _write_sidecar(path: Path) -> None:
    x0, y0, x1, y1 = KEYBED
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    sidecar = {
        "kind": "snap",
        "corners": [{"x": x / IMAGE_SIZE[0], "y": y / IMAGE_SIZE[1]} for x, y in corners],
        "imageWidth": IMAGE_SIZE[0],
        "imageHeight": IMAGE_SIZE[1],
    }
    path.write_text(json.dumps(sidecar))


def _make_frames(frames_dir: Path) -> None:
    recordings_dir = frames_dir.parent / "recordings"
    recordings_dir.mkdir(parents=True)
    _render_keybed_snap(recordings_dir / "snap-eval.png")
    _write_sidecar(recordings_dir / "snap-eval.json")
    extract(recordings_dir, frames_dir)


def test_run_locks_synthetic_frames_and_writes_previews(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    out_dir = tmp_path / "out"
    _make_frames(frames_dir)
    results = run(frames_dir, out_dir)
    assert len(results) == 1
    result = results[0]
    assert result.locked
    assert result.success
    assert result.mean_error_px is not None and result.mean_error_px <= 15.0
    assert (out_dir / "snap-eval.png").is_file()


def test_evaluate_frame_reports_miss_on_noise(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    noise = rng.integers(0, 256, size=(480, 640, 3)).astype(np.uint8)
    image_path = tmp_path / "noise.png"
    cv2.imwrite(str(image_path), noise)
    frame = Frame(
        image_path=image_path,
        corners_px=np.zeros((4, 2)),
        source_stem="noise",
        kind="snap",
    )
    result = evaluate_frame(frame)
    assert not result.locked
    assert result.quad_px is None
    assert result.mean_error_px is None


def test_evaluate_frame_without_ground_truth_still_locks(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    _make_frames(frames_dir)
    frame = Frame(
        image_path=frames_dir / "snap-eval.png",
        corners_px=None,
        source_stem="snap-eval",
        kind="snap",
    )
    result = evaluate_frame(frame)
    assert result.locked
    assert not result.success
    assert result.mean_error_px is None


def test_evaluate_frame_with_net_detector(tmp_path: Path) -> None:
    sample = render_sample(np.random.default_rng(7))
    image_path = tmp_path / "synthetic.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR))
    frame = Frame(
        image_path=image_path,
        corners_px=sample.quad_px,
        source_stem="synthetic",
        kind="snap",
    )
    result = evaluate_frame(frame, "net", net_detector(KeybedNet()))
    height, width = sample.image.shape[:2]
    if result.quad_px is None:
        assert not result.locked
        assert result.mean_error_px is None
    else:
        assert result.locked
        assert np.all(result.quad_px[:, 0] >= 0.0)
        assert np.all(result.quad_px[:, 0] <= width)
        assert np.all(result.quad_px[:, 1] >= 0.0)
        assert np.all(result.quad_px[:, 1] <= height)


def test_evaluate_frame_raises_on_unreadable_image(tmp_path: Path) -> None:
    frame = Frame(
        image_path=tmp_path / "missing.png",
        corners_px=None,
        source_stem="x",
        kind="snap",
    )
    with pytest.raises(ValueError, match="cannot read frame"):
        evaluate_frame(frame)


def test_fine_tuned_frames_flags_snapshots_reusing_a_clip_quad(tmp_path: Path) -> None:
    x0, y0, x1, y1 = KEYBED
    quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    frames = [
        Frame(tmp_path / "a.png", quad.copy(), "clip", "rec"),
        Frame(tmp_path / "b.png", quad.copy(), "twin", "snap"),
        Frame(tmp_path / "c.png", quad + 40.0, "other", "snap"),
        Frame(tmp_path / "d.png", None, "unlabelled", "snap"),
    ]
    assert fine_tuned_frames(frames) == [True, True, False, False]


def test_run_marks_held_out_frames(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    out_dir = tmp_path / "out"
    _make_frames(frames_dir)
    results = run(frames_dir, out_dir)
    assert [result.trained for result in results] == [False]


def test_run_filters_repeatable_source_stems(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    out_dir = tmp_path / "out"
    _make_frames(frames_dir)
    image_path = frames_dir / "snap-eval.png"
    corners = np.array([[60, 80], [260, 80], [260, 104], [60, 104]])
    frames = [
        Frame(image_path, corners, "keep", "snap"),
        Frame(image_path, corners, "drop", "snap"),
    ]
    with patch("kvt.evaluate.load_frames", return_value=frames):
        results = run(frames_dir, out_dir, source_stems={"keep"})
    assert [result.source_stem for result in results] == ["keep"]


def test_dataset_main_with_explicit_dirs(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _render_keybed_snap(recordings_dir / "snap-main.png")
    _write_sidecar(recordings_dir / "snap-main.json")
    argv = [
        "dataset",
        "--recordings-dir",
        str(recordings_dir),
        "--frames-dir",
        str(frames_dir),
        "--gemini-dir",
        str(tmp_path / "gemini"),
    ]
    with patch.object(sys, "argv", argv):
        dataset_main()
    assert (frames_dir / "labels.json").is_file()


def test_evaluate_main_with_explicit_dirs(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    out_dir = tmp_path / "out"
    _make_frames(frames_dir)
    argv = [
        "evaluate",
        "--frames-dir",
        str(frames_dir),
        "--out-dir",
        str(out_dir),
    ]
    with patch.object(sys, "argv", argv):
        main()
    assert (out_dir / "snap-eval.png").is_file()
