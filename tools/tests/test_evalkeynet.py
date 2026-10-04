import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.evalkeynet import (
    camera_elevation,
    centroid,
    label_points,
    main,
    parabola,
    predicted,
    score,
    score_by_view,
)
from pianocv.keymatch import load_synth_keys
from pianocv.keynet import (
    CHANNELS,
    TRACK_HEIGHT,
    TRACK_MARGIN_ACROSS,
    TRACK_MARGIN_ALONG,
    TRACK_WIDTH,
    Crop,
    heatmap_targets,
    keypoints,
    oriented_crop,
    rectified_crop,
)
from pianocv.keynetmodel import KeyNet, export_keynet_onnx

_WIDTH, _HEIGHT = 640, 480


def _gaussian(offset: tuple[float, float]) -> np.ndarray:
    ys, xs = np.mgrid[0:9, 0:9]
    return np.asarray(np.exp(-((xs - 4 - offset[1]) ** 2 + (ys - 4 - offset[0]) ** 2) / 2.0))


def test_the_parabola_finds_a_gaussian_centre_the_centroid_falls_short_of() -> None:
    channel = _gaussian((0.3, -0.2))
    assert parabola(channel, None, 4, 4) == pytest.approx((4.3, 3.8), abs=1e-6)
    row, col = centroid(channel, None, 4, 4)
    assert abs(row - 4) < 0.25
    assert abs(col - 4) < 0.15


def _write_frame(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

    def fraction(points: list[tuple[float, float]]) -> list[dict[str, float]]:
        return [{"x": x / _WIDTH, "y": y / _HEIGHT} for x, y in points]

    whites = [
        {
            "pitch": 60 + 2 * i,
            "black": False,
            "top": fraction(
                [(64 + 64 * i, 150), (128 + 64 * i, 150), (128 + 64 * i, 330), (64 + 64 * i, 330)]
            ),
            "front": None,
        }
        for i in range(8)
    ]
    cv2.imwrite(str(path / "rec-a-f0.png"), np.full((_HEIGHT, _WIDTH, 3), 90, np.uint8))
    sidecar = {
        "imageWidth": _WIDTH,
        "imageHeight": _HEIGHT,
        "kind": "real-keys",
        "corners": fraction([(64, 150), (576, 150), (576, 330), (64, 330)]),
        "keys": whites,
    }
    (path / "rec-a-f0.json").write_text(json.dumps(sidecar))


def test_keypoints_past_the_frame_edge_are_not_scored(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    sidecar = tmp_path / "rec-a-f0.json"
    data = json.loads(sidecar.read_text())
    for key in data["keys"]:
        key["top"] = [{"x": p["x"] + 0.5, "y": p["y"]} for p in key["top"]]
    data["corners"] = [{"x": p["x"] + 0.5, "y": p["y"]} for p in data["corners"]]
    sidecar.write_text(json.dumps(data))
    frames = load_synth_keys(tmp_path)
    exact = _labelled_heat(tmp_path, 0.0)
    result = score(frames, lambda _: (exact[None], None), parabola)
    points = keypoints(frames[0])
    assert points is not None
    seen = int((points.gaps[:, 0] < _WIDTH).sum())
    assert 0 < result.groups["gaps"].labels == seen < len(points.gaps)


def test_label_points_lists_every_key_low_to_high_in_frame_pixels(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    exported = label_points(load_synth_keys(tmp_path))["rec-a-f0"]
    assert exported["whiteKeys"] == 8
    gaps = np.array(exported["gaps"])
    assert np.allclose(gaps[:, 0], [128 + 64 * i for i in range(7)])
    assert np.allclose(gaps[:, 1], 330)


def _write_fixed(directory: Path, corners: list[list[float] | None]) -> None:
    directory.mkdir()
    (directory / "rec-a-f0.json").write_text(
        json.dumps(
            {
                "width": _WIDTH,
                "height": _HEIGHT,
                "whiteKeys": 8,
                "phase": "E",
                "corners": corners,
                "gaps": [[128 + 64 * i, 330] if i != 2 else None for i in range(7)],
                "blackLow": [],
                "blackHigh": [],
                "blackTopLow": [],
                "blackTopHigh": [],
            }
        )
    )


def test_corrected_labels_replace_the_derived_ones_and_a_hidden_point_is_not_scored(
    tmp_path: Path,
) -> None:
    _write_frame(tmp_path)
    fixed = tmp_path / "fixed"
    _write_fixed(fixed, [[64, 150], [576, 150], [576, 330], [64, 330]])
    frames = load_synth_keys(tmp_path)
    exact = _labelled_heat(tmp_path, 0.0)
    result = score(frames, lambda _: (exact[None], None), parabola, fixed_dir=fixed)
    assert result.groups["gaps"].labels == 6
    exported = label_points(frames, fixed)["rec-a-f0"]
    assert exported["gaps"][2] is None  # type: ignore[index]
    json.dumps(exported, allow_nan=False)


def test_a_frame_with_a_hidden_keybed_corner_is_left_out(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    fixed = tmp_path / "fixed"
    _write_fixed(fixed, [[64, 150], None, [576, 330], [64, 330]])
    exact = _labelled_heat(tmp_path, 0.0)
    result = score(
        load_synth_keys(tmp_path), lambda _: (exact[None], None), parabola, fixed_dir=fixed
    )
    assert result.frames == 0


def test_a_hidden_corner_is_not_scored_and_does_not_fail_the_frame(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    sidecar = tmp_path / "rec-a-f0.json"
    data = json.loads(sidecar.read_text())
    sidecar.write_text(json.dumps(data | {"cornerVisible": [False, True, True, True]}))
    exact = _labelled_heat(tmp_path, 0.0)
    exact[0] = 0.0
    result = score(load_synth_keys(tmp_path), lambda _: (exact[None], None), parabola)
    assert result.groups["corners"].labels == 3
    assert result.failure_rate == 0.0


def _labelled_heat(path: Path, shift_px: float, rectified: bool = False) -> np.ndarray:
    frame = load_synth_keys(path)[0]
    points = keypoints(frame)
    assert points is not None
    make_crop = rectified_crop if rectified else oriented_crop
    crop = make_crop(
        points.corners, TRACK_WIDTH, TRACK_HEIGHT, TRACK_MARGIN_ALONG, TRACK_MARGIN_ACROSS
    )
    moved = Crop(to_crop=crop.to_crop + np.array([[0, 0, shift_px], [0, 0, 0], [0, 0, 0]]))
    heat, _ = heatmap_targets(points, moved, (TRACK_HEIGHT, TRACK_WIDTH), frame.image_size, None)
    assert heat.shape[0] == CHANNELS
    return heat


def test_a_perfect_prediction_scores_no_error_and_a_shifted_one_scores_the_shift(
    tmp_path: Path,
) -> None:
    _write_frame(tmp_path)
    frames = load_synth_keys(tmp_path)
    exact = _labelled_heat(tmp_path, 0.0)
    perfect = score(frames, lambda _: (exact[None], None), parabola)
    assert perfect.failure_rate == 0.0
    assert perfect.groups["gaps"].recall == 1.0
    assert perfect.groups["gaps"].median_keys < 0.01

    shifted = _labelled_heat(tmp_path, 20.0)
    off = score(frames, lambda _: (shifted[None], None), parabola)
    points = keypoints(frames[0])
    assert points is not None
    crop = oriented_crop(
        points.corners, TRACK_WIDTH, TRACK_HEIGHT, TRACK_MARGIN_ALONG, TRACK_MARGIN_ACROSS
    )
    key_crop_px = 64 * float(np.linalg.norm(crop.to_crop[0, :2]))
    assert off.groups["gaps"].median_keys == pytest.approx(20.0 / key_crop_px, rel=0.05)


def test_a_rectified_perfect_prediction_scores_no_error(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    exact = _labelled_heat(tmp_path, 0.0, rectified=True)
    result = score(
        load_synth_keys(tmp_path), lambda _: (exact[None], None), parabola, rectified=True
    )
    assert result.frames == 1
    assert result.failure_rate == 0.0
    assert result.groups["gaps"].recall == 1.0
    assert result.groups["gaps"].median_keys < 0.01


def test_the_command_scores_an_export_and_writes_the_scores_and_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    _write_frame(frames)
    model = tmp_path / "keynet.onnx"
    export_keynet_onnx(KeyNet(pretrained=False), str(model))
    scores = tmp_path / "scores.json"
    points = tmp_path / "points.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evalkeynet",
            "--frames",
            str(frames),
            "--clips",
            "rec-a",
            "--model",
            str(model),
            "--decoder",
            "offset",
            "--by-view",
            "--json",
            str(scores),
            "--points-out",
            str(points),
        ],
    )
    main()
    result = json.loads(scores.read_text())["keynet.onnx offset"]
    assert result["frames"] == 1
    assert "keynet.onnx offset elevation low" in json.loads(scores.read_text())
    assert set(result["groups"]) == {"corners", "gaps", "black", "tops", "back", "backtops"}
    assert "rec-a-f0" in json.loads(points.read_text())


def test_the_offset_decoder_moves_the_peak_cell_by_the_predicted_offset() -> None:
    channel = _gaussian((0.0, 0.0))
    offset = np.zeros((2, 9, 9), np.float32)
    offset[0, 4, 4] = 0.25
    offset[1, 4, 4] = -0.4
    assert predicted(channel, offset, 4, 4) == pytest.approx((3.6, 4.25))
    assert predicted(channel, None, 4, 4) == pytest.approx(centroid(channel, None, 4, 4))


def test_elevation_buckets_follow_the_keybed_depth_against_its_width(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    points = keypoints(load_synth_keys(tmp_path)[0])
    assert points is not None
    depth_for_90 = 150.0 / (23.5 * 8) * 512

    def at(depth: float) -> float:
        corners = np.array([[0, 0], [512, 0], [512, depth], [0, depth]], np.float64)
        return camera_elevation(replace(points, corners=corners))

    assert at(depth_for_90 * 0.2) < 30
    assert 30 < at(depth_for_90 * 0.7) < 60
    assert at(depth_for_90) == pytest.approx(90.0)


def test_by_view_scores_each_board_size_and_elevation_bucket(tmp_path: Path) -> None:
    _write_frame(tmp_path)
    exact = _labelled_heat(tmp_path, 0.0)
    frames = load_synth_keys(tmp_path)
    views = score_by_view(frames, lambda _: (exact[None], None), parabola)
    assert set(views) == {"all", "8 white keys", "elevation low"}
    assert views["8 white keys"].frames == views["all"].frames == 1
    assert views["elevation low"].groups["gaps"].recall == 1.0
