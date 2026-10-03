import json
from pathlib import Path

import cv2
import numpy as np
import torch

from pianocv.keymatch import (
    PEAK_TOLERANCE_PX,
    STRIP_HEIGHT,
    STRIP_WIDTH,
    KeyMatchNet,
    PerturbConfig,
    SidecarKey,
    SynthKeysFrame,
    channel_metrics,
    compute_targets,
    find_peaks,
    load_synth_keys,
    match_peaks,
    perturb_quad,
)

# an axis-aligned quad has no projective skew, so a key at board unit `u` always lands at
# strip_x = (u / BOARD_WIDTH) * (STRIP_WIDTH - 1), which lets every test assert exact geometry
FAR_LEFT, FAR_RIGHT = (64.0, 96.0), (576.0, 96.0)
NEAR_LEFT, NEAR_RIGHT = (64.0, 384.0), (576.0, 384.0)
QUAD_PX = np.array([FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT])
BOARD_WIDTH = 3.0


def _strip_x(u: float) -> float:
    return (u / BOARD_WIDTH) * (STRIP_WIDTH - 1)


def _face(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
    def point(u: float, v: float) -> tuple[float, float]:
        x = FAR_LEFT[0] + (u / BOARD_WIDTH) * (FAR_RIGHT[0] - FAR_LEFT[0])
        y = FAR_LEFT[1] + v * (NEAR_LEFT[1] - FAR_LEFT[1])
        return x, y

    return np.array([point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)])


def test_finds_every_edge_when_perspective_tilts_depth_along_the_keyboard() -> None:
    # the near edge sits far to the right of the far edge, as in an oblique view, so each key's
    # front corners lie further along the keyboard than its back corners
    oblique = np.array([(64.0, 96.0), (420.0, 60.0), (600.0, 300.0), (200.0, 400.0)])
    board = np.array([(0, 0), (BOARD_WIDTH, 0), (BOARD_WIDTH, 1), (0, 1)], dtype=np.float32)
    to_image = cv2.getPerspectiveTransform(board, oblique.astype(np.float32))

    def face(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
        corners = np.array([[(u0, v0), (u1, v0), (u1, v1), (u0, v1)]], dtype=np.float32)
        return np.asarray(cv2.perspectiveTransform(corners, to_image)[0], dtype=np.float64)

    white = [
        SidecarKey(pitch=60 + 2 * i, black=False, top=face(i, i + 1, 0.0, 1.0), front=None)
        for i in range(3)
    ]
    black = [
        SidecarKey(pitch=p, black=True, top=face(b - 0.3, b + 0.3, 0.05, 0.35), front=None)
        for p, b in ((61, 1.0), (63, 2.0))
    ]
    frame = SynthKeysFrame(Path("unused.png"), oblique, [*white, *black], (640, 480))
    targets = compute_targets(frame, oblique)
    assert np.allclose(sorted(targets.white_positions), [_strip_x(1.0), _strip_x(2.0)], atol=0.5)
    assert np.allclose(
        sorted(targets.black_left_positions), [_strip_x(0.7), _strip_x(1.7)], atol=0.5
    )
    assert np.allclose(
        sorted(targets.black_right_positions), [_strip_x(1.3), _strip_x(2.3)], atol=0.5
    )


def _three_white_two_black_frame() -> SynthKeysFrame:
    white = [
        SidecarKey(
            pitch=60 + 2 * i, black=False, top=_face(float(i), float(i + 1), 0.0, 1.0), front=None
        )
        for i in range(3)
    ]
    black = [
        SidecarKey(
            pitch=pitch,
            black=True,
            top=_face(boundary - 0.3, boundary + 0.3, 0.05, 0.35),
            front=_face(boundary - 0.3, boundary + 0.3, 0.35, 0.42),
        )
        for pitch, boundary in ((61, 1.0), (63, 2.0))
    ]
    return SynthKeysFrame(
        image_path=Path("unused.png"),
        corners_px=QUAD_PX,
        keys=[*white, *black],
        image_size=(640, 480),
    )


def test_white_and_black_targets_land_where_geometry_says() -> None:
    frame = _three_white_two_black_frame()
    targets = compute_targets(frame, QUAD_PX)

    assert targets.white_positions == sorted(targets.white_positions)
    assert np.allclose(sorted(targets.white_positions), [_strip_x(1.0), _strip_x(2.0)], atol=1e-6)
    assert np.allclose(
        sorted(targets.black_left_positions), sorted([_strip_x(0.7), _strip_x(1.7)]), atol=1e-6
    )
    assert np.allclose(
        sorted(targets.black_right_positions), sorted([_strip_x(1.3), _strip_x(2.3)]), atol=1e-6
    )
    assert targets.heatmaps.shape == (3, STRIP_WIDTH)
    assert targets.heatmaps.dtype == np.float32


def test_targets_cut_off_by_a_cropped_quad_produce_no_peaks() -> None:
    frame = _three_white_two_black_frame()
    # a quad spanning only the board's first 1.5 units crops the second white-white boundary
    # and the second black key off entirely, so neither should produce a target
    cropped_far_right = (
        FAR_LEFT[0] + (1.5 / BOARD_WIDTH) * (FAR_RIGHT[0] - FAR_LEFT[0]),
        FAR_RIGHT[1],
    )
    cropped_near_right = (cropped_far_right[0], NEAR_RIGHT[1])
    cropped_quad = np.array([FAR_LEFT, cropped_far_right, cropped_near_right, NEAR_LEFT])

    targets = compute_targets(frame, cropped_quad)

    assert len(targets.white_positions) == 1
    assert len(targets.black_left_positions) == 1
    assert len(targets.black_right_positions) == 1


def test_load_synth_keys_round_trips_a_sidecar(tmp_path: Path) -> None:
    frame = _three_white_two_black_frame()
    assert frame.corners_px is not None
    width, height = 640, 480
    scale = np.array([float(width), float(height)])

    def face_fraction(face_px: np.ndarray) -> list[dict[str, float]]:
        return [{"x": float(x), "y": float(y)} for x, y in (face_px / scale)]

    sidecar = {
        "corners": face_fraction(frame.corners_px),
        "imageWidth": width,
        "imageHeight": height,
        "board": {"keys": 5, "whiteKeys": 3, "lowestPitch": 60, "lowestNote": "C4"},
        "keys": [
            {
                "pitch": key.pitch,
                "black": key.black,
                "top": face_fraction(key.top),
                "front": None if key.front is None else face_fraction(key.front),
            }
            for key in frame.keys
        ],
    }
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((height, width, 3), dtype=np.uint8))
    (tmp_path / "a.json").write_text(json.dumps(sidecar))

    loaded = load_synth_keys(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].corners_px is not None
    assert np.allclose(loaded[0].corners_px, frame.corners_px, atol=1e-3)
    assert sorted(key.pitch for key in loaded[0].keys) == sorted(key.pitch for key in frame.keys)


def test_perturb_quad_keeps_a_valid_non_degenerate_quad() -> None:
    rng = np.random.default_rng(0)
    original_area = float(cv2.contourArea(QUAD_PX.astype(np.float32)))
    for _ in range(500):
        perturbed = perturb_quad(QUAD_PX, rng)
        area = float(cv2.contourArea(perturbed.astype(np.float32)))
        assert area > 0.1 * original_area


def test_perturb_quad_leaves_the_configured_fraction_exactly_unperturbed() -> None:
    rng = np.random.default_rng(1)
    config = PerturbConfig(unperturbed_fraction=0.15)
    exact = sum(np.array_equal(perturb_quad(QUAD_PX, rng, config), QUAD_PX) for _ in range(4000))
    assert abs(exact / 4000 - 0.15) < 0.03


def test_perturb_quad_always_unperturbed_returns_the_input_unchanged() -> None:
    rng = np.random.default_rng(2)
    config = PerturbConfig(unperturbed_fraction=1.0)
    for _ in range(20):
        assert np.array_equal(perturb_quad(QUAD_PX, rng, config), QUAD_PX)


def test_model_output_shape_and_parameter_budget() -> None:
    model = KeyMatchNet()
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count < 300_000

    output = model(torch.zeros(2, 3, STRIP_HEIGHT, STRIP_WIDTH))
    assert output.shape == (2, 3, STRIP_WIDTH)


def test_find_peaks_recovers_well_separated_local_maxima() -> None:
    values = np.zeros(STRIP_WIDTH, dtype=np.float32)
    values[100] = 0.9
    values[400] = 0.8
    peaks = find_peaks(values, threshold=0.5, min_distance_px=3.0)
    assert peaks == [100.0, 400.0]


def test_match_peaks_counts_true_and_false_positives_and_negatives() -> None:
    true_positives, false_positives, false_negatives, errors = match_peaks(
        predicted=[10.0, 20.0, 500.0], truth=[11.0, 21.0], tolerance_px=2.0
    )
    assert (true_positives, false_positives, false_negatives) == (2, 1, 0)
    assert errors == [1.0, 1.0]


def test_channel_metrics_precision_recall_and_mean_error() -> None:
    metrics = channel_metrics(
        predicted=[10.0, 20.0], truth=[11.0, 40.0], tolerance_px=PEAK_TOLERANCE_PX
    )
    assert metrics.precision == 0.5
    assert metrics.recall == 0.5
    assert metrics.mean_error_px == 1.0
