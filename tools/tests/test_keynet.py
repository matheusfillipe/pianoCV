import json
from pathlib import Path

import cv2
import numpy as np

from pianocv.keymatch import SidecarKey, SynthKeysFrame, parse_synth_keys_sidecar
from pianocv.keynet import (
    CHANNELS,
    SEARCH_SIZE,
    TRACK_HEIGHT,
    TRACK_MARGIN_ACROSS,
    TRACK_MARGIN_ALONG,
    TRACK_WIDTH,
    Crop,
    KeyPoints,
    fixed_path,
    frame_points,
    has_back_labels,
    heatmap_targets,
    keypoints,
    load_fixed_points,
    offset_targets,
    oriented_crop,
    presence_target,
    rectified_crop,
    squash_crop,
)
from pianocv.keyseg import CROP_HEIGHT, CROP_MARGIN_ACROSS, CROP_MARGIN_ALONG, CROP_WIDTH, crop_for

# one octave, C4 to B4: 7 white keys and the standard 2-and-3 black pattern
WHITE_PITCHES = (60, 62, 64, 65, 67, 69, 71)
BLACK_KEYS = (  # pitch, white index it starts over, offset into that white key
    (61, 0, 0.6),
    (63, 1, 0.75),
    (66, 3, 0.6),
    (68, 4, 0.63),
    (70, 5, 0.66),
)
BLACK_WIDTH = 0.56
BLACK_TOP_DEPTH = 0.55
BLACK_BASE_DEPTH = 0.63
BOARD_WIDTH = float(len(WHITE_PITCHES))

DEFAULT_CORNERS = np.array([(120.0, 120.0), (760.0, 100.0), (860.0, 480.0), (60.0, 500.0)])


def _rotated(image_corners: np.ndarray) -> np.ndarray:
    centre = image_corners.mean(axis=0)
    rotated = (image_corners - centre) @ np.array([[0.0, -1.0], [1.0, 0.0]]) + centre
    return np.asarray(rotated, dtype=np.float64)


def _mirrored(image_corners: np.ndarray) -> np.ndarray:
    flipped = image_corners.copy()
    flipped[:, 0] = -flipped[:, 0]
    return np.asarray(flipped - flipped.min(axis=0) + 50.0, dtype=np.float64)


def _board_quad(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
    return np.array([(u0, v0), (u1, v0), (u1, v1), (u0, v1)], dtype=np.float64)


def _project(points: np.ndarray, to_image: np.ndarray) -> np.ndarray:
    pts = points.reshape(1, -1, 2).astype(np.float32)
    return np.asarray(cv2.perspectiveTransform(pts, to_image)[0], dtype=np.float64)


def _homography(image_corners: np.ndarray) -> np.ndarray:
    board = np.array(
        [(0.0, 0.0), (BOARD_WIDTH, 0.0), (BOARD_WIDTH, 1.0), (0.0, 1.0)], dtype=np.float32
    )
    return cv2.getPerspectiveTransform(board, image_corners.astype(np.float32))


def _keyboard_frame(
    image_corners: np.ndarray, image_size: tuple[int, int] = (960, 540)
) -> SynthKeysFrame:
    to_image = _homography(image_corners)

    whites = [
        SidecarKey(
            pitch=pitch,
            black=False,
            top=_project(_board_quad(float(i), float(i + 1), 0.0, 1.0), to_image),
            front=None,
        )
        for i, pitch in enumerate(WHITE_PITCHES)
    ]
    blacks = []
    for pitch, base, offset in BLACK_KEYS:
        u0, u1 = base + offset, base + offset + BLACK_WIDTH
        top = _project(_board_quad(u0, u1, 0.0, BLACK_TOP_DEPTH), to_image)
        shared_low, shared_high = top[3], top[2]
        lower_high, lower_low = _project(
            np.array([(u1, BLACK_BASE_DEPTH), (u0, BLACK_BASE_DEPTH)]), to_image
        )
        front = np.array([shared_low, shared_high, lower_high, lower_low])
        blacks.append(SidecarKey(pitch=pitch, black=True, top=top, front=front))

    corners_px = _project(_board_quad(0.0, BOARD_WIDTH, 0.0, 1.0), to_image)
    return SynthKeysFrame(
        image_path=Path("unused.png"),
        corners_px=corners_px,
        keys=[*whites, *blacks],
        image_size=image_size,
    )


def test_derived_corners_match_the_sidecar_back_low_back_high_front_high_front_low_order() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    points = keypoints(frame)
    assert points is not None
    assert frame.corners_px is not None
    assert np.allclose(points.corners, frame.corners_px, atol=0.1)


def test_back_points_sit_on_the_back_edge_of_the_keys() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    points = keypoints(frame)
    assert points is not None
    to_image = _homography(DEFAULT_CORNERS)
    back_edge = _project(
        np.array([(float(i), 0.0) for i in range(1, len(WHITE_PITCHES))]), to_image
    )
    assert points.back_gaps.shape == (len(WHITE_PITCHES) - 1, 2)
    assert np.allclose(np.sort(points.back_gaps, axis=0), np.sort(back_edge, axis=0), atol=0.1)
    expected_low = _project(
        np.array([(base + offset, 0.0) for _, base, offset in BLACK_KEYS]), to_image
    )
    expected_high = _project(
        np.array([(base + offset + BLACK_WIDTH, 0.0) for _, base, offset in BLACK_KEYS]), to_image
    )
    assert np.allclose(points.black_back_low, expected_low, atol=0.1)
    assert np.allclose(points.black_back_high, expected_high, atol=0.1)


def test_a_label_file_without_back_fields_loads_them_as_hidden_rows(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_corrected(blackLow=[[1, 2]], blackHigh=[[3, 4]])))
    points = load_fixed_points(path)
    assert np.isnan(points.back_gaps).all()
    assert points.back_gaps.shape == (2, 2)
    assert points.black_back_low.shape == (1, 2)
    assert np.isnan(points.black_back_low).all()
    assert not has_back_labels(path)


def test_a_label_file_with_back_fields_loads_them(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_corrected(backGaps=[[200, 50], None])))
    assert has_back_labels(path)
    points = load_fixed_points(path)
    assert points.back_gaps[0].tolist() == [200.0, 50.0]
    assert np.isnan(points.back_gaps[1]).all()


def test_role_derivation_is_robust_to_a_mirrored_view() -> None:
    frame = _keyboard_frame(_mirrored(DEFAULT_CORNERS))
    points = keypoints(frame)
    assert points is not None
    assert frame.corners_px is not None
    assert np.allclose(points.corners, frame.corners_px, atol=0.1)


def test_role_derivation_is_robust_to_a_rotated_view() -> None:
    frame = _keyboard_frame(_rotated(DEFAULT_CORNERS))
    points = keypoints(frame)
    assert points is not None
    assert frame.corners_px is not None
    assert np.allclose(points.corners, frame.corners_px, atol=0.1)


def test_gaps_count_is_white_keys_minus_one_and_sits_on_the_shared_boundary() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    to_image = _homography(DEFAULT_CORNERS)
    points = keypoints(frame)
    assert points is not None
    assert points.gaps.shape == (len(WHITE_PITCHES) - 1, 2)
    expected = _project(np.array([(float(i), 1.0) for i in range(1, len(WHITE_PITCHES))]), to_image)
    assert np.allclose(points.gaps, expected, atol=0.1)


def test_black_corners_pair_up_per_black_key_low_and_high_side() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    to_image = _homography(DEFAULT_CORNERS)
    points = keypoints(frame)
    assert points is not None
    assert points.black_low.shape == points.black_high.shape == (len(BLACK_KEYS), 2)
    for (_, base, offset), low, high in zip(
        BLACK_KEYS, points.black_low, points.black_high, strict=True
    ):
        u0, u1 = base + offset, base + offset + BLACK_WIDTH
        expected_low = _project(np.array([(u0, BLACK_BASE_DEPTH)]), to_image)[0]
        expected_high = _project(np.array([(u1, BLACK_BASE_DEPTH)]), to_image)[0]
        assert np.allclose(low, expected_low, atol=0.1)
        assert np.allclose(high, expected_high, atol=0.1)


def test_black_top_corners_are_where_the_front_face_meets_the_key_top() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    to_image = _homography(DEFAULT_CORNERS)
    points = keypoints(frame)
    assert points is not None
    assert points.black_top_low.shape == points.black_top_high.shape == (len(BLACK_KEYS), 2)
    for (_, base, offset), low, high in zip(
        BLACK_KEYS, points.black_top_low, points.black_top_high, strict=True
    ):
        u0, u1 = base + offset, base + offset + BLACK_WIDTH
        expected_low = _project(np.array([(u0, BLACK_TOP_DEPTH)]), to_image)[0]
        expected_high = _project(np.array([(u1, BLACK_TOP_DEPTH)]), to_image)[0]
        assert np.allclose(low, expected_low, atol=0.1)
        assert np.allclose(high, expected_high, atol=0.1)


def test_keypoints_is_none_for_a_frame_with_no_keys() -> None:
    frame = SynthKeysFrame(Path("unused.png"), None, [], (640, 480))
    assert keypoints(frame) is None


def test_presence_target_is_zero_for_a_negative_and_one_for_a_fully_visible_keyboard() -> None:
    negative = SynthKeysFrame(Path("unused.png"), None, [], (640, 480))
    assert presence_target(negative) == 0.0

    frame = _keyboard_frame(DEFAULT_CORNERS, image_size=(960, 540))
    assert presence_target(frame) == 1.0


def test_presence_target_is_zero_when_the_keybed_is_mostly_out_of_frame() -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS, image_size=(960, 540))
    shifted = SynthKeysFrame(
        frame.image_path, frame.corners_px + np.array([2000.0, 0.0]), frame.keys, frame.image_size
    )
    assert presence_target(shifted) == 0.0


def test_heatmap_peaks_land_at_the_projected_cell_within_half_a_cell() -> None:
    identity = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    points = KeyPoints(
        corners=np.array([[101.4, 51.2], [400.0, 60.0], [400.0, 140.0], [101.4, 141.8]]),
        gaps=np.array([[250.0, 100.0]]),
        black_low=np.array([[300.0, 90.0]]),
        black_high=np.array([[320.0, 90.0]]),
        black_top_low=np.array([[300.0, 70.0]]),
        black_top_high=np.array([[320.0, 70.0]]),
        back_gaps=np.array([[250.0, 60.0]]),
        black_back_low=np.array([[300.0, 55.0]]),
        black_back_high=np.array([[320.0, 55.0]]),
    )
    heat, weight = heatmap_targets(
        points, identity, (TRACK_HEIGHT, TRACK_WIDTH), (TRACK_WIDTH, TRACK_HEIGHT), None
    )
    assert heat.shape == weight.shape == (CHANNELS, TRACK_HEIGHT // 2, TRACK_WIDTH // 2)
    assert np.array_equal(weight, np.ones_like(weight))

    all_points = np.vstack(
        [
            points.corners,
            points.gaps,
            points.black_low,
            points.black_high,
            points.black_top_low,
            points.black_top_high,
            points.back_gaps,
            points.black_back_low,
            points.black_back_high,
        ]
    )
    for channel, (x, y) in enumerate(all_points):
        row, col = np.unravel_index(np.argmax(heat[channel]), heat[channel].shape)
        assert abs(row - (y / 2.0 - 0.25)) <= 0.5 + 1e-6
        assert abs(col - (x / 2.0 - 0.25)) <= 0.5 + 1e-6


def test_ignore_mask_zeroes_weight_and_drops_the_peak() -> None:
    identity = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    point = np.array([200.0, 80.0])
    points = KeyPoints(
        corners=np.array([point, point, point, point]),
        gaps=np.empty((0, 2)),
        black_low=np.empty((0, 2)),
        black_high=np.empty((0, 2)),
        black_top_low=np.empty((0, 2)),
        black_top_high=np.empty((0, 2)),
        back_gaps=np.empty((0, 2)),
        black_back_low=np.empty((0, 2)),
        black_back_high=np.empty((0, 2)),
    )
    ignore = np.zeros((TRACK_HEIGHT, TRACK_WIDTH), dtype=np.uint8)
    ignore[70:90, 190:210] = 255

    heat, weight = heatmap_targets(
        points, identity, (TRACK_HEIGHT, TRACK_WIDTH), (TRACK_WIDTH, TRACK_HEIGHT), ignore
    )
    assert float(heat[0].max()) == 0.0
    assert weight[0, 40, 100] == 0.0


def test_points_outside_the_input_get_no_peak_and_the_weight_dip_is_clipped_to_the_array() -> None:
    identity = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    point = np.array([-4.0, 80.0])
    points = KeyPoints(
        corners=np.array([point, point, point, point]),
        gaps=np.empty((0, 2)),
        black_low=np.empty((0, 2)),
        black_high=np.empty((0, 2)),
        black_top_low=np.empty((0, 2)),
        black_top_high=np.empty((0, 2)),
        back_gaps=np.empty((0, 2)),
        black_back_low=np.empty((0, 2)),
        black_back_high=np.empty((0, 2)),
    )
    heat, weight = heatmap_targets(
        points, identity, (TRACK_HEIGHT, TRACK_WIDTH), (TRACK_WIDTH, TRACK_HEIGHT), None
    )
    assert float(heat[0].max()) == 0.0
    assert weight[0, 40, 0] == 0.0
    assert weight[0, 40, 1] == 0.0
    assert weight[0, 40, 5] == 1.0


def _corrected(**fields: object) -> dict[str, object]:
    return {
        "width": TRACK_WIDTH,
        "height": TRACK_HEIGHT,
        "whiteKeys": 3,
        "phase": "E",
        "corners": [[100, 50], [400, 50], [400, 140], [100, 140]],
        "gaps": [[200, 140], [300, 140]],
        "blackLow": [],
        "blackHigh": [],
        "blackTopLow": [],
        "blackTopHigh": [],
    } | fields


def test_loader_reads_a_corrected_file_and_turns_null_into_nan_rows(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(
        json.dumps(
            _corrected(corners=[[100, 50], None, [400, 140], [100, 140]], gaps=[None, [300, 140]])
        )
    )
    points = load_fixed_points(path)
    assert points.corners.shape == (4, 2)
    assert np.isnan(points.corners[1]).all()
    assert np.isnan(points.gaps[0]).all()
    assert points.gaps[1].tolist() == [300.0, 140.0]
    assert points.black_low.shape == (0, 2)


def test_a_hidden_point_gets_no_peak_and_no_crash(tmp_path: Path) -> None:
    path = tmp_path / "f.json"
    path.write_text(json.dumps(_corrected(gaps=[None, [300, 140]], blackLow=[None])))
    points = load_fixed_points(path)
    identity = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    size = (TRACK_HEIGHT, TRACK_WIDTH)
    heat, weight = heatmap_targets(points, identity, size, (TRACK_WIDTH, TRACK_HEIGHT), None)
    assert float(heat[4].max()) > 0.5
    assert float(heat[4][:, :120].max()) == 0.0
    assert float(heat[5].max()) == 0.0
    assert np.array_equal(weight, np.ones_like(weight))
    offset, near = offset_targets(points, identity, size)
    assert np.isfinite(offset).all()
    assert float(near[5].max()) == 0.0


def test_frame_points_prefer_the_corrected_file(tmp_path: Path) -> None:
    frame = _keyboard_frame(DEFAULT_CORNERS)
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    derived = frame_points(frame, fixed)
    assert derived is not None
    assert len(derived.gaps) == 6
    assert fixed_path(frame, fixed) is None
    (fixed / f"{frame.image_path.stem}.json").write_text(json.dumps(_corrected()))
    points = frame_points(frame, fixed)
    assert points is not None
    assert len(points.gaps) == 2


def test_heatmap_targets_with_no_keypoints_is_all_zero_heat_and_all_one_weight() -> None:
    crop = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    heat, weight = heatmap_targets(
        None, crop, (TRACK_HEIGHT, TRACK_WIDTH), (TRACK_WIDTH, TRACK_HEIGHT), None
    )
    assert float(heat.max()) == 0.0
    assert np.array_equal(weight, np.ones_like(weight))


def test_squash_crop_maps_the_whole_frame_onto_the_search_square() -> None:
    crop = squash_crop((640, 480), SEARCH_SIZE, SEARCH_SIZE)
    corners = crop.points(np.array([(0.0, 0.0), (640.0, 480.0)]))
    assert np.allclose(corners, [(0.0, 0.0), (SEARCH_SIZE, SEARCH_SIZE)])


def test_oriented_crop_generalises_the_keyseg_crop_size() -> None:
    quad = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 300.0), (100.0, 300.0)])
    generic = oriented_crop(
        quad, TRACK_WIDTH, TRACK_HEIGHT, TRACK_MARGIN_ALONG, TRACK_MARGIN_ACROSS
    )
    corners = generic.points(quad)
    assert np.allclose(corners.mean(axis=0), (TRACK_WIDTH / 2, TRACK_HEIGHT / 2))


def test_keyseg_crop_for_stays_identical_to_the_general_oriented_crop() -> None:
    quad = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 300.0), (100.0, 300.0)])
    expected = oriented_crop(quad, CROP_WIDTH, CROP_HEIGHT, CROP_MARGIN_ALONG, CROP_MARGIN_ACROSS)
    assert np.array_equal(crop_for(quad).to_crop, expected.to_crop)


def test_sidecar_parses_a_negative_with_no_corners_and_keeps_the_sequence_fields(
    tmp_path: Path,
) -> None:
    sidecar = {
        "imageWidth": 640,
        "imageHeight": 480,
        "keys": [],
        "piano": False,
        "sequence": "clip-004",
        "frameIndex": 12,
    }
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((480, 640, 3), dtype=np.uint8))
    (tmp_path / "a.json").write_text(json.dumps(sidecar))

    frame = parse_synth_keys_sidecar(tmp_path / "a.json")

    assert frame.corners_px is None
    assert frame.keys == []
    assert frame.sequence == "clip-004"
    assert frame.frame_index == 12
    assert keypoints(frame) is None
    assert presence_target(frame) == 0.0


_SLANTED_QUAD = np.array([[200.0, 100.0], [440.0, 120.0], [620.0, 380.0], [60.0, 340.0]])


def test_rectified_crop_sends_the_quad_corners_to_the_margined_rectangle() -> None:
    crop = rectified_crop(_SLANTED_QUAD, TRACK_WIDTH, TRACK_HEIGHT, 0.08, 0.15)
    left, top = TRACK_WIDTH * 0.08 / 1.16, TRACK_HEIGHT * 0.15 / 1.3
    expected = [
        (left, top),
        (TRACK_WIDTH - left, top),
        (TRACK_WIDTH - left, TRACK_HEIGHT - top),
        (left, TRACK_HEIGHT - top),
    ]
    assert np.allclose(crop.points(_SLANTED_QUAD), expected, atol=1e-3)


def test_rectified_crop_gives_equal_template_steps_equal_crop_steps() -> None:
    world = np.array([[0.0, 0.0], [52.0, 0.0], [52.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    camera = cv2.getPerspectiveTransform(
        world, np.array([[300, 100], [500, 160], [640, 400], [100, 330]], dtype=np.float32)
    )
    quad = cv2.perspectiveTransform(world[None], camera)[0]
    keys = np.stack([np.arange(53.0), np.full(53, 6.0)], axis=1).astype(np.float32)
    front = cv2.perspectiveTransform(keys[None], camera)[0]
    crop = rectified_crop(quad.astype(np.float64), TRACK_WIDTH, TRACK_HEIGHT, 0.08, 0.15)
    steps = np.diff(crop.points(front.astype(np.float64))[:, 0])
    assert np.allclose(steps, steps[0], rtol=1e-4)
    frame_steps = np.linalg.norm(np.diff(front, axis=0), axis=1)
    assert frame_steps.max() > 1.3 * frame_steps.min()


def test_an_affine_crop_keeps_a_flat_third_row() -> None:
    crop = oriented_crop(_SLANTED_QUAD, TRACK_WIDTH, TRACK_HEIGHT, 0.08, 0.15)
    assert crop.to_crop.shape == (3, 3)
    assert np.array_equal(crop.to_crop[2], [0.0, 0.0, 1.0])
