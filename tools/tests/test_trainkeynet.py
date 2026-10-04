import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pianocv.keymatch import PerturbConfig, load_synth_keys
from pianocv.keynet import (
    BACK_CHANNELS,
    BLACK_TOP_CHANNELS,
    TRACK_HEIGHT,
    TRACK_MARGIN_ACROSS,
    TRACK_MARGIN_ALONG,
    TRACK_WIDTH,
    Crop,
    KeyPoints,
    offset_targets,
)
from pianocv.keynetmodel import KeyNet
from pianocv.trainkeynet import (
    Recipe,
    _lower_resolution,
    camera_sim,
    heatmap_loss,
    offset_loss,
    sample,
    score_peaks,
    train_keynet,
)

_WIDTH, _HEIGHT = 640, 480


def _fraction(points: list[tuple[float, float]]) -> list[dict[str, float]]:
    return [{"x": x / _WIDTH, "y": y / _HEIGHT} for x, y in points]


def _write_frames(path: Path, count: int, piano: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    whites = [
        {
            "pitch": 60 + 2 * i,
            "black": False,
            "top": _fraction(
                [
                    (64 + 128 * i, 150),
                    (192 + 128 * i, 150),
                    (192 + 128 * i, 330),
                    (64 + 128 * i, 330),
                ]
            ),
            "front": None,
        }
        for i in range(4)
    ]
    top = [(170.0, 150.0), (214.0, 150.0), (214.0, 250.0), (170.0, 250.0)]
    black = {
        "pitch": 61,
        "black": True,
        "top": _fraction(top),
        "front": _fraction([top[3], top[2], (214.0, 260.0), (170.0, 260.0)]),
    }
    for index in range(count):
        image = np.full((_HEIGHT, _WIDTH, 3), 90, np.uint8)
        cv2.imwrite(str(path / f"f{index}.png"), image)
        sidecar: dict[str, object] = {"imageWidth": _WIDTH, "imageHeight": _HEIGHT}
        if piano:
            sidecar |= {
                "corners": _fraction([(64, 150), (576, 150), (576, 330), (64, 330)]),
                "keys": [*whites, black],
            }
        else:
            sidecar |= {"keys": [], "piano": False}
        (path / f"f{index}.json").write_text(json.dumps(sidecar))


def test_training_runs_on_stills_negatives_and_held_out_real_frames(tmp_path: Path) -> None:
    _write_frames(tmp_path / "stills", 3)
    _write_frames(tmp_path / "motion", 2, piano=False)
    _write_frames(tmp_path / "real", 2)
    (tmp_path / "real" / "f1.png").rename(tmp_path / "real" / "rec-held-f1.png")
    (tmp_path / "real" / "f1.json").rename(tmp_path / "real" / "rec-held-f1.json")
    model, history = train_keynet(
        tmp_path / "stills",
        motion_dir=tmp_path / "motion",
        real_dir=tmp_path / "real",
        hold_out=("rec-held",),
        epochs=1,
        batch=2,
        steps_per_epoch=4,
        workers=0,
        pretrained=False,
    )
    assert len(history) == 1
    assert history[0].real is not None
    assert not model.training


def test_heads_only_training_leaves_the_encoder_and_decoder_untouched(tmp_path: Path) -> None:
    _write_frames(tmp_path / "stills", 3)
    torch.manual_seed(0)
    before = KeyNet(pretrained=False).state_dict()
    model, _ = train_keynet(
        tmp_path / "stills",
        epochs=1,
        batch=2,
        steps_per_epoch=3,
        workers=0,
        pretrained=False,
        recipe=Recipe(heads_only=True),
    )
    after = model.state_dict()
    for name, value in before.items():
        moved = not torch.equal(value, after[name].cpu())
        assert moved == name.startswith(("heat.", "presence.")), name


def test_a_rectified_sample_peaks_at_the_projected_corners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_frames(tmp_path / "real", 1)
    frame = load_synth_keys(tmp_path / "real")[0]
    monkeypatch.setattr("pianocv.trainkeynet.perturb_quad", lambda quad, _rng, _config: quad)
    monkeypatch.setattr("pianocv.trainkeynet._jitter", lambda quad, _rng: quad)
    monkeypatch.setattr("pianocv.trainkeynet._jitter_corners", lambda quad, _rng: quad)
    _, heat, _, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, Recipe(rectified=True)
    )
    left = TRACK_WIDTH * TRACK_MARGIN_ALONG / (1 + 2 * TRACK_MARGIN_ALONG)
    top = TRACK_HEIGHT * TRACK_MARGIN_ACROSS / (1 + 2 * TRACK_MARGIN_ACROSS)
    for channel, (x, y) in enumerate(
        [(left, top), (TRACK_WIDTH - left, top), (TRACK_WIDTH - left, TRACK_HEIGHT - top)]
    ):
        row, col = np.unravel_index(np.argmax(heat[channel]), heat[channel].shape)
        assert (int(col), int(row)) == (round(x / 2 - 0.25), round(y / 2 - 0.25))


def test_a_real_frame_teaches_no_black_key_tops(tmp_path: Path) -> None:
    _write_frames(tmp_path / "real", 1)
    sidecar = tmp_path / "real" / "f0.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()) | {"kind": "real-keys"}))
    frame = load_synth_keys(tmp_path / "real")[0]
    recipe = Recipe(mask_real_tops=True)
    _, heat, weight, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, recipe
    )
    assert float(heat[BLACK_TOP_CHANNELS[0]].max()) > 0.5
    assert float(weight[list(BLACK_TOP_CHANNELS)].max()) == 0.0
    assert float(weight[5].min()) == 1.0


def test_a_hidden_corner_has_no_peak_and_no_weight_around_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pianocv.trainkeynet.perturb_quad", lambda quad, _rng, _config: quad)
    monkeypatch.setattr("pianocv.trainkeynet._jitter", lambda quad, _rng: quad)
    monkeypatch.setattr("pianocv.trainkeynet._jitter_corners", lambda quad, _rng: quad)
    _write_frames(tmp_path / "synth", 1)
    sidecar = tmp_path / "synth" / "f0.json"
    sidecar.write_text(
        json.dumps(json.loads(sidecar.read_text()) | {"cornerVisible": [False, True, True, True]})
    )
    frame = load_synth_keys(tmp_path / "synth")[0]
    _, heat, weight, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, Recipe(rectified=True)
    )
    assert float(heat[0].max()) == 0.0
    assert float(weight[0].min()) == 0.0
    assert float(heat[1].max()) > 0.5
    assert float(weight[1].min()) == 1.0


def test_a_real_frame_without_back_labels_has_zero_weight_on_the_back_channels(
    tmp_path: Path,
) -> None:
    _write_frames(tmp_path / "real", 1)
    sidecar = tmp_path / "real" / "f0.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()) | {"kind": "real-keys"}))
    frame = load_synth_keys(tmp_path / "real")[0]
    _, _, weight, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, Recipe()
    )
    assert float(weight[list(BACK_CHANNELS)].max()) == 0.0
    assert float(weight[4].min()) == 1.0


def test_a_real_frame_with_back_labels_keeps_their_weight(tmp_path: Path) -> None:
    _write_frames(tmp_path / "real", 1)
    sidecar = tmp_path / "real" / "f0.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()) | {"kind": "real-keys"}))
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "f0.json").write_text(
        json.dumps(
            {
                "corners": [[64, 150], [576, 150], [576, 330], [64, 330]],
                "gaps": [[192, 330]],
                "blackLow": [],
                "blackHigh": [],
                "blackTopLow": [],
                "blackTopHigh": [],
                "backGaps": [[192, 150]],
                "blackBackLow": [],
                "blackBackHigh": [],
            }
        )
    )
    frame = load_synth_keys(tmp_path / "real")[0]
    _, heat, weight, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, Recipe(fixed_dir=fixed)
    )
    assert float(weight[BACK_CHANNELS[0]].min()) == 1.0
    assert float(heat[BACK_CHANNELS[0]].max()) > 0.5


def test_a_corrected_real_frame_trains_on_its_points_and_keeps_its_tops(tmp_path: Path) -> None:
    _write_frames(tmp_path / "real", 1)
    sidecar = tmp_path / "real" / "f0.json"
    sidecar.write_text(json.dumps(json.loads(sidecar.read_text()) | {"kind": "real-keys"}))
    fixed = tmp_path / "fixed"
    fixed.mkdir()
    (fixed / "f0.json").write_text(
        json.dumps(
            {
                "width": _WIDTH,
                "height": _HEIGHT,
                "whiteKeys": 4,
                "phase": "E",
                "corners": [[64, 150], None, [576, 330], [64, 330]],
                "gaps": [[192, 330], None, [448, 330]],
                "blackLow": [],
                "blackHigh": [],
                "blackTopLow": [[300, 250]],
                "blackTopHigh": [None],
            }
        )
    )
    frame = load_synth_keys(tmp_path / "real")[0]
    recipe = Recipe(mask_real_tops=True, fixed_dir=fixed)
    _, heat, weight, _, _, _ = sample(
        frame, False, np.random.default_rng(0), PerturbConfig(), False, recipe
    )
    assert float(heat[BLACK_TOP_CHANNELS[0]].max()) > 0.5
    assert float(heat[BLACK_TOP_CHANNELS[1]].max()) == 0.0
    assert float(weight[list(BLACK_TOP_CHANNELS)].min()) == 1.0
    assert float(heat[1].max()) == 0.0


def test_a_lower_resolution_frame_keeps_its_size_and_loses_fine_detail() -> None:
    stripes = np.zeros((480, 1920, 3), np.uint8)
    stripes[:, ::2] = 255
    lowered = _lower_resolution(stripes, np.random.default_rng(0))
    assert lowered.shape == stripes.shape
    assert float(lowered.std()) < float(stripes.std()) / 4


def test_peak_scores_leave_out_truths_the_weights_leave_out() -> None:
    target = np.zeros((7, 16, 16), np.float32)
    target[4, 4, 4] = 1.0
    target[5, 10, 10] = 1.0
    weight = np.ones_like(target)
    weight[5] = 0.0
    predicted = np.zeros_like(target)
    predicted[4, 4, 4] = 0.9
    assert score_peaks(predicted, target, weight).recall == pytest.approx(1.0)


def test_training_needs_rendered_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no rendered frames"):
        train_keynet(tmp_path, epochs=1, workers=0, pretrained=False)


def test_the_loss_rewards_a_confident_peak_where_the_target_has_one() -> None:
    target = torch.zeros(1, 7, 8, 8)
    target[0, 4, 3, 3] = 1.0
    weight = torch.ones_like(target)
    right = torch.full_like(target, -6.0)
    right[0, 4, 3, 3] = 6.0
    wrong = torch.full_like(target, -6.0)
    wrong[0, 4, 5, 5] = 6.0
    assert float(heatmap_loss(right, target, weight)) < float(heatmap_loss(wrong, target, weight))


def test_missing_a_keybed_corner_costs_more_than_missing_one_of_many_gaps() -> None:
    target = torch.zeros(1, 7, 8, 64)
    target[0, 0, 2, 2] = 1.0
    for column in range(2, 62):
        target[0, 4, 5, column] = 1.0
    weight = torch.ones_like(target)
    perfect = torch.where(target > 0, torch.tensor(6.0), torch.tensor(-6.0))
    no_corner = perfect.clone()
    no_corner[0, 0, 2, 2] = -6.0
    no_gap = perfect.clone()
    no_gap[0, 4, 5, 30] = -6.0
    assert float(heatmap_loss(no_corner, target, weight)) > float(
        heatmap_loss(no_gap, target, weight)
    )


def test_the_single_peak_loss_pulls_only_a_points_own_cell_to_one() -> None:
    # an off-centre point leaves its neighbour above 0.8 too
    ys, xs = torch.meshgrid(torch.arange(8.0), torch.arange(8.0), indexing="ij")
    gaussian = torch.exp(-((xs - 3.4) ** 2 + (ys - 3.0) ** 2) / 2)
    target = torch.zeros(1, 7, 8, 8)
    target[0, 4] = gaussian
    weight = torch.ones_like(target)
    own_cell = torch.full_like(target, -6.0)
    own_cell[0, 4, 3, 3] = 6.0
    both = own_cell.clone()
    both[0, 4, 3, 4] = 6.0
    assert float(gaussian[3, 4]) >= 0.8
    single = heatmap_loss(own_cell, target, weight, single_peak=True)
    flat = heatmap_loss(own_cell, target, weight)
    assert float(single) < float(heatmap_loss(both, target, weight, single_peak=True))
    assert float(single) < float(flat)


def test_offset_targets_point_each_nearby_cell_to_its_keypoint() -> None:
    identity = Crop(to_crop=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
    point = np.array([101.3, 50.9])
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
    offset, weight = offset_targets(points, identity, (TRACK_HEIGHT, TRACK_WIDTH))
    cx, cy = point / 2 - 0.25
    rows, cols = np.nonzero(weight[0])
    assert len(rows) == 9
    for i, j in zip(rows, cols, strict=True):
        assert offset[0, i, j] == pytest.approx(cx - j)
        assert offset[1, i, j] == pytest.approx(cy - i)
    nearest = (round(cy), round(cx))
    assert weight[0][nearest] == weight[0].max()


def test_the_offset_loss_is_zero_for_exact_offsets_and_ignores_unweighted_cells() -> None:
    target = torch.zeros(1, 2 * 9, 4, 4)
    target[0, 0, 1, 1] = 0.3
    weight = torch.zeros(1, 9, 4, 4)
    weight[0, 0, 1, 1] = 1.0
    exact = target.clone()
    exact[0, 5, 2, 2] = 7.0
    assert float(offset_loss(exact, target, weight)) == pytest.approx(0.0)
    assert float(offset_loss(torch.zeros_like(target), target, weight)) == pytest.approx(0.15)


def test_peak_scores_count_matches_misses_and_extras() -> None:
    target = np.zeros((7, 16, 16), np.float32)
    target[4, 4, 4] = 1.0
    target[4, 10, 10] = 1.0
    predicted = np.zeros_like(target)
    predicted[4, 4, 5] = 0.9
    predicted[4, 2, 13] = 0.9
    score = score_peaks(predicted, target, np.ones_like(target))
    assert score.recall == pytest.approx(0.5)
    assert score.precision == pytest.approx(0.5)
    assert score.error_px == pytest.approx(2.0)


def test_camera_sim_changes_pixels_keeps_targets_and_repeats_under_a_seed(tmp_path: Path) -> None:
    _write_frames(tmp_path / "real", 1)
    frame = load_synth_keys(tmp_path / "real")[0]

    def draw(recipe: Recipe) -> tuple[np.ndarray, np.ndarray]:
        pixels, heat, *_ = sample(
            frame, False, np.random.default_rng(3), PerturbConfig(), True, recipe
        )
        return pixels, heat

    plain_pixels, plain_heat = draw(Recipe())
    sim_pixels, sim_heat = draw(Recipe(camera_sim=True))
    again_pixels, _ = draw(Recipe(camera_sim=True))
    assert sim_pixels.shape == plain_pixels.shape
    assert not np.array_equal(sim_pixels, plain_pixels)
    assert np.array_equal(sim_pixels, again_pixels)
    assert np.array_equal(sim_heat, plain_heat)
    assert np.unravel_index(np.argmax(sim_heat[0]), sim_heat[0].shape) == np.unravel_index(
        np.argmax(plain_heat[0]), plain_heat[0].shape
    )


def test_camera_sim_keeps_the_crop_size_on_a_flat_crop() -> None:
    flat = np.full((TRACK_HEIGHT, TRACK_WIDTH, 3), 120, np.uint8)
    assert camera_sim(flat, np.random.default_rng(1)).shape == flat.shape
