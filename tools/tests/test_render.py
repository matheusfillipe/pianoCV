import math

import cv2
import numpy as np
import pytest

from pianocv.render import (
    _MIN_VISIBLE_FRACTION,
    _WHITE_COUNTS,
    _sample_quad,
    _visible_fraction,
    render_sample,
)
from pianocv.template import score

WIDTH = 640
HEIGHT = 480
STRIP_WIDTH = 520
STRIP_HEIGHT = 64
BACKGROUND_QUAD = np.array([[100.0, 80.0], [520.0, 60.0], [560.0, 400.0], [60.0, 420.0]])


def _flat_background() -> np.ndarray:
    return np.full((HEIGHT, WIDTH, 3), 80.0, dtype=np.float32)


def test_render_sample_is_deterministic_given_seed() -> None:
    first = render_sample(np.random.default_rng(3))
    second = render_sample(np.random.default_rng(3))
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.quad_px, second.quad_px)
    assert first.present == second.present


def test_render_sample_composite_mode_is_deterministic_given_seed() -> None:
    first = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    second = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.quad_px, second.quad_px)
    assert first.present == second.present


def test_render_sample_composite_mode_renders_keybed_at_true_quad() -> None:
    sample = render_sample(
        np.random.default_rng(4), WIDTH, HEIGHT, _flat_background(), BACKGROUND_QUAD
    )
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.image.dtype == np.float32
    assert float(sample.image.min()) >= 0.0
    assert float(sample.image.max()) <= 255.0
    assert sample.present
    image_bgr = cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    assert score(image_bgr, sample.quad_px, STRIP_WIDTH, STRIP_HEIGHT) > 0.3


def test_render_sample_composite_mode_absent_leaves_inpaint_only() -> None:
    background = _flat_background()
    cv2.rectangle(background, (200, 100), (440, 380), (200.0, 200.0, 200.0), -1)
    samples = [
        render_sample(np.random.default_rng(seed), WIDTH, HEIGHT, background, BACKGROUND_QUAD)
        for seed in range(40)
    ]
    assert any(not sample.present for sample in samples)
    for sample in samples:
        if not sample.present:
            image_bgr = cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
            assert score(image_bgr, sample.quad_px, STRIP_WIDTH, STRIP_HEIGHT) <= 0.3


def test_render_sample_composite_mode_resizes_background() -> None:
    background = np.full((240, 320, 3), 80.0, dtype=np.float32)
    quad = np.array([[50.0, 40.0], [260.0, 30.0], [280.0, 200.0], [30.0, 210.0]])
    sample = render_sample(np.random.default_rng(9), WIDTH, HEIGHT, background, quad)
    assert sample.image.shape == (HEIGHT, WIDTH, 3)


def test_render_sample_falls_back_to_synthetic_without_real_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pianocv.render._REAL_FRAMES", [])
    sample = render_sample(np.random.default_rng(3))
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.quad_px.shape == (4, 2)


def test_sampled_quads_hold_the_keybed_aspect_ratio() -> None:
    ratios = []
    for seed in range(200):
        quad = _sample_quad(np.random.default_rng(seed), WIDTH, HEIGHT)
        span = np.linalg.norm(quad[1] - quad[0])
        depth = np.linalg.norm(quad[3] - quad[0])
        ratios.append(float(span / depth))
    assert min(ratios) > 2.0
    assert max(ratios) < 40.0
    assert np.median(ratios) > 5.0


def test_sampled_quads_cover_a_range_of_orientations() -> None:
    angles = []
    for seed in range(200):
        quad = _sample_quad(np.random.default_rng(seed), WIDTH, HEIGHT)
        edge = quad[1] - quad[0]
        angles.append(math.degrees(math.atan2(edge[1], edge[0])))
    assert np.std(angles) > 10.0


def test_synthetic_keybed_renders_against_the_canonical_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("pianocv.render._REAL_FRAMES", [])
    scores = []
    for seed in range(120):
        sample = render_sample(np.random.default_rng(seed))
        # a quad running off the frame warps padding into the strip, which the template cannot match
        if not sample.present or not _fully_visible(sample.quad_px):
            continue
        image_bgr = cv2.cvtColor(sample.image.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # the sheet starts on a random note, so the oracle has to try every one of the 7 phases
        scores.append(
            max(
                score(
                    image_bgr,
                    sample.quad_px,
                    STRIP_WIDTH,
                    STRIP_HEIGHT,
                    sample.white_count,
                    float(phase),
                )
                for phase in range(7)
            )
        )
    assert len(scores) > 10
    assert np.median(scores) > 0.5


def _fully_visible(quad: np.ndarray) -> bool:
    x = quad[:, 0]
    y = quad[:, 1]
    return bool(((x >= 0) & (x < WIDTH) & (y >= 0) & (y < HEIGHT)).all())


def test_render_sample_shapes_and_ranges() -> None:
    sample = render_sample(np.random.default_rng(0))
    assert sample.image.shape == (HEIGHT, WIDTH, 3)
    assert sample.image.dtype == np.float32
    assert float(sample.image.min()) >= 0.0
    assert float(sample.image.max()) <= 255.0
    assert sample.quad_px.shape == (4, 2)
    assert isinstance(sample.present, bool)


def test_render_sample_supports_custom_size() -> None:
    sample = render_sample(np.random.default_rng(1), width=320, height=240)
    assert sample.image.shape == (240, 320, 3)


def test_render_sample_present_flag_varies() -> None:
    samples = [render_sample(np.random.default_rng(seed)) for seed in range(40)]
    assert any(sample.present for sample in samples)
    assert any(not sample.present for sample in samples)


def test_render_sample_keeps_enough_of_the_keybed_on_screen() -> None:
    # corners off the frame are wanted, a keybed that missed the frame entirely is not
    fractions = [
        _visible_fraction(render_sample(np.random.default_rng(seed)).quad_px, WIDTH, HEIGHT)
        for seed in range(80)
    ]
    assert min(fractions) > 0.2
    assert np.median(fractions) >= _MIN_VISIBLE_FRACTION


def test_render_sample_covers_real_keyboard_sizes() -> None:
    counts = {render_sample(np.random.default_rng(seed)).white_count for seed in range(120)}
    assert len(counts) >= 4
    assert counts <= set(_WHITE_COUNTS)
