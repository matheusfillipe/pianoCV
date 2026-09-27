import cv2
import numpy as np

from pianocv.refine_edges import refine_quad

QUAD = np.array([[90.0, 200.0], [520.0, 150.0], [530.0, 250.0], [96.0, 300.0]])


def _keybed(quad: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.fillPoly(image, [quad.astype(np.int32)], (225, 225, 225))
    return image


def _max_error(quad: np.ndarray, truth: np.ndarray) -> float:
    return float(np.linalg.norm(quad - truth, axis=1).max())


def test_pulls_a_nudged_quad_back_onto_the_edges() -> None:
    image = _keybed(QUAD)
    rng = np.random.default_rng(0)
    nudged = QUAD + rng.uniform(-5.0, 5.0, size=(4, 2))
    refined = refine_quad(image, nudged)
    assert _max_error(refined, QUAD) < _max_error(nudged, QUAD)


def test_leaves_a_true_quad_alone() -> None:
    refined = refine_quad(_keybed(QUAD), QUAD)
    assert _max_error(refined, QUAD) < 3.0


def test_keeps_the_input_when_there_is_no_edge_to_find() -> None:
    flat = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert np.array_equal(refine_quad(flat, QUAD), QUAD)


def test_refuses_a_correction_larger_than_the_search_window() -> None:
    image = _keybed(QUAD)
    far = QUAD + np.array([160.0, 120.0])
    # nothing within the search window, so the coarse quad has to survive untouched
    assert np.array_equal(refine_quad(image, far), far)
