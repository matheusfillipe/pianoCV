import cv2
import numpy as np

from pianocv.fitquad import quad_from_mask

QUAD = np.array([[90.0, 210.0], [520.0, 150.0], [545.0, 240.0], [100.0, 320.0]])


def _mask(quad: np.ndarray, width: int = 640, height: int = 480) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    return mask


def _match(pred: np.ndarray, truth: np.ndarray) -> float:
    best = float("inf")
    for reverse in (False, True):
        ordered = pred[::-1] if reverse else pred
        for roll in range(4):
            error = float(np.linalg.norm(np.roll(ordered, roll, axis=0) - truth, axis=1).mean())
            best = min(best, error)
    return best


def test_recovers_a_clean_quad_to_subpixel() -> None:
    recovered = quad_from_mask(_mask(QUAD))
    assert recovered is not None
    assert _match(recovered, QUAD) < 1.5


def test_survives_a_ragged_mask() -> None:
    rng = np.random.default_rng(0)
    mask = _mask(QUAD)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    speckle = rng.random(mask.shape) < 0.02
    mask[speckle] = 255 - mask[speckle]
    recovered = quad_from_mask(mask)
    assert recovered is not None
    assert _match(recovered, QUAD) < 4.0


def test_ignores_a_smaller_blob() -> None:
    mask = _mask(QUAD)
    cv2.circle(mask, (600, 430), 20, 255, -1)
    recovered = quad_from_mask(mask)
    assert recovered is not None
    assert _match(recovered, QUAD) < 2.0


def test_returns_none_without_a_keybed() -> None:
    assert quad_from_mask(np.zeros((480, 640), dtype=np.uint8)) is None


def test_returns_none_for_a_speck() -> None:
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.circle(mask, (320, 240), 3, 255, -1)
    assert quad_from_mask(mask) is None
