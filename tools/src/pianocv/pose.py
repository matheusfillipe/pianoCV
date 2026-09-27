"""Can this quad be a perspective view of the keybed rectangle? Port of web/src/pose.ts.

The keybed is 52 white keys of 23.5 mm by 150 mm deep. With a pinhole camera of unknown focal
length, the rectangle-to-image homography decomposes into two rotation columns that must be
orthonormal; the focal that makes them so is the camera's, and how far they stay from
orthonormal is the residual. Measured on the labelled frames it is under 0.44 for a real
keybed quad and over 18 for a broken one.
"""

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from pianocv.dataset import canonical_quad

WHITE_KEY_MM = 23.5
# the instrument in front of the camera: a 61-key board, 36 white keys, depth measured from
# labelled frames at a span-to-depth ratio of 7.2; synthetic renders use their own 88-key geometry
KEYBED_DEPTH_MM = 118.0
WHITE_KEY_COUNT = 36
RENDER_WHITE_KEY_COUNT = 52
RENDER_DEPTH_UNITS = 150.0 / WHITE_KEY_MM
DEPTH_UNITS = KEYBED_DEPTH_MM / WHITE_KEY_MM
_WORLD = np.array(
    [[0.0, 0.0], [WHITE_KEY_COUNT, 0.0], [WHITE_KEY_COUNT, DEPTH_UNITS], [0.0, DEPTH_UNITS]],
    dtype=np.float32,
)
_SCAN_SAMPLES = 200
_GOLDEN_ITERATIONS = 100
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True)
class PoseFit:
    focal: float
    residual: float


def _residual_at(h: np.ndarray, focal: float, cx: float, cy: float) -> float:
    b = np.empty((3, 3))
    b[0] = (h[0] - cx * h[2]) / focal
    b[1] = (h[1] - cy * h[2]) / focal
    b[2] = h[2]
    n1, n2 = float(np.linalg.norm(b[:, 0])), float(np.linalg.norm(b[:, 1]))
    r1, r2 = b[:, 0] / n1, b[:, 1] / n2
    # a rotation's columns are orthogonal AND the same length; the length test is what stops a
    # frontal view of the wrong aspect passing, where orthogonality alone says nothing
    return float(
        abs(np.dot(r1, r2))
        + abs(1.0 - np.linalg.norm(np.cross(r1, r2)))
        + abs(1.0 - min(n1, n2) / max(n1, n2))
    )


def _golden(fn: Callable[[float], float], lo: float, hi: float) -> float:
    a, b = lo, hi
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(_GOLDEN_ITERATIONS):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = fn(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = fn(d)
    return (a + b) / 2.0


def fit_pose(quad_px: np.ndarray, width: int, height: int) -> PoseFit:
    """Focal length and orthonormality residual for the quad read as the keybed rectangle."""
    ordered = canonical_quad(quad_px).astype(np.float32)
    h = cv2.getPerspectiveTransform(_WORLD, ordered)
    cx, cy = width / 2.0, height / 2.0

    def residual(focal: float) -> float:
        return _residual_at(h, focal, cx, cy)

    lo, hi = 0.3 * width, 3.0 * width
    step = (hi - lo) / _SCAN_SAMPLES
    best_focal, best = lo, residual(lo)
    for i in range(1, _SCAN_SAMPLES + 1):
        f = lo + i * step
        value = residual(f)
        if value < best:
            best_focal, best = f, value
    golden = _golden(residual, lo, hi)
    golden_value = residual(golden)
    if golden_value < best:
        best_focal, best = golden, golden_value
    return PoseFit(focal=float(best_focal), residual=float(best))
