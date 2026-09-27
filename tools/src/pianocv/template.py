"""Canonical keybed appearance in rectified strip space, used to score a candidate quad."""

import cv2
import numpy as np

_WHITE_COUNT = 52.0
_BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
_BLACK_WIDTH = 0.58
_BLACK_DEPTH_FRACTION = 0.62
_WHITE_VALUE = 1.0
_BLACK_VALUE = 0.2
_SEAM_VALUE = 0.5

STRIP_WIDTH = 520
STRIP_HEIGHT = 64


def build_template(
    strip_width: int,
    strip_height: int,
    white_count: float = _WHITE_COUNT,
    phase: float = 0.0,
) -> np.ndarray:
    unit = strip_width / white_count
    bar_height = round(strip_height * _BLACK_DEPTH_FRACTION)
    template = np.zeros((strip_height, strip_width), dtype=np.float64)
    for key in range(int(white_count)):
        x0 = round(key * unit)
        x1 = round((key + 1) * unit)
        template[:, x0:x1] = _WHITE_VALUE
        if key > 0:
            template[:, x0 - 1 : x0] = _SEAM_VALUE
    for octave in range(-1, int(white_count // 7) + 2):
        for offset in _BLACK_OFFSETS:
            u0 = 7.0 * octave + offset - phase
            u1 = u0 + _BLACK_WIDTH
            if u1 <= 0.0 or u0 >= white_count:
                continue
            x0 = max(0, round(u0 * unit))
            x1 = min(strip_width, round(u1 * unit))
            if x1 > x0:
                template[:bar_height, x0:x1] = _BLACK_VALUE
    return template


def strip_destination(strip_width: int, strip_height: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(strip_width - 1), 0.0],
            [float(strip_width - 1), float(strip_height - 1)],
            [0.0, float(strip_height - 1)],
        ],
        dtype=np.float32,
    )


TEMPLATE = build_template(STRIP_WIDTH, STRIP_HEIGHT)


def score(
    image_bgr: np.ndarray,
    quad_px: np.ndarray,
    strip_width: int = STRIP_WIDTH,
    strip_height: int = STRIP_HEIGHT,
    white_count: float = _WHITE_COUNT,
    phase: float = 0.0,
) -> float:
    if (strip_width, strip_height, white_count, phase) == (
        STRIP_WIDTH,
        STRIP_HEIGHT,
        _WHITE_COUNT,
        0.0,
    ):
        template = TEMPLATE
    else:
        template = build_template(strip_width, strip_height, white_count, phase)
    matrix = cv2.getPerspectiveTransform(
        quad_px.astype(np.float32), strip_destination(strip_width, strip_height)
    )
    strip = cv2.warpPerspective(image_bgr, matrix, (strip_width, strip_height))
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    warped = gray - gray.mean()
    reference = template - template.mean()
    norm = float(np.sqrt((warped * warped).sum() * (reference * reference).sum()))
    if norm <= 0.0:
        return 0.0
    return float((warped * reference).sum() / norm)
