"""Snap a fitted keybed quad onto the image gradients that actually mark its edges."""

import cv2
import numpy as np

_SAMPLES = 48
# measured over the corpus: 10 px lands on the true edge, wider searches find the wrong one
# (key separators, the case, the floor line) and cost more than the mask error they recover
SEARCH_PX = 10.0
_SEARCH_STEP = 0.5
_MIN_RESPONSE = 6.0
_MIN_POINTS = 8
_TRIM = 0.08
_BLUR_SIGMA = 1.2
# refinement nudges an edge onto a gradient; a corner that moves further than the search
# window did not find its edge, it found somebody else's, so the coarse quad is kept
_MAX_SHIFT = 4.0

Line = tuple[np.ndarray, np.ndarray]


def _sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    return cv2.remap(
        image,
        xs.astype(np.float32).reshape(-1, 1),
        ys.astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    ).ravel()


def _refine_edge(
    gray: np.ndarray, start: np.ndarray, end: np.ndarray, search_px: float
) -> np.ndarray | None:
    edge = end - start
    length = float(np.linalg.norm(edge))
    if length < 1.0:
        return None
    direction = edge / length
    normal = np.array([-direction[1], direction[0]])
    ts = np.linspace(_TRIM, 1.0 - _TRIM, _SAMPLES)
    offsets = np.arange(-search_px, search_px + 1e-6, _SEARCH_STEP)

    base = start[None, :] + ts[:, None] * edge
    points = base[:, None, :] + offsets[None, :, None] * normal[None, None, :]
    values = _sample(gray, points[:, :, 0].ravel(), points[:, :, 1].ravel())
    values = values.reshape(len(ts), len(offsets))
    # the edge sits where brightness changes fastest along the normal, not where the mask stopped
    gradient = np.abs(np.gradient(values, _SEARCH_STEP, axis=1))
    best = np.argmax(gradient, axis=1)
    keep = gradient[np.arange(len(ts)), best] > _MIN_RESPONSE
    if int(keep.sum()) < _MIN_POINTS:
        return None
    return np.asarray(base[keep] + offsets[best[keep]][:, None] * normal[None, :])


def _fit(points: np.ndarray) -> Line:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    return centre, vt[0]


def _intersect(a: Line, b: Line) -> np.ndarray | None:
    matrix = np.column_stack([a[1], -b[1]])
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        return None
    t = np.linalg.solve(matrix, b[0] - a[0])
    return np.asarray(a[0] + t[0] * a[1], dtype=np.float64)


def refine_quad(
    image_bgr: np.ndarray, quad: np.ndarray, search_px: float = SEARCH_PX
) -> np.ndarray:
    gray = cv2.GaussianBlur(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), (0, 0), _BLUR_SIGMA)
    lines: list[Line] = []
    for i in range(4):
        points = _refine_edge(gray, quad[i], quad[(i + 1) % 4], search_px)
        if points is None:
            return quad
        lines.append(_fit(points))
    corners = []
    for i in range(4):
        corner = _intersect(lines[i - 1], lines[i])
        if corner is None:
            return quad
        corners.append(corner)
    refined = np.asarray(corners, dtype=np.float64)
    if float(np.linalg.norm(refined - quad, axis=1).max()) > _MAX_SHIFT * search_px:
        return quad
    return refined
