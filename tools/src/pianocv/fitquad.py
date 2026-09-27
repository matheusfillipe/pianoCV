"""Recover a keybed quad from a mask by fitting its four sides, not by predicting its corners."""

import cv2
import numpy as np

_MIN_AREA_PX = 200
_MIN_SIDE_POINTS = 8
_TRIM_FRACTION = 0.12
_EPSILON_STEPS = 40
_EPSILON_RANGE = (0.001, 0.2)


def _largest_contour(mask: np.ndarray) -> np.ndarray | None:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 2:
        return None
    index = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    if int(stats[index, cv2.CC_STAT_AREA]) < _MIN_AREA_PX:
        return None
    contours, _ = cv2.findContours(
        (labels == index).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return np.asarray(max(contours, key=cv2.contourArea).reshape(-1, 2), dtype=np.float64)


def _seed_quad(contour: np.ndarray) -> np.ndarray | None:
    hull = cv2.convexHull(contour.astype(np.float32))
    perimeter = cv2.arcLength(hull, True)
    if perimeter <= 0.0:
        return None
    # the loosest simplification that still keeps four sides is the one that found the real corners
    low, high = _EPSILON_RANGE
    for _ in range(_EPSILON_STEPS):
        middle = (low + high) / 2.0
        if len(cv2.approxPolyDP(hull, middle * perimeter, True)) > 4:
            low = middle
        else:
            high = middle
    approx = cv2.approxPolyDP(hull, high * perimeter, True)
    if len(approx) != 4:
        return None
    return approx.reshape(4, 2).astype(np.float64)


def _segment_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    edge = end - start
    length_sq = float(edge @ edge)
    if length_sq < 1e-9:
        return np.asarray(np.linalg.norm(points - start, axis=1), dtype=np.float64)
    t = np.clip((points - start) @ edge / length_sq, 0.0, 1.0)
    return np.asarray(
        np.linalg.norm(points - (start + t[:, None] * edge), axis=1), dtype=np.float64
    )


def _trim_ends(side: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    edge = end - start
    length_sq = float(edge @ edge)
    if length_sq < 1e-9:
        return side
    t = (side - start) @ edge / length_sq
    keep = side[(t > _TRIM_FRACTION) & (t < 1.0 - _TRIM_FRACTION)]
    return keep if len(keep) >= _MIN_SIDE_POINTS else side


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    return centre, vt[0]


def _intersect(
    a_point: np.ndarray, a_dir: np.ndarray, b_point: np.ndarray, b_dir: np.ndarray
) -> np.ndarray | None:
    matrix = np.column_stack([a_dir, -b_dir])
    if abs(float(np.linalg.det(matrix))) < 1e-9:
        return None
    t = np.linalg.solve(matrix, b_point - a_point)
    return np.asarray(a_point + t[0] * a_dir, dtype=np.float64)


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    contour = _largest_contour(mask)
    if contour is None:
        return None
    seed = _seed_quad(contour)
    if seed is None:
        return None
    distances = np.stack([_segment_distance(contour, seed[i], seed[(i + 1) % 4]) for i in range(4)])
    owner = np.argmin(distances, axis=0)
    lines: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(4):
        side = contour[owner == i]
        if len(side) < _MIN_SIDE_POINTS:
            return seed
        lines.append(_fit_line(_trim_ends(side, seed[i], seed[(i + 1) % 4])))
    corners = []
    for i in range(4):
        corner = _intersect(*lines[i - 1], *lines[i])
        if corner is None:
            return seed
        corners.append(corner)
    return np.asarray(corners, dtype=np.float64)
