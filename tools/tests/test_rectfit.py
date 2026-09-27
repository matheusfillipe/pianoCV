import cv2
import numpy as np

from pianocv.dataset import align
from pianocv.model import MASK_SIZE
from pianocv.pose import DEPTH_UNITS, WHITE_KEY_COUNT
from pianocv.rectfit import boundary_points, fit_rectangle, snap_to_gradient

WIDTH, HEIGHT = 640, 480
# the mask is 144 cells across the frame, so the boundary is known to about one cell
CELL_PX = WIDTH / (MASK_SIZE - 1)


def _project(focal: float, yaw: float, pitch: float, distance: float) -> np.ndarray:
    world = np.array(
        [
            [0.0, 0.0, 0.0],
            [WHITE_KEY_COUNT, 0.0, 0.0],
            [WHITE_KEY_COUNT, DEPTH_UNITS, 0.0],
            [0.0, DEPTH_UNITS, 0.0],
        ]
    ) - [WHITE_KEY_COUNT / 2, DEPTH_UNITS / 2, 0.0]
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
        [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
    )
    camera = world @ rotation.T + [0.0, 0.0, distance]
    return np.column_stack(
        [
            WIDTH / 2 + focal * camera[:, 0] / camera[:, 2],
            HEIGHT / 2 + focal * camera[:, 1] / camera[:, 2],
        ]
    )


def _mask_of(quad: np.ndarray) -> np.ndarray:
    # a probability map the way the net produces one: the coverage of each cell, so the
    # half-probability line runs through the true edge, never along cell boundaries
    fine = 8
    mask = np.zeros((MASK_SIZE * fine, MASK_SIZE * fine), dtype=np.float32)
    grid = quad / np.array([WIDTH, HEIGHT]) * (MASK_SIZE - 1) * fine
    cv2.fillPoly(mask, [grid.astype(np.int32)], 1.0)
    return cv2.resize(mask, (MASK_SIZE, MASK_SIZE), interpolation=cv2.INTER_AREA)


def _short_far_end(quad: np.ndarray, fraction: float) -> np.ndarray:
    q = quad.copy()
    far_first = np.linalg.norm(q[2] - q[1]) < np.linalg.norm(q[3] - q[0])
    i, j, k, m = (1, 2, 0, 3) if far_first else (0, 3, 1, 2)
    q[i] = q[i] + (q[k] - q[i]) * fraction
    q[j] = q[j] + (q[m] - q[j]) * fraction
    return q


def test_a_far_end_read_short_is_placed_by_the_outline_and_the_shape() -> None:
    truth = _project(focal=750.0, yaw=0.9, pitch=-0.8, distance=100.0)
    points = boundary_points(_mask_of(truth), WIDTH, HEIGHT)
    assert points is not None
    coarse = _short_far_end(truth, 0.15)
    before = np.linalg.norm(align(coarse, truth) - truth, axis=1).max()
    fitted = fit_rectangle(points, coarse, WIDTH, HEIGHT).quad_px
    after = np.linalg.norm(align(fitted, truth) - truth, axis=1).max()
    assert before > 25.0
    assert after < 1.5 * CELL_PX


def test_an_end_with_no_evidence_is_placed_by_the_length_once_the_focal_is_held() -> None:
    truth = _project(focal=750.0, yaw=0.9, pitch=-0.8, distance=100.0)
    points = boundary_points(_mask_of(truth), WIDTH, HEIGHT)
    assert points is not None
    # drop every boundary point near the far end, the way an end out of frame has none
    far_first = np.linalg.norm(truth[2] - truth[1]) < np.linalg.norm(truth[3] - truth[0])
    far = (truth[1] + truth[2]) / 2 if far_first else (truth[0] + truth[3]) / 2
    span = np.linalg.norm(truth[1] - truth[0])
    kept = points[np.linalg.norm(points - far, axis=1) > 0.2 * span]
    coarse = _short_far_end(truth, 0.12)
    held = fit_rectangle(kept, coarse, WIDTH, HEIGHT, focal=750.0)
    assert np.isfinite(held.cost)
    # the seen end is exact; the unseen end lies on the long edges, but where along them is
    # only pinned by perspective, so its position is not promised
    fitted = align(held.quad_px, truth)
    seen = [0, 3] if far_first else [1, 2]
    assert np.linalg.norm(fitted[seen] - truth[seen], axis=1).max() < 1.5 * CELL_PX
    for corner in [1, 2] if far_first else [0, 3]:
        a, b = (truth[0], truth[1]) if corner in (0, 1) else (truth[3], truth[2])
        assert _line_distance(fitted[corner], a, b) < 1.5 * CELL_PX


def test_a_correct_quad_stays_put() -> None:
    truth = _project(focal=900.0, yaw=-0.4, pitch=-0.7, distance=90.0)
    points = boundary_points(_mask_of(truth), WIDTH, HEIGHT)
    assert points is not None
    fitted = fit_rectangle(points, truth, WIDTH, HEIGHT).quad_px
    assert np.linalg.norm(align(fitted, truth) - truth, axis=1).max() < 1.2 * CELL_PX


def test_boundary_points_snap_onto_the_image_edge_and_drop_where_there_is_none() -> None:
    truth = _project(focal=900.0, yaw=-0.4, pitch=-0.7, distance=90.0)
    gray = np.full((HEIGHT, WIDTH), 30, dtype=np.uint8)
    cv2.fillPoly(gray, [truth.astype(np.int32)], 230)
    # the mask boundary sits three pixels inside the real edge, the way it does on real frames
    inward = truth.mean(axis=0) - truth
    inside = truth + inward / np.linalg.norm(inward, axis=1)[:, None] * 3.0
    points = boundary_points(_mask_of(inside), WIDTH, HEIGHT)
    assert points is not None
    _, before = _distances(points, truth)
    snapped = snap_to_gradient(gray, points, inside)
    _, after = _distances(snapped, truth)
    assert before > 2.0
    assert after < 0.75
    flat = np.full((HEIGHT, WIDTH), 30, dtype=np.uint8)
    assert len(snap_to_gradient(flat, points, inside)) == 0


def _line_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    direction = (b - a) / np.linalg.norm(b - a)
    offset = point - a
    return float(abs(direction[0] * offset[1] - direction[1] * offset[0]))


def _distances(points: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, float]:
    best = np.full(len(points), np.inf)
    for i in range(4):
        a, d = quad[i], quad[(i + 1) % 4] - quad[i]
        t = np.clip(((points - a) @ d) / float(d @ d), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(points - (a + t[:, None] * d), axis=1))
    return best, float(np.median(best))
