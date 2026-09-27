import numpy as np

from pianocv.pose import DEPTH_UNITS, WHITE_KEY_COUNT, fit_pose

WIDTH, HEIGHT = 640, 480


def _project(focal: float, yaw: float, pitch: float, distance: float) -> np.ndarray:
    """The keybed rectangle seen through a pinhole camera, corners in label order."""
    world = np.array(
        [
            [0.0, 0.0, 0.0],
            [WHITE_KEY_COUNT, 0.0, 0.0],
            [WHITE_KEY_COUNT, DEPTH_UNITS, 0.0],
            [0.0, DEPTH_UNITS, 0.0],
        ]
    )
    world -= [WHITE_KEY_COUNT / 2, DEPTH_UNITS / 2, 0.0]
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


def test_a_true_view_of_the_keybed_solves_with_almost_no_residual() -> None:
    quad = _project(focal=700.0, yaw=0.5, pitch=-0.9, distance=60.0)
    fit = fit_pose(quad, WIDTH, HEIGHT)
    assert fit.residual < 0.02
    assert abs(fit.focal - 700.0) / 700.0 < 0.05


def test_a_square_cannot_be_the_keybed() -> None:
    square = np.array([[200.0, 150.0], [440.0, 150.0], [440.0, 390.0], [200.0, 390.0]])
    # frontal, so orthogonality says nothing; the column-length ratio is what refuses it,
    # and for a square read as 52 by 6.4 that ratio term is 1 - 6.4/52
    assert fit_pose(square, WIDTH, HEIGHT).residual > 0.5


def test_corner_order_does_not_change_the_answer() -> None:
    quad = _project(focal=900.0, yaw=-0.3, pitch=-0.7, distance=55.0)
    a = fit_pose(quad, WIDTH, HEIGHT).residual
    b = fit_pose(np.roll(quad, 2, axis=0), WIDTH, HEIGHT).residual
    assert abs(a - b) < 1e-6
