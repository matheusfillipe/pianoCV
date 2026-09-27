import numpy as np
import pytest

from pianocv.homography import find_homography

SQUARE = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])


def apply_homography(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogenous = matrix @ np.array([point[0], point[1], 1.0])
    return np.asarray(homogenous[:2] / homogenous[2], dtype=np.float64)


def test_identity_for_square_to_itself() -> None:
    assert np.allclose(find_homography(SQUARE, SQUARE), np.eye(3))


def test_square_to_parallelogram() -> None:
    dst = SQUARE + np.array([2.0, 3.0])
    expected = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]])
    assert np.allclose(find_homography(SQUARE, dst), expected)


def test_round_trip_maps_src_onto_dst() -> None:
    dst = np.array([[0.0, 0.0], [2.0, 0.1], [1.6, 1.2], [0.1, 0.9]])
    homography = find_homography(SQUARE, dst)
    for src_point, dst_point in zip(SQUARE, dst, strict=True):
        assert np.allclose(apply_homography(homography, src_point), dst_point, atol=1e-9)


def test_wrong_shape_raises() -> None:
    with pytest.raises(ValueError):
        find_homography(np.zeros((3, 2)), SQUARE)
    with pytest.raises(ValueError):
        find_homography(SQUARE, np.zeros((4, 3)))
