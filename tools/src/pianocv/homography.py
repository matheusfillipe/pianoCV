"""4-point homography solve for keyboard rectification."""

import numpy as np


def find_homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    if src.shape != (4, 2) or dst.shape != (4, 2):
        raise ValueError("src and dst must have shape (4, 2)")
    matrix = np.zeros((8, 8), dtype=np.float64)
    rhs = np.zeros(8, dtype=np.float64)
    for i in range(4):
        x, y = src[i]
        u, v = dst[i]
        matrix[2 * i] = (x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
        matrix[2 * i + 1] = (0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
        rhs[2 * i] = u
        rhs[2 * i + 1] = v
    solution = np.linalg.solve(matrix, rhs)
    return np.append(solution, 1.0).reshape(3, 3)
