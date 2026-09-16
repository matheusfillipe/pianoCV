import numpy as np

from kvt.keyalignmenteval import _iou


def test_iou_counts_overlapping_masks() -> None:
    predicted = np.array([[True, True], [False, False]])
    target = np.array([[True, False], [True, False]])

    assert _iou(predicted, target) == 1 / 3
