import cv2
import numpy as np
import torch

from pianocv.jitter import (
    OneEuro,
    _align,
    detect_quad,
    keybed_shaped,
    quad_from_probability,
)
from pianocv.model import MASK_SIZE
from pianocv.segnet2 import KeybedSegNet2

QUAD = np.array([[100.0, 200.0], [500.0, 190.0], [505.0, 250.0], [104.0, 262.0]])


def test_align_undoes_a_half_turn_between_frames() -> None:
    run = _align([QUAD, np.roll(QUAD, 2, axis=0), QUAD])
    assert np.allclose(run[1], QUAD)
    assert np.allclose(run.std(axis=0), 0.0)


def test_align_keeps_real_motion() -> None:
    moved = QUAD + np.array([3.0, 0.0])
    run = _align([QUAD, moved])
    assert np.allclose(run[1], moved)


def test_detect_quad_returns_nothing_for_an_empty_mask() -> None:
    class Empty(KeybedSegNet2):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 1, MASK_SIZE, MASK_SIZE), -20.0)

    image = np.full((480, 640, 3), 90, dtype=np.uint8)
    assert detect_quad(Empty(pretrained=False), image) is None


def test_one_euro_holds_still_and_follows_a_step() -> None:
    smoother = OneEuro()
    still = QUAD.copy()
    for _ in range(10):
        out = smoother.push(still + np.random.default_rng(1).normal(0.0, 0.5, QUAD.shape))
    # a static input with sub-pixel noise settles to sub-pixel of the truth
    assert np.linalg.norm(out - still, axis=1).max() < 1.0
    moved = still + np.array([60.0, 0.0])
    for _ in range(30):
        out = smoother.push(moved)
    # and a real move is followed, not lagged away
    assert np.linalg.norm(out - moved, axis=1).max() < 2.0


def test_quad_from_probability_matches_a_drawn_keybed() -> None:
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [QUAD.astype(np.int32)], (235, 235, 235))
    small = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    grid = QUAD / np.array([640.0, 480.0]) * (MASK_SIZE - 1)
    cv2.fillPoly(small, [grid.astype(np.int32)], 1.0)
    quad = quad_from_probability(image, small)
    assert quad is not None
    best = min(
        np.linalg.norm(quad - QUAD, axis=1).max(),
        np.linalg.norm(np.roll(quad, 2, axis=0) - QUAD, axis=1).max(),
    )
    assert best < 25.0


def test_keybed_shaped_accepts_a_keybed_and_rejects_a_fold() -> None:
    assert keybed_shaped(QUAD)
    bowtie = QUAD[[0, 1, 3, 2]]
    assert not keybed_shaped(bowtie)
    square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
    assert not keybed_shaped(square)
