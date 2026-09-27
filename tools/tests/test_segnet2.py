from pathlib import Path

import cv2
import numpy as np
import torch

from pianocv.evaluate import seg2_detector
from pianocv.model import MASK_SIZE
from pianocv.segnet2 import (
    SEG2_INPUT_SIZE,
    KeybedSegNet2,
    load_seg2,
    normalise,
    predict_mask2,
    preprocess_seg2,
    resize_rgb,
)

IMAGE = np.full((480, 640, 3), 90, dtype=np.uint8)


def test_resize_rgb_swaps_channels_and_squares_the_frame() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    resized = resize_rgb(image)
    assert resized.shape == (SEG2_INPUT_SIZE, SEG2_INPUT_SIZE, 3)
    assert resized[0, 0, 2] == 255
    assert resized[0, 0, 0] == 0


def test_normalise_centres_on_imagenet_statistics() -> None:
    grey = np.full((1, 8, 8, 3), 255, dtype=np.uint8)
    out = normalise(grey)
    assert out.shape == (1, 3, 8, 8)
    # a saturated channel lands at (1 - mean) / std, the largest value the backbone expects
    assert out[0, 0].min() > 2.0


def test_preprocess_produces_a_single_chw_sample() -> None:
    assert preprocess_seg2(IMAGE).shape == (3, SEG2_INPUT_SIZE, SEG2_INPUT_SIZE)


def test_forward_returns_a_mask_the_geometry_can_fit() -> None:
    model = KeybedSegNet2(pretrained=False)
    logits = model(torch.zeros(2, 3, SEG2_INPUT_SIZE, SEG2_INPUT_SIZE))
    assert logits.shape == (2, 1, MASK_SIZE, MASK_SIZE)


def test_predict_mask_is_a_probability_at_mask_resolution() -> None:
    probability = predict_mask2(KeybedSegNet2(pretrained=False), IMAGE)
    assert probability.shape == (MASK_SIZE, MASK_SIZE)
    assert float(probability.min()) >= 0.0
    assert float(probability.max()) <= 1.0


def test_head_resamples_when_the_input_does_not_land_on_the_mask_grid() -> None:
    model = KeybedSegNet2(pretrained=False)
    logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 1, MASK_SIZE, MASK_SIZE)


def test_a_saved_model_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "seg2.pt"
    torch.save(KeybedSegNet2(pretrained=False).state_dict(), path)
    assert predict_mask2(load_seg2(path), IMAGE).shape == (MASK_SIZE, MASK_SIZE)


def test_seg2_detector_turns_a_mask_into_an_oriented_quad() -> None:
    keybed = np.array([[80.0, 200.0], [560.0, 190.0], [566.0, 250.0], [84.0, 262.0]])
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [keybed.astype(np.int32)], (235, 235, 235))

    class Flat(KeybedSegNet2):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
            grid = keybed / np.array([640.0, 480.0]) * (MASK_SIZE - 1)
            cv2.fillPoly(mask, [grid.astype(np.int32)], 1.0)
            logits = torch.from_numpy(mask * 20.0 - 10.0)
            return logits.view(1, 1, MASK_SIZE, MASK_SIZE).expand(x.shape[0], 1, -1, -1)

    detection = seg2_detector(Flat(pretrained=False))(image)
    assert detection is not None
    assert np.linalg.norm(detection.quad_px - keybed, axis=1).max() < 25.0


def test_seg2_detector_reports_nothing_when_the_mask_is_empty() -> None:
    class Empty(KeybedSegNet2):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full((x.shape[0], 1, MASK_SIZE, MASK_SIZE), -20.0)

    assert seg2_detector(Empty(pretrained=False))(IMAGE) is None
