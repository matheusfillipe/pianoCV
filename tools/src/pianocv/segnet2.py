"""Keybed segmentation on a pretrained backbone.

Same output as KeybedSegNet, a mask the geometry then fits, because that fit measures 2.0 px
given a correct mask. What changes is where the features come from: ImageNet weights carry the
lighting and viewpoint invariance that 6000 renders could not teach, which is what failed when
the old net met a room it had not been trained in.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from pianocv.model import MASK_SIZE

SEG2_INPUT_SIZE = 288

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
# mobilenet_v3_small stages whose output stride is 4, 8, 16 and 32
_SKIP_LAYERS = (1, 3, 8, 12)
_SKIP_CHANNELS = (16, 24, 48, 576)
_DECODER_CHANNELS = (96, 48, 24, 16)


class _Up(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs + skip, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out: torch.Tensor = self.block(x)
        return out


class KeybedSegNet2(nn.Module):
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        ups: list[_Up] = []
        inputs = _SKIP_CHANNELS[-1]
        for index, outputs in enumerate(_DECODER_CHANNELS):
            skip = _SKIP_CHANNELS[len(_SKIP_CHANNELS) - 2 - index] if index < 3 else 0
            ups.append(_Up(inputs, skip, outputs))
            inputs = outputs
        self.ups = nn.ModuleList(ups)
        self.out = nn.Conv2d(inputs, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in _SKIP_LAYERS:
                skips.append(x)
        y = skips[-1]
        for index, up in enumerate(self.ups):
            y = up(y, skips[len(skips) - 2 - index] if index < 3 else None)
        logits: torch.Tensor = self.out(y)
        if logits.shape[-1] != MASK_SIZE:
            logits = nn.functional.interpolate(
                logits, size=(MASK_SIZE, MASK_SIZE), mode="bilinear", align_corners=False
            )
        return logits


def resize_rgb(image_bgr: np.ndarray, size: int = SEG2_INPUT_SIZE) -> np.ndarray:
    """The input as uint8, so a whole corpus fits in memory before normalising."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    return np.asarray(resized, dtype=np.uint8)


def normalise(batch_hwc: np.ndarray) -> np.ndarray:
    """The pretrained backbone was trained on ImageNet statistics and needs them back."""
    scaled = np.asarray(batch_hwc, dtype=np.float32) / 255.0
    standard = (scaled - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.asarray(standard.transpose(0, 3, 1, 2), dtype=np.float32)


def preprocess_seg2(image_bgr: np.ndarray) -> np.ndarray:
    resized = resize_rgb(image_bgr, SEG2_INPUT_SIZE)
    return np.asarray(normalise(resized[None])[0], dtype=np.float32)


def predict_mask2(model: KeybedSegNet2, image_bgr: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(preprocess_seg2(image_bgr)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0].numpy()
    return np.asarray(probability, dtype=np.float32)


def load_seg2(path: Path) -> KeybedSegNet2:
    model = KeybedSegNet2(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model
