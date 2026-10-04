"""ImageNet input normalisation for the pretrained backbones."""

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalise(batch_hwc: np.ndarray) -> np.ndarray:
    """The pretrained backbone was trained on ImageNet statistics and needs them back."""
    scaled = np.asarray(batch_hwc, dtype=np.float32) / 255.0
    standard = (scaled - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.asarray(standard.transpose(0, 3, 1, 2), dtype=np.float32)
