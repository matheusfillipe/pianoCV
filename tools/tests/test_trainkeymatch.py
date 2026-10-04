import numpy as np

from pianocv.trainkeymatch import _augment_strip


def test_augment_strip_covers_every_augmentation_branch() -> None:
    strip = np.full((64, 768, 3), 128, dtype=np.uint8)
    for seed in range(60):
        augmented = _augment_strip(strip, np.random.default_rng(seed))
        assert augmented.shape == strip.shape
        assert augmented.dtype == np.uint8
