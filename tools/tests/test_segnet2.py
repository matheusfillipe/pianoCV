import numpy as np

from pianocv.segnet2 import normalise


def test_normalise_centres_on_imagenet_statistics() -> None:
    grey = np.full((1, 8, 8, 3), 255, dtype=np.uint8)
    out = normalise(grey)
    assert out.shape == (1, 3, 8, 8)
    assert out[0, 0].min() > 2.0
