from pathlib import Path

import numpy as np
import torch

from conftest import ExportKeyseg
from pianocv.keyseg import CROP_HEIGHT, CROP_WIDTH, KeySegNet, keyseg_from_onnx, preprocess_crop


def test_model_keeps_the_crop_resolution() -> None:
    model = KeySegNet(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH))
    assert out.shape == (1, 4, CROP_HEIGHT, CROP_WIDTH)


def test_preprocess_gives_the_model_a_channels_first_crop() -> None:
    crop = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
    assert preprocess_crop(crop).shape == (3, CROP_HEIGHT, CROP_WIDTH)


def test_a_model_rebuilt_from_its_export_gives_the_same_classes(
    tmp_path: Path, export_keyseg: ExportKeyseg
) -> None:
    torch.manual_seed(0)
    model = KeySegNet(pretrained=False)
    with torch.no_grad():
        for norm in model.modules():
            if isinstance(norm, torch.nn.BatchNorm2d):
                assert norm.running_mean is not None and norm.running_var is not None
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.uniform_(-0.2, 0.2)
                norm.running_mean.uniform_(-0.2, 0.2)
                norm.running_var.uniform_(0.5, 1.5)
    path = tmp_path / "keyseg.onnx"
    export_keyseg(model, str(path))
    rebuilt = keyseg_from_onnx(str(path))
    crop = torch.randn(1, 3, CROP_HEIGHT, CROP_WIDTH)
    with torch.no_grad():
        expected = torch.softmax(model.eval()(crop), dim=1)
        got = torch.softmax(rebuilt(crop), dim=1)
    assert float((expected - got).abs().max()) < 1e-3
