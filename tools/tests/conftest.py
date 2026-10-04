from collections.abc import Callable

import pytest
import torch

from pianocv.keyseg import CROP_HEIGHT, CROP_WIDTH, KeySegNet

ExportKeyseg = Callable[[KeySegNet, str], None]


class _Softmax(torch.nn.Module):
    def __init__(self, model: KeySegNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def _export(model: KeySegNet, onnx_path: str) -> None:
    model.eval()
    torch.onnx.export(
        _Softmax(model),
        (torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH),),
        onnx_path,
        input_names=["crop"],
        output_names=["classes"],
        opset_version=17,
        dynamo=False,
    )


@pytest.fixture
def export_keyseg() -> ExportKeyseg:
    return _export
