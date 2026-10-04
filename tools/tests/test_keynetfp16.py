from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from pianocv.keynetfp16 import to_fp16
from pianocv.keynetmodel import KeyNet, export_keynet_onnx


def test_the_half_precision_export_gives_the_same_heatmaps(tmp_path: Path) -> None:
    full = tmp_path / "keynet.onnx"
    half = tmp_path / "keynet-fp16.onnx"
    export_keynet_onnx(KeyNet(pretrained=False), str(full))
    to_fp16(full, half)
    model = onnx.load(str(half))
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert any(i.data_type == onnx.TensorProto.FLOAT16 for i in model.graph.initializer)
    image = np.random.default_rng(0).standard_normal((1, 3, 160, 768)).astype(np.float32)
    outputs = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
            ["heatmaps"], {"image": image}
        )[0]
        for path in (full, half)
    ]
    assert outputs[1].dtype == np.float32
    assert np.abs(outputs[0] - outputs[1]).max() < 1e-2
