from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from pianocv.keynetfp16 import main, to_fp16
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


def test_main_converts_the_source_and_reports_the_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    full = tmp_path / "keynet.onnx"
    half = tmp_path / "keynet-fp16.onnx"
    export_keynet_onnx(KeyNet(pretrained=False), str(full))
    monkeypatch.setattr("sys.argv", ["keynetfp16", str(full), str(half)])

    main()

    assert half.is_file()
    assert capsys.readouterr().out.startswith(f"wrote {half}")
