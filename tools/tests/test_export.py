from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from pianocv.export import export_onnx
from pianocv.model import INPUT_SIZE, KeybedNet, load_model, predict_corners, preprocess


def _checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    path = tmp_path / "keybed_net.pt"
    torch.save(KeybedNet().state_dict(), path)
    return path


def _session(onnx_path: Path) -> onnxruntime.InferenceSession:
    return onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])


def test_export_onnx_matches_torch(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path)
    onnx_path = export_onnx(checkpoint, tmp_path / "keybed_net.onnx")
    assert onnx_path.is_file()

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(240, 320, 3), dtype=np.uint8)
    gray = preprocess(image).reshape(1, 1, INPUT_SIZE, INPUT_SIZE)
    corners, present = _session(onnx_path).run(None, {"image": gray})

    quad, probability = predict_corners(load_model(checkpoint), image)
    expected = quad / np.array([float(image.shape[1]), float(image.shape[0])])
    assert np.allclose(corners.reshape(4, 2), expected, atol=1e-4)
    assert abs(float(present[0]) - probability) < 1e-4


def test_export_onnx_declares_the_browser_contract(tmp_path: Path) -> None:
    session = _session(export_onnx(_checkpoint(tmp_path), tmp_path / "keybed_net.onnx"))
    inputs = session.get_inputs()
    assert [i.name for i in inputs] == ["image"]
    assert inputs[0].shape == [1, 1, INPUT_SIZE, INPUT_SIZE]
    assert [o.name for o in session.get_outputs()] == ["corners", "present"]


def test_export_onnx_carries_no_source_paths(tmp_path: Path) -> None:
    onnx_path = export_onnx(_checkpoint(tmp_path), tmp_path / "keybed_net.onnx")
    model = onnx.load(str(onnx_path))
    assert model.doc_string == ""
    assert model.graph.doc_string == ""
    for graph in [model.graph, *model.functions]:
        for node in graph.node:
            assert node.doc_string == ""
            assert list(node.metadata_props) == []
    assert b"site-packages" not in onnx_path.read_bytes()
