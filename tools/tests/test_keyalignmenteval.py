import json
import runpy
from pathlib import Path

import cv2
import numpy as np
import pytest

import kvt.keyalignmenteval as evaluation


def test_iou_counts_overlapping_and_empty_masks() -> None:
    predicted = np.array([[True, True], [False, False]])
    target = np.array([[True, False], [True, False]])

    assert evaluation._iou(predicted, target) == 1 / 3
    assert evaluation._iou(np.zeros((1, 1), dtype=bool), np.zeros((1, 1), dtype=bool)) == 1


def test_entries_paths_input_and_preview_validate_data(tmp_path: Path) -> None:
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text('{"image":"images/a.png"}\n')
    assert evaluation._entries(manifest) == [{"image": "images/a.png"}]
    assert (
        evaluation._path({"image": "images/a.png"}, "image", tmp_path) == tmp_path / "images/a.png"
    )
    with pytest.raises(ValueError, match="field image"):
        evaluation._path({}, "image", tmp_path)

    image = np.array([[[0, 0, 255], [0, 255, 0]]], dtype=np.uint8)
    model_input = evaluation._input(image)
    assert model_input.shape == (1, 3, 240, 320)
    assert model_input.dtype == np.float32
    assert model_input[0, 0, 0, 0] == 1

    mask = np.array([[True, False]])
    preview = evaluation._preview(image, mask, mask)
    assert preview.shape == image.shape
    assert not np.array_equal(preview, image)


class _Session:
    def __init__(self, path: str, providers: list[str]) -> None:
        self.path = path
        self.providers = providers

    def run(self, _: None, values: dict[str, np.ndarray]) -> list[np.ndarray]:
        assert values["image"].shape == (1, 3, 240, 320)
        return [np.full((1, 1, 240, 320), 8, dtype=np.float32)]


def _dataset(root: Path, *, readable: bool = True, entries: int = 1) -> None:
    (root / "images").mkdir(parents=True)
    (root / "visible").mkdir()
    if readable:
        image = np.full((4, 6, 3), 80, dtype=np.uint8)
        target = np.full((4, 6), 255, dtype=np.uint8)
        cv2.imwrite(str(root / "images" / "sample.png"), image)
        cv2.imwrite(str(root / "visible" / "sample.png"), target)
    record = json.dumps({"image": "images/sample.png", "visible": "visible/sample.png"})
    (root / "validation.jsonl").write_text("\n".join([record] * entries) + "\n")


def test_evaluate_writes_preview_and_scores_visible_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    monkeypatch.setattr("kvt.keyalignmenteval.ort.InferenceSession", _Session)

    score = evaluation.evaluate(tmp_path / "model.onnx", dataset, tmp_path / "out")

    assert score == 1
    assert (tmp_path / "out" / "sample.png").is_file()


def test_evaluate_stops_writing_previews_after_twelve_samples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _dataset(dataset, entries=13)
    monkeypatch.setattr("kvt.keyalignmenteval.ort.InferenceSession", _Session)

    assert evaluation.evaluate(tmp_path / "model.onnx", dataset, tmp_path / "out") == 1


def test_evaluate_rejects_missing_validation_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "dataset"
    _dataset(dataset, readable=False)
    monkeypatch.setattr("kvt.keyalignmenteval.ort.InferenceSession", _Session)

    with pytest.raises(ValueError, match="cannot read validation sample"):
        evaluation.evaluate(tmp_path / "model.onnx", dataset, tmp_path / "out")


def test_main_uses_argument_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "model.onnx"
    dataset = tmp_path / "dataset"
    output = tmp_path / "out"
    captured: list[tuple[Path, Path, Path]] = []

    def fake_evaluate(model_path: Path, dataset_dir: Path, out_dir: Path) -> float:
        captured.append((model_path, dataset_dir, out_dir))
        return 0.625

    monkeypatch.setattr(evaluation, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "keyalignmenteval",
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(output),
        ],
    )

    evaluation.main()

    assert captured == [(model, dataset, output)]
    assert capsys.readouterr().out == "validation visible IoU 0.6250\n"


def test_module_entrypoint_runs_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = tmp_path / "dataset"
    _dataset(dataset)
    model = tmp_path / "model.onnx"
    output = tmp_path / "out"
    monkeypatch.setattr("onnxruntime.InferenceSession", _Session)
    monkeypatch.setattr(
        "sys.argv",
        [
            "keyalignmenteval",
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--out-dir",
            str(output),
        ],
    )

    runpy.run_module("kvt.keyalignmenteval", run_name="__main__")

    assert capsys.readouterr().out == "validation visible IoU 1.0000\n"
