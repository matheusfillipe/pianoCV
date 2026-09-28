import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pianocv.keyseg import KeySegNet, export_keyseg_onnx, keyseg_from_onnx
from pianocv.trainkeyseg import (
    EpochMetrics,
    _build_parser,
    _require_mlflow,
    main,
    train_keyseg,
    write_previews,
)

_WIDTH, _HEIGHT = 640, 480
QUAD = [(64.0, 96.0), (576.0, 96.0), (576.0, 384.0), (64.0, 384.0)]


def _write_frames(path: Path, count: int) -> None:
    def fraction(points: list[tuple[float, float]]) -> list[dict[str, float]]:
        return [{"x": x / _WIDTH, "y": y / _HEIGHT} for x, y in points]

    keys = [
        {
            "pitch": 60 + 2 * i,
            "black": False,
            "top": fraction(
                [(64 + 170 * i, 96), (234 + 170 * i, 96), (234 + 170 * i, 384), (64 + 170 * i, 384)]
            ),
            "front": None,
        }
        for i in range(3)
    ]
    top = [(200.0, 96.0), (268.0, 96.0), (268.0, 200.0), (200.0, 200.0)]
    keys.append(
        {
            "pitch": 61,
            "black": True,
            "top": fraction(top),
            "front": fraction([top[3], top[2], (268.0, 210.0), (200.0, 210.0)]),
        }
    )
    for index in range(count):
        cv2.imwrite(str(path / f"f{index}.png"), np.full((_HEIGHT, _WIDTH, 3), 120, np.uint8))
        (path / f"f{index}.json").write_text(
            json.dumps(
                {
                    "corners": fraction(QUAD),
                    "imageWidth": _WIDTH,
                    "imageHeight": _HEIGHT,
                    "keys": keys,
                }
            )
        )


def test_train_keyseg_raises_without_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no synth-keys frames"):
        train_keyseg(tmp_path)


def test_train_keyseg_runs_a_tiny_schedule(tmp_path: Path) -> None:
    _write_frames(tmp_path, 3)
    seen: list[EpochMetrics] = []
    model, history = train_keyseg(
        tmp_path,
        epochs=1,
        batch=1,
        steps_per_epoch=1,
        workers=0,
        pretrained=False,
        on_epoch=seen.append,
    )
    assert len(history) == 1
    assert seen == history
    assert len(history[0].iou) == 4
    assert not model.training


def test_fine_tuning_from_an_export_keeps_its_statistics_and_scores_held_out_real_frames(
    tmp_path: Path,
) -> None:
    synthetic = tmp_path / "synth"
    real = tmp_path / "real"
    synthetic.mkdir()
    real.mkdir()
    _write_frames(synthetic, 2)
    _write_frames(real, 2)
    (real / "f1.png").rename(real / "rec-held-f1.png")
    (real / "f1.json").rename(real / "rec-held-f1.json")
    export = tmp_path / "start.onnx"
    export_keyseg_onnx(KeySegNet(pretrained=False), str(export))
    started = keyseg_from_onnx(str(export))
    model, history = train_keyseg(
        synthetic,
        epochs=1,
        batch=1,
        steps_per_epoch=1,
        workers=0,
        init_onnx=export,
        real_dir=real,
        real_share=1.0,
        hold_out=("rec-held",),
    )
    assert history[0].real_iou is not None
    for before, after in zip(started.modules(), model.cpu().modules(), strict=True):
        if isinstance(before, torch.nn.BatchNorm2d):
            assert isinstance(after, torch.nn.BatchNorm2d)
            assert after.running_var is not None and before.running_var is not None
            assert torch.equal(after.running_var, before.running_var)


def test_previews_draw_the_labels_over_the_crops(tmp_path: Path) -> None:
    _write_frames(tmp_path, 2)
    written = write_previews(tmp_path, tmp_path / "previews", 2)
    assert len(written) == 2
    preview = cv2.imread(str(written[0]))
    assert preview is not None and preview.shape[0] == 2 * 224


def test_parser_reads_the_mlflow_experiment_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "segments")
    assert _build_parser().parse_args([]).mlflow_experiment == "segments"


def test_require_mlflow_raises_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pianocv.trainkeyseg.mlflow", None)
    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        _require_mlflow()


class _FakeModel:
    def cpu(self) -> "_FakeModel":
        return self

    def state_dict(self) -> dict[str, object]:
        return {}


class _FakeRun:
    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None


class _FakeMlflow:
    def __init__(self) -> None:
        self.experiment: str | None = None
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.artifacts: list[str] = []

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self) -> _FakeRun:
        return _FakeRun()

    def log_params(self, _params: dict[str, object]) -> None:
        return None

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((step, metrics))

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)


def _fake_train(*_args: object, **kwargs: object) -> tuple[_FakeModel, list[EpochMetrics]]:
    metrics = EpochMetrics(epoch=1, loss=0.5, val_loss=0.4, iou=(0.9, 0.6, 0.8, 0.3))
    on_epoch = kwargs.get("on_epoch")
    if callable(on_epoch):
        on_epoch(metrics)
    return _FakeModel(), [metrics]


def _patch_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pianocv.trainkeyseg.train_keyseg", _fake_train)
    monkeypatch.setattr(
        "pianocv.trainkeyseg.export_keyseg_onnx",
        lambda _model, path: Path(path).write_bytes(b"onnx"),
    )
    monkeypatch.setattr(
        "pianocv.trainkeyseg.torch.save", lambda _state, path: Path(path).write_bytes(b"pt")
    )


def test_main_writes_the_model_and_its_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_training(monkeypatch)
    out_dir = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["trainkeyseg", "--out-dir", str(out_dir)])
    main()
    assert (out_dir / "keyseg.onnx").is_file()
    assert (out_dir / "iou_white.txt").read_text() == "0.600000"
    assert (out_dir / "iou_black.txt").read_text() == "0.800000"
    assert (out_dir / "iou_boundary.txt").read_text() == "0.300000"


def test_main_logs_to_mlflow_when_asked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_training(monkeypatch)
    fake = _FakeMlflow()
    monkeypatch.setattr("pianocv.trainkeyseg.mlflow", fake)
    monkeypatch.setattr("sys.argv", ["trainkeyseg", "--out-dir", str(tmp_path / "out"), "--mlflow"])
    main()
    assert fake.experiment == "keyseg"
    assert fake.metrics[0][1]["iou_black"] == 0.8
    assert len(fake.artifacts) == 2


def test_main_previews_without_training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_frames(tmp_path, 1)
    monkeypatch.setattr(
        "sys.argv",
        ["trainkeyseg", "--data-dir", str(tmp_path), "--out-dir", str(tmp_path), "--preview", "1"],
    )
    main()
    assert (tmp_path / "keyseg-preview-0.png").is_file()
