import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.keymatch import DEFAULT_DATA_DIR, STRIP_HEIGHT, STRIP_WIDTH, ChannelMetrics, SidecarKey
from pianocv.trainkeymatch import (
    DEFAULT_OUT_DIR,
    EpochMetrics,
    _augment_strip,
    _build_parser,
    _require_mlflow,
    main,
    train_keymatch,
)

_WIDTH, _HEIGHT = 640, 480
FAR_LEFT, FAR_RIGHT = (64.0, 96.0), (576.0, 96.0)
NEAR_LEFT, NEAR_RIGHT = (64.0, 384.0), (576.0, 384.0)
QUAD_PX = np.array([FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT])
BOARD_WIDTH = 3.0


def _face(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
    def point(u: float, v: float) -> tuple[float, float]:
        x = FAR_LEFT[0] + (u / BOARD_WIDTH) * (FAR_RIGHT[0] - FAR_LEFT[0])
        y = FAR_LEFT[1] + v * (NEAR_LEFT[1] - FAR_LEFT[1])
        return x, y

    return np.array([point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)])


def _synth_frame_keys() -> list[SidecarKey]:
    white = [
        SidecarKey(
            pitch=60 + 2 * i, black=False, top=_face(float(i), float(i + 1), 0.0, 1.0), front=None
        )
        for i in range(3)
    ]
    black = [
        SidecarKey(
            pitch=pitch,
            black=True,
            top=_face(boundary - 0.3, boundary + 0.3, 0.05, 0.35),
            front=None,
        )
        for pitch, boundary in ((61, 1.0), (63, 2.0))
    ]
    return [*white, *black]


def _face_fraction(face_px: np.ndarray, scale: np.ndarray) -> list[dict[str, float]]:
    return [{"x": float(x), "y": float(y)} for x, y in (face_px / scale)]


def _write_synth_keys_dir(path: Path, stems: list[str]) -> None:
    scale = np.array([float(_WIDTH), float(_HEIGHT)])
    keys = _synth_frame_keys()
    for stem in stems:
        cv2.imwrite(str(path / f"{stem}.png"), np.full((_HEIGHT, _WIDTH, 3), 100, dtype=np.uint8))
        sidecar = {
            "corners": _face_fraction(QUAD_PX, scale),
            "imageWidth": _WIDTH,
            "imageHeight": _HEIGHT,
            "keys": [
                {
                    "pitch": key.pitch,
                    "black": key.black,
                    "top": _face_fraction(key.top, scale),
                    "front": None if key.front is None else _face_fraction(key.front, scale),
                }
                for key in keys
            ],
        }
        (path / f"{stem}.json").write_text(json.dumps(sidecar))


def test_parser_uses_the_documented_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.data_dir == DEFAULT_DATA_DIR
    assert args.out_dir == DEFAULT_OUT_DIR
    assert (args.epochs, args.batch, args.lr, args.seed) == (25, 32, 3e-4, 0)
    assert (args.steps_per_epoch, args.workers) == (100, 4)
    assert args.mlflow is False
    assert args.mlflow_experiment == "keymatch"


def test_parser_reads_mlflow_experiment_default_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-experiment")

    args = _build_parser().parse_args(["--mlflow"])

    assert args.mlflow_experiment == "custom-experiment"
    assert args.mlflow is True


def test_require_mlflow_raises_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pianocv.trainkeymatch.mlflow", None)
    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        _require_mlflow()


def test_augment_strip_covers_every_augmentation_branch() -> None:
    strip = np.full((STRIP_HEIGHT, STRIP_WIDTH, 3), 128, dtype=np.uint8)
    for seed in range(60):
        augmented = _augment_strip(strip, np.random.default_rng(seed))
        assert augmented.shape == strip.shape
        assert augmented.dtype == np.uint8


def test_train_keymatch_raises_without_frames(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no synth-keys frames found"):
        train_keymatch(data_dir=tmp_path, epochs=1, batch=2, steps_per_epoch=1, workers=0)


def test_train_keymatch_runs_a_tiny_schedule(tmp_path: Path) -> None:
    _write_synth_keys_dir(tmp_path, ["a", "b", "c", "d"])
    collected: list[EpochMetrics] = []

    model, history = train_keymatch(
        data_dir=tmp_path,
        epochs=2,
        batch=2,
        steps_per_epoch=2,
        workers=0,
        on_epoch=collected.append,
    )

    assert len(history) == 2
    assert collected == history
    metrics = history[-1]
    assert metrics.loss >= 0.0
    assert metrics.val_loss >= 0.0
    for channel in (metrics.white, metrics.black_left, metrics.black_right):
        assert 0.0 <= channel.precision <= 1.0
        assert 0.0 <= channel.recall <= 1.0
        assert channel.mean_error_px >= 0.0
    assert not model.training


class _FakeModel:
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
        self.params: dict[str, object] = {}
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.artifacts: list[str] = []

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self) -> _FakeRun:
        return _FakeRun()

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((step, metrics))

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)


def _fake_metrics() -> EpochMetrics:
    return EpochMetrics(
        epoch=1,
        loss=0.5,
        val_loss=0.4,
        white=ChannelMetrics(0.9, 0.8, 1.1),
        black_left=ChannelMetrics(0.7, 0.6, 1.2),
        black_right=ChannelMetrics(0.5, 0.4, 1.3),
    )


def _fake_train_keymatch(*_args: object, **kwargs: object) -> tuple[_FakeModel, list[EpochMetrics]]:
    metrics = _fake_metrics()
    on_epoch = kwargs.get("on_epoch")
    if callable(on_epoch):
        on_epoch(metrics)
    return _FakeModel(), [metrics]


def _fake_export_keymatch_onnx(_model_path: Path, onnx_path: Path) -> Path:
    onnx_path.write_bytes(b"onnx")
    return onnx_path


def test_main_writes_outputs_without_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setattr("pianocv.trainkeymatch.train_keymatch", _fake_train_keymatch)
    monkeypatch.setattr("pianocv.trainkeymatch.export_keymatch_onnx", _fake_export_keymatch_onnx)
    monkeypatch.setattr(
        "pianocv.trainkeymatch.torch.save", lambda state_dict, path: Path(path).write_bytes(b"pt")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["trainkeymatch", "--out-dir", str(out_dir), "--data-dir", str(tmp_path)],
    )

    main()

    assert (out_dir / "keymatch.pt").is_file()
    assert (out_dir / "keymatch.onnx").is_file()
    assert (out_dir / "white_precision.txt").read_text() == "0.900000"
    assert (out_dir / "white_recall.txt").read_text() == "0.800000"
    assert (out_dir / "white_mean_error_px.txt").read_text() == "1.100000"
    assert (out_dir / "black_left_precision.txt").read_text() == "0.700000"
    assert (out_dir / "black_left_recall.txt").read_text() == "0.600000"
    assert (out_dir / "black_left_mean_error_px.txt").read_text() == "1.200000"
    assert (out_dir / "black_right_precision.txt").read_text() == "0.500000"
    assert (out_dir / "black_right_recall.txt").read_text() == "0.400000"
    assert (out_dir / "black_right_mean_error_px.txt").read_text() == "1.300000"


def test_main_logs_to_mlflow_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("pianocv.trainkeymatch.mlflow", fake_mlflow)
    monkeypatch.setattr("pianocv.trainkeymatch.train_keymatch", _fake_train_keymatch)
    monkeypatch.setattr("pianocv.trainkeymatch.export_keymatch_onnx", _fake_export_keymatch_onnx)
    monkeypatch.setattr(
        "pianocv.trainkeymatch.torch.save", lambda state_dict, path: Path(path).write_bytes(b"pt")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["trainkeymatch", "--out-dir", str(out_dir), "--data-dir", str(tmp_path), "--mlflow"],
    )

    main()

    assert fake_mlflow.experiment == "keymatch"
    assert fake_mlflow.params["epochs"] == 25
    assert fake_mlflow.metrics == [
        (
            1,
            {
                "loss": 0.5,
                "val_loss": 0.4,
                "white_precision": 0.9,
                "white_recall": 0.8,
                "white_mean_error_px": 1.1,
                "black_left_precision": 0.7,
                "black_left_recall": 0.6,
                "black_left_mean_error_px": 1.2,
                "black_right_precision": 0.5,
                "black_right_recall": 0.4,
                "black_right_mean_error_px": 1.3,
            },
        )
    ]
    assert len(fake_mlflow.artifacts) == 2
