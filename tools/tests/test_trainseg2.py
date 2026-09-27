import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.trainseg2 import (
    DEFAULT_OUT_DIR,
    DEFAULT_REAL_DIR,
    DEFAULT_SYNTHETIC_DIR,
    EpochMetrics,
    _build_parser,
    _dataset_source_uri,
    _load_real,
    _load_synthetic,
    _require_mlflow,
    main,
    mask_from_quad,
    train_seg2,
    validate_real_splits,
)

QUAD = [[0.1, 0.1], [0.6, 0.1], [0.6, 0.5], [0.1, 0.5]]
LABELS: dict[str, object] = {"a.png": {}, "b.png": {}, "c.png": {}}


def test_parser_uses_the_documented_defaults() -> None:
    args = _build_parser().parse_args([])
    assert args.synthetic_dir == DEFAULT_SYNTHETIC_DIR
    assert args.real_dir == DEFAULT_REAL_DIR
    assert args.out_dir == DEFAULT_OUT_DIR
    assert (args.epochs, args.batch, args.lr, args.seed) == (25, 48, 3e-4, 0)
    assert args.synthetic_colour is True
    assert args.mlflow is False
    assert args.dataset_bucket == "datasets"
    assert args.dataset_key == ""


def test_parser_no_synthetic_colour_switches_to_grayscale() -> None:
    args = _build_parser().parse_args(["--no-synthetic-colour"])
    assert args.synthetic_colour is False


def test_parser_reads_dataset_and_experiment_defaults_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATASET_BUCKET", "custom-bucket")
    monkeypatch.setenv("DATASET_KEY", "keybed/seg2-2026-09-20/bundle.tar.zst")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "custom-experiment")

    args = _build_parser().parse_args(["--mlflow"])

    assert args.dataset_bucket == "custom-bucket"
    assert args.dataset_key == "keybed/seg2-2026-09-20/bundle.tar.zst"
    assert args.mlflow_experiment == "custom-experiment"
    assert args.mlflow is True


def test_parser_accepts_explicit_flags_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATASET_BUCKET", "ignored")

    args = _build_parser().parse_args(
        [
            "--epochs",
            "3",
            "--batch",
            "8",
            "--lr",
            "0.01",
            "--seed",
            "7",
            "--dataset-bucket",
            "explicit",
        ]
    )

    assert (args.epochs, args.batch, args.lr, args.seed) == (3, 8, 0.01, 7)
    assert args.dataset_bucket == "explicit"


def test_mask_from_quad_fills_only_the_quad_region() -> None:
    mask = mask_from_quad(QUAD, mask_size=64, raster=256)
    assert mask.shape == (64, 64)
    assert mask.dtype == np.uint8
    inside = mask[20, 20]
    outside = mask[5, 60]
    assert inside > 0
    assert outside == 0


def test_mask_from_quad_area_matches_the_normalised_quad_fraction() -> None:
    mask = mask_from_quad(QUAD, mask_size=128, raster=512)
    width, height = 0.6 - 0.1, 0.5 - 0.1
    expected_fraction = width * height
    actual_fraction = float((mask > 127).sum()) / mask.size
    assert abs(actual_fraction - expected_fraction) < 0.02


def test_validate_real_splits_accepts_a_disjoint_full_cover() -> None:
    splits = {"train": ["a.png"], "validation": ["b.png"], "held_out": ["c.png"]}
    validate_real_splits(splits, LABELS)


def test_validate_real_splits_rejects_overlapping_members() -> None:
    splits = {"train": ["a.png", "b.png"], "validation": ["b.png"], "held_out": ["c.png"]}
    with pytest.raises(ValueError, match="disjoint"):
        validate_real_splits(splits, LABELS)


def test_validate_real_splits_rejects_an_incomplete_cover() -> None:
    splits = {"train": ["a.png"], "validation": ["b.png"], "held_out": []}
    with pytest.raises(ValueError, match="disjoint"):
        validate_real_splits(splits, LABELS)


def test_validate_real_splits_rejects_a_missing_required_split() -> None:
    splits = {"train": ["a.png"], "validation": ["b.png", "c.png"]}
    with pytest.raises(ValueError, match="must include"):
        validate_real_splits(splits, LABELS)


def test_load_synthetic_colour_flag_picks_the_read_mode(tmp_path: Path) -> None:
    synthetic_dir = tmp_path / "synthetic"
    (synthetic_dir / "frames").mkdir(parents=True)
    blue = np.zeros((20, 20, 3), dtype=np.uint8)
    blue[:, :, 0] = 200
    cv2.imwrite(str(synthetic_dir / "frames" / "a.png"), blue)
    (synthetic_dir / "corners.json").write_text(json.dumps({"a": QUAD}))

    colour_images, _ = _load_synthetic(synthetic_dir, mask_size=16, raster=64, colour=True)
    gray_images, _ = _load_synthetic(synthetic_dir, mask_size=16, raster=64, colour=False)

    assert int(colour_images[0, 0, 0, 2]) > int(colour_images[0, 0, 0, 0])
    pixel = gray_images[0, 0, 0]
    assert pixel[0] == pixel[1] == pixel[2]


def _write_real_dir(path: Path, names_by_split: dict[str, list[str]]) -> None:
    (path / "frames").mkdir(parents=True)
    (path / "masks").mkdir(parents=True)
    labels: dict[str, object] = {}
    for names in names_by_split.values():
        for name in names:
            cv2.imwrite(str(path / "frames" / name), np.full((20, 20, 3), 80, dtype=np.uint8))
            mask = np.zeros((20, 20), dtype=np.uint8)
            mask[5:15, 5:15] = 255
            cv2.imwrite(str(path / "masks" / name), mask)
            labels[name] = {}
    (path / "splits.json").write_text(json.dumps(names_by_split))
    (path / "labels.json").write_text(json.dumps(labels))


def test_load_real_reads_every_split(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_real_dir(
        real_dir, {"train": ["a.png", "b.png"], "validation": ["c.png"], "held_out": ["d.png"]}
    )

    pairs = _load_real(real_dir)

    assert pairs["train"][0].shape[0] == 2
    assert pairs["validation"][0].shape[0] == 1
    assert pairs["held_out"][0].shape[0] == 1


def test_load_real_rejects_a_missing_frame(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    _write_real_dir(real_dir, {"train": ["a.png"], "validation": ["b.png"], "held_out": ["c.png"]})
    (real_dir / "frames" / "a.png").unlink()

    with pytest.raises(FileNotFoundError, match="missing real frame or mask"):
        _load_real(real_dir)


def test_dataset_source_uri_prefers_the_s3_key(tmp_path: Path) -> None:
    assert _dataset_source_uri("datasets", "keybed/seg2-x/bundle.tar.zst", tmp_path) == (
        "s3://datasets/keybed/seg2-x/bundle.tar.zst"
    )


def test_dataset_source_uri_falls_back_to_the_local_real_dir(tmp_path: Path) -> None:
    assert _dataset_source_uri("datasets", "", tmp_path) == tmp_path.resolve().as_uri()


def test_require_mlflow_raises_when_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pianocv.trainseg2.mlflow", None)
    with pytest.raises(RuntimeError, match="mlflow is not installed"):
        _require_mlflow()


def _write_synthetic_dir(path: Path, stems: list[str]) -> None:
    frames_dir = path / "frames"
    frames_dir.mkdir(parents=True)
    corners = {}
    for stem in stems:
        cv2.imwrite(str(frames_dir / f"{stem}.png"), np.full((20, 20, 3), 120, dtype=np.uint8))
        corners[stem] = QUAD
    (path / "corners.json").write_text(json.dumps(corners))


def test_train_seg2_runs_a_tiny_schedule(tmp_path: Path) -> None:
    synthetic_dir = tmp_path / "synthetic"
    _write_synthetic_dir(synthetic_dir, ["a", "b", "c", "d"])
    real_dir = tmp_path / "real"
    _write_real_dir(
        real_dir,
        {"train": ["train.png"], "validation": ["val.png"], "held_out": ["held.png"]},
    )

    model, history = train_seg2(
        synthetic_dir=synthetic_dir, real_dir=real_dir, epochs=1, batch=2, pretrained=False
    )

    assert len(history) == 1
    metrics = history[0]
    assert 0.0 <= metrics.synthetic_iou <= 1.0
    assert 0.0 <= metrics.real_val_iou <= 1.0
    assert 0.0 <= metrics.held_out_iou <= 1.0
    assert metrics.loss >= 0.0
    assert not model.training


class _FakeModel:
    def state_dict(self) -> dict[str, object]:
        return {}


class _FakeRun:
    def __enter__(self) -> "_FakeRun":
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None


class _FakeMlflowData:
    def from_numpy(self, array: np.ndarray, source: str, name: str) -> dict[str, object]:
        return {"source": source, "name": name, "shape": array.shape}


class _FakeMlflow:
    def __init__(self) -> None:
        self.data = _FakeMlflowData()
        self.experiment: str | None = None
        self.params: dict[str, object] = {}
        self.metrics: list[tuple[int, dict[str, float]]] = []
        self.inputs: list[tuple[dict[str, object], str]] = []
        self.artifacts: list[str] = []

    def set_experiment(self, name: str) -> None:
        self.experiment = name

    def start_run(self) -> _FakeRun:
        return _FakeRun()

    def log_params(self, params: dict[str, object]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, float], step: int) -> None:
        self.metrics.append((step, metrics))

    def log_input(self, dataset: dict[str, object], context: str) -> None:
        self.inputs.append((dataset, context))

    def log_artifact(self, path: str) -> None:
        self.artifacts.append(path)


def _fake_train_seg2(*args: object, **kwargs: object) -> tuple[_FakeModel, list[EpochMetrics]]:
    on_dataset_loaded = kwargs.get("on_dataset_loaded")
    if callable(on_dataset_loaded):
        on_dataset_loaded(np.zeros((1, 4, 4, 3), dtype=np.uint8))
    metrics = EpochMetrics(epoch=1, loss=0.5, synthetic_iou=0.9, real_val_iou=0.8, held_out_iou=0.7)
    on_epoch = kwargs.get("on_epoch")
    if callable(on_epoch):
        on_epoch(metrics)
    return _FakeModel(), [metrics]


def _fake_export_seg2_onnx(model_path: Path, onnx_path: Path) -> Path:
    onnx_path.write_bytes(b"onnx")
    return onnx_path


def test_main_writes_outputs_without_mlflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    monkeypatch.setattr("pianocv.trainseg2.train_seg2", _fake_train_seg2)
    monkeypatch.setattr("pianocv.trainseg2.export_seg2_onnx", _fake_export_seg2_onnx)
    monkeypatch.setattr(
        "pianocv.trainseg2.torch.save", lambda state_dict, path: Path(path).write_bytes(b"pt")
    )
    monkeypatch.setattr(
        "sys.argv",
        ["trainseg2", "--out-dir", str(out_dir), "--synthetic-dir", str(tmp_path)],
    )

    main()

    assert (out_dir / "keybed_seg2.pt").is_file()
    assert (out_dir / "keybed_seg2.onnx").is_file()
    assert (out_dir / "synthetic_iou.txt").read_text() == "0.900000"
    assert (out_dir / "real_val_iou.txt").read_text() == "0.800000"
    assert (out_dir / "held_out_iou.txt").read_text() == "0.700000"


def test_main_logs_to_mlflow_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "out"
    fake_mlflow = _FakeMlflow()
    monkeypatch.setattr("pianocv.trainseg2.mlflow", fake_mlflow)
    monkeypatch.setattr("pianocv.trainseg2.train_seg2", _fake_train_seg2)
    monkeypatch.setattr("pianocv.trainseg2.export_seg2_onnx", _fake_export_seg2_onnx)
    monkeypatch.setattr(
        "pianocv.trainseg2.torch.save", lambda state_dict, path: Path(path).write_bytes(b"pt")
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "trainseg2",
            "--out-dir",
            str(out_dir),
            "--synthetic-dir",
            str(tmp_path),
            "--mlflow",
            "--dataset-key",
            "keybed/seg2-x/bundle.tar.zst",
        ],
    )

    main()

    assert fake_mlflow.experiment == "keybed-seg2"
    assert fake_mlflow.params["synthetic_colour"] is True
    assert fake_mlflow.metrics == [
        (1, {"loss": 0.5, "synthetic_iou": 0.9, "real_val_iou": 0.8, "held_out_iou": 0.7})
    ]
    assert fake_mlflow.inputs[0][0]["source"] == "s3://datasets/keybed/seg2-x/bundle.tar.zst"
    assert len(fake_mlflow.artifacts) == 2
