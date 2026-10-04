import tarfile
from pathlib import Path

import pytest

from pianocv.keysegpush import build_keyseg_bundle, main, push_keyseg


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    real_dir, pianocv_dir = tmp_path / "real-keys", tmp_path / "pianocv"
    _write(real_dir / "rec-00.json", "{}")
    _write(real_dir / "rec-00.png")
    _write(real_dir / "rec-00.ignore.png")
    _write(pianocv_dir / "trainkeyseg.py", "# module")
    _write(pianocv_dir / "__pycache__" / "trainkeyseg.cpython-312.pyc", "compiled")
    init_onnx = tmp_path / "keyseg.onnx"
    _write(init_onnx, "onnx")
    return real_dir, init_onnx, pianocv_dir


def test_bundle_holds_the_real_frames_the_code_and_the_starting_export(tmp_path: Path) -> None:
    real_dir, init_onnx, pianocv_dir = _sources(tmp_path)
    tar_path = build_keyseg_bundle(real_dir, init_onnx, pianocv_dir, tmp_path / "bundle.tar")

    with tarfile.open(tar_path) as archive:
        names = set(archive.getnames())
    assert {"real-keys/rec-00.json", "real-keys/rec-00.ignore.png", "keyseg.onnx"} <= names
    assert "pianocv/trainkeyseg.py" in names
    assert not any("__pycache__" in name for name in names)


def test_bundle_needs_the_starting_export(tmp_path: Path) -> None:
    real_dir, _, pianocv_dir = _sources(tmp_path)
    with pytest.raises(FileNotFoundError, match="starting export"):
        build_keyseg_bundle(real_dir, tmp_path / "none.onnx", pianocv_dir, tmp_path / "b.tar")


def test_push_uploads_the_bundle_under_a_new_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_dir, init_onnx, pianocv_dir = _sources(tmp_path)
    uploaded: list[str] = []
    monkeypatch.setattr("pianocv.keysegpush._destination_taken", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(
        "pianocv.keysegpush._mc_cp", lambda _path, destination: uploaded.append(destination)
    )

    uri = push_keyseg(real_dir, init_onnx, pianocv_dir, "2026-09-28", alias="store")

    assert uri.startswith("s3://datasets/keybed/keyseg-real-2026-09-28/bundle.tar.")
    assert uploaded[0].startswith("store/datasets/keybed/keyseg-real-2026-09-28/")


def test_push_refuses_a_version_already_there(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("pianocv.keysegpush._destination_taken", lambda *_args, **_kwargs: True)
    with pytest.raises(FileExistsError, match="already has a bundle"):
        push_keyseg(tmp_path, tmp_path, tmp_path, "taken", alias="store")


def test_bundle_packs_the_corrected_labels_when_the_directory_exists(tmp_path: Path) -> None:
    real_dir, init_onnx, pianocv_dir = _sources(tmp_path)
    fixed_dir = tmp_path / "real-keys-fixed"
    _write(fixed_dir / "a.json", "{}")
    tar_path = build_keyseg_bundle(
        real_dir, init_onnx, pianocv_dir, tmp_path / "bundle.tar", fixed_dir=fixed_dir
    )
    with tarfile.open(tar_path) as archive:
        assert "real-keys-fixed/a.json" in archive.getnames()
    absent = build_keyseg_bundle(
        real_dir, init_onnx, pianocv_dir, tmp_path / "b.tar", fixed_dir=tmp_path / "none"
    )
    with tarfile.open(absent) as archive:
        assert not any(name.startswith("real-keys-fixed") for name in archive.getnames())


def test_bundle_packs_rendered_camera_paths_when_given(tmp_path: Path) -> None:
    real_dir, init_onnx, pianocv_dir = _sources(tmp_path)
    motion_dir = tmp_path / "synth-motion"
    _write(motion_dir / "motion-a-000.json", "{}")
    tar_path = build_keyseg_bundle(
        real_dir, init_onnx, pianocv_dir, tmp_path / "bundle.tar", motion_dir
    )
    with tarfile.open(tar_path) as archive:
        assert "synth-motion/motion-a-000.json" in archive.getnames()


def test_main_prints_the_uri_from_push(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "pianocv.keysegpush.push_keyseg",
        lambda *_args, **_kwargs: "s3://datasets/keybed/keyseg-real-x/bundle.tar.zst",
    )
    monkeypatch.setattr("sys.argv", ["keysegpush", "--alias", "store"])

    main()

    assert capsys.readouterr().out.strip() == "s3://datasets/keybed/keyseg-real-x/bundle.tar.zst"
