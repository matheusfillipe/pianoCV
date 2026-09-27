import tarfile
from pathlib import Path

import pytest

from pianocv.keymatchpush import build_keymatch_bundle, main, push_keymatch


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_build_keymatch_bundle_lays_out_both_directories_at_the_tar_root(tmp_path: Path) -> None:
    data_dir, pianocv_dir = tmp_path / "synth-keys", tmp_path / "pianocv"
    _write(data_dir / "a.json", "{}")
    _write(data_dir / "a.png")
    _write(pianocv_dir / "keymatch.py", "# module")
    tar_path = tmp_path / "bundle.tar"

    build_keymatch_bundle(data_dir, pianocv_dir, tar_path)

    with tarfile.open(tar_path) as archive:
        names = set(archive.getnames())
    assert "synth-keys/a.json" in names
    assert "synth-keys/a.png" in names
    assert "pianocv/keymatch.py" in names


def test_build_keymatch_bundle_excludes_pycache(tmp_path: Path) -> None:
    data_dir, pianocv_dir = tmp_path / "synth-keys", tmp_path / "pianocv"
    _write(data_dir / "a.json", "{}")
    _write(pianocv_dir / "keymatch.py", "# module")
    _write(pianocv_dir / "__pycache__" / "keymatch.cpython-312.pyc", "compiled")
    tar_path = tmp_path / "bundle.tar"

    build_keymatch_bundle(data_dir, pianocv_dir, tar_path)

    with tarfile.open(tar_path) as archive:
        names = archive.getnames()
    assert not any("__pycache__" in name for name in names)


def test_build_keymatch_bundle_rejects_a_missing_source_directory(tmp_path: Path) -> None:
    pianocv_dir = tmp_path / "pianocv"
    _write(pianocv_dir / "keymatch.py", "# module")

    with pytest.raises(FileNotFoundError, match="synth-keys"):
        build_keymatch_bundle(tmp_path / "synth-keys", pianocv_dir, tmp_path / "bundle.tar")


def test_push_keymatch_rejects_an_existing_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("pianocv.keymatchpush._destination_taken", lambda *a, **k: True)

    with pytest.raises(FileExistsError, match="already has a bundle"):
        push_keymatch(tmp_path, tmp_path, "taken", alias="store")


def test_push_keymatch_builds_compresses_and_uploads_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data_dir, pianocv_dir = tmp_path / "synth-keys", tmp_path / "pianocv"
    _write(data_dir / "a.json", "{}")
    _write(pianocv_dir / "keymatch.py", "# module")
    uploaded: list[tuple[Path, str]] = []
    monkeypatch.setattr("pianocv.keymatchpush._destination_taken", lambda *a, **k: False)
    monkeypatch.setattr(
        "pianocv.keymatchpush._mc_cp",
        lambda local_path, destination: uploaded.append((local_path, destination)),
    )

    uri = push_keymatch(data_dir, pianocv_dir, "2026-09-20", bucket="datasets", alias="store")

    assert uri.startswith("s3://datasets/keybed/keymatch-2026-09-20/bundle.tar.")
    assert len(uploaded) == 1
    assert uploaded[0][1] == f"store/datasets/keybed/keymatch-2026-09-20/{uploaded[0][0].name}"


def test_main_prints_the_uri_from_push_keymatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "pianocv.keymatchpush.push_keymatch",
        lambda *a, **k: "s3://datasets/keybed/keymatch-x/bundle.tar.zst",
    )
    monkeypatch.setattr("sys.argv", ["keymatchpush", "--alias", "store"])

    main()

    assert capsys.readouterr().out.strip() == "s3://datasets/keybed/keymatch-x/bundle.tar.zst"
