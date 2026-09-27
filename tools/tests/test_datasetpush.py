import gzip
import subprocess
import tarfile
from pathlib import Path

import pytest

from pianocv.datasetpush import (
    _compress,
    _destination_taken,
    _mc_cp,
    build_bundle,
    main,
    push,
)


def _write(path: Path, content: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_build_bundle_lays_out_the_three_directories_at_the_tar_root(tmp_path: Path) -> None:
    corpus_dir, real_dir, pianocv_dir = (
        tmp_path / "corpus",
        tmp_path / "real-seg2",
        tmp_path / "pianocv",
    )
    _write(corpus_dir / "frames" / "a.png")
    _write(corpus_dir / "corners.json", "{}")
    _write(real_dir / "frames" / "b.png")
    _write(real_dir / "splits.json", "{}")
    _write(pianocv_dir / "segnet2.py", "# module")
    tar_path = tmp_path / "bundle.tar"

    build_bundle(corpus_dir, real_dir, pianocv_dir, tar_path)

    with tarfile.open(tar_path) as archive:
        names = set(archive.getnames())
    assert "corpus/frames/a.png" in names
    assert "corpus/corners.json" in names
    assert "real-seg2/frames/b.png" in names
    assert "real-seg2/splits.json" in names
    assert "pianocv/segnet2.py" in names


def test_build_bundle_excludes_pycache(tmp_path: Path) -> None:
    corpus_dir, real_dir, pianocv_dir = (
        tmp_path / "corpus",
        tmp_path / "real-seg2",
        tmp_path / "pianocv",
    )
    _write(corpus_dir / "corners.json", "{}")
    _write(real_dir / "splits.json", "{}")
    _write(pianocv_dir / "segnet2.py", "# module")
    _write(pianocv_dir / "__pycache__" / "segnet2.cpython-312.pyc", "compiled")
    tar_path = tmp_path / "bundle.tar"

    build_bundle(corpus_dir, real_dir, pianocv_dir, tar_path)

    with tarfile.open(tar_path) as archive:
        names = archive.getnames()
    assert not any("__pycache__" in name for name in names)


def test_build_bundle_rejects_a_missing_source_directory(tmp_path: Path) -> None:
    corpus_dir, real_dir, pianocv_dir = (
        tmp_path / "corpus",
        tmp_path / "real-seg2",
        tmp_path / "pianocv",
    )
    _write(real_dir / "splits.json", "{}")
    _write(pianocv_dir / "segnet2.py", "# module")

    with pytest.raises(FileNotFoundError, match="corpus"):
        build_bundle(corpus_dir, real_dir, pianocv_dir, tmp_path / "bundle.tar")


def test_destination_taken_ignores_an_empty_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: "/usr/bin/mc")
    monkeypatch.setattr("pianocv.datasetpush.subprocess.run", lambda *a, **k: empty)

    assert _destination_taken("store", "datasets", "keybed/seg2-none/") is False


def test_destination_taken_detects_an_existing_object(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"bundle.tar.zst\n", stderr=b""
    )
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: "/usr/bin/mc")
    monkeypatch.setattr("pianocv.datasetpush.subprocess.run", lambda *a, **k: listed)

    assert _destination_taken("store", "datasets", "keybed/seg2-taken/") is True


def test_destination_taken_requires_mc_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: None)

    with pytest.raises(FileNotFoundError, match="mc binary"):
        _destination_taken("store", "datasets", "keybed/seg2-taken/")


def test_compress_uses_zstd_when_present(tmp_path: Path) -> None:
    tar_path = tmp_path / "bundle.tar"
    tar_path.write_bytes(b"not really a tar, just bytes to compress")

    target = _compress(tar_path, tmp_path)

    assert target == tmp_path / "bundle.tar.zst"
    assert target.is_file()


def test_compress_falls_back_to_gzip_without_zstd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: None)
    tar_path = tmp_path / "bundle.tar"
    payload = b"tar bytes to round-trip through gzip"
    tar_path.write_bytes(payload)

    target = _compress(tar_path, tmp_path)

    assert target == tmp_path / "bundle.tar.gz"
    with gzip.open(target, "rb") as compressed:
        assert compressed.read() == payload


def test_mc_cp_requires_mc_on_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: None)

    with pytest.raises(FileNotFoundError, match="mc binary"):
        _mc_cp(tmp_path / "bundle.tar.zst", "store/datasets/keybed/seg2-x/bundle.tar.zst")


def test_mc_cp_invokes_mc_with_the_expected_argv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: "/usr/bin/mc")
    monkeypatch.setattr(
        "pianocv.datasetpush.subprocess.run",
        lambda argv, **_kwargs: calls.append(argv),
    )

    _mc_cp(tmp_path / "bundle.tar.zst", "store/datasets/keybed/seg2-x/bundle.tar.zst")

    assert calls == [
        [
            "/usr/bin/mc",
            "cp",
            str(tmp_path / "bundle.tar.zst"),
            "store/datasets/keybed/seg2-x/bundle.tar.zst",
        ]
    ]


def test_push_rejects_an_existing_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("pianocv.datasetpush._destination_taken", lambda *a, **k: True)

    with pytest.raises(FileExistsError, match="already has a bundle"):
        push(tmp_path, tmp_path, tmp_path, "taken", alias="store")


def test_push_builds_compresses_and_uploads_the_bundle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    corpus_dir, real_dir, pianocv_dir = (
        tmp_path / "corpus",
        tmp_path / "real-seg2",
        tmp_path / "pianocv",
    )
    _write(corpus_dir / "corners.json", "{}")
    _write(real_dir / "splits.json", "{}")
    _write(pianocv_dir / "segnet2.py", "# module")
    uploaded: list[tuple[Path, str]] = []
    monkeypatch.setattr("pianocv.datasetpush._destination_taken", lambda *a, **k: False)
    monkeypatch.setattr(
        "pianocv.datasetpush._mc_cp",
        lambda local_path, destination: uploaded.append((local_path, destination)),
    )

    uri = push(corpus_dir, real_dir, pianocv_dir, "2026-09-20", bucket="datasets", alias="store")

    assert uri.startswith("s3://datasets/keybed/seg2-2026-09-20/bundle.tar.")
    assert len(uploaded) == 1
    assert uploaded[0][1] == f"store/datasets/keybed/seg2-2026-09-20/{uploaded[0][0].name}"


def test_main_prints_the_uri_from_push(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        "pianocv.datasetpush.push", lambda *a, **k: "s3://datasets/keybed/seg2-x/bundle.tar.zst"
    )
    monkeypatch.setattr("sys.argv", ["datasetpush", "--alias", "store"])

    main()

    assert capsys.readouterr().out.strip() == "s3://datasets/keybed/seg2-x/bundle.tar.zst"
