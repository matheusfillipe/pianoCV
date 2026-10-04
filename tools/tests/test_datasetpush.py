import gzip
import subprocess
from pathlib import Path

import pytest

from pianocv.datasetpush import (
    _compress,
    _destination_taken,
    _mc_cp,
)


def test_destination_taken_ignores_an_empty_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    empty = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: "/usr/bin/mc")
    monkeypatch.setattr("pianocv.datasetpush.subprocess.run", lambda *_args, **_kwargs: empty)

    assert _destination_taken("store", "datasets", "keybed/seg2-none/") is False


def test_destination_taken_detects_an_existing_object(monkeypatch: pytest.MonkeyPatch) -> None:
    listed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=b"bundle.tar.zst\n", stderr=b""
    )
    monkeypatch.setattr("pianocv.datasetpush.shutil.which", lambda name: "/usr/bin/mc")
    monkeypatch.setattr("pianocv.datasetpush.subprocess.run", lambda *_args, **_kwargs: listed)

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
