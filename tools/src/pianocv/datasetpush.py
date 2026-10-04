"""Bundle compression and MinIO client helpers shared by the bundle upload scripts."""

import gzip
import shutil
import subprocess  # nosec B404
import tarfile
from pathlib import Path

_BUNDLE_STEM = "bundle.tar"


def _skip_pycache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name.split("/") else info


def _compress(tar_path: Path, out_dir: Path) -> Path:
    zstd = shutil.which("zstd")
    if zstd is not None:
        target = out_dir / f"{_BUNDLE_STEM}.zst"
        subprocess.run([zstd, "-q", "-f", "-o", str(target), str(tar_path)], check=True)  # nosec B603
        return target
    target = out_dir / f"{_BUNDLE_STEM}.gz"
    with tar_path.open("rb") as raw, gzip.open(target, "wb") as compressed:
        shutil.copyfileobj(raw, compressed)
    return target


def _mc_cp(local_path: Path, destination: str) -> None:
    mc = shutil.which("mc")
    if mc is None:
        raise FileNotFoundError("mc binary not found on PATH")
    subprocess.run([mc, "cp", str(local_path), destination], check=True)  # nosec B603


def _destination_taken(alias: str, bucket: str, key: str) -> bool:
    mc = shutil.which("mc")
    if mc is None:
        raise FileNotFoundError("mc binary not found on PATH")
    result = subprocess.run(  # nosec B603
        [mc, "ls", f"{alias}/{bucket}/{key}"], capture_output=True, check=False
    )
    # mc ls exits 0 for an empty prefix too, so only a bucket typo or missing bucket fails outright
    return result.returncode == 0 and bool(result.stdout.strip())
