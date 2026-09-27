"""Pack the Seg2 training inputs and upload them to an S3 bucket with the MinIO client."""

import argparse
import gzip
import shutil
import subprocess  # nosec B404
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from pianocv.bake import DEFAULT_CORPUS_DIR
from pianocv.realseg2 import DEFAULT_OUTPUT_DIR as DEFAULT_REAL_DIR

DEFAULT_PIANOCV_DIR = Path(__file__).resolve().parent
_BUNDLE_STEM = "bundle.tar"


def _skip_pycache(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
    return None if "__pycache__" in info.name.split("/") else info


def build_bundle(corpus_dir: Path, real_dir: Path, pianocv_dir: Path, tar_path: Path) -> Path:
    for name, path in (("corpus", corpus_dir), ("real-seg2", real_dir), ("pianocv", pianocv_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {name} directory: {path}")
    with tarfile.open(tar_path, mode="w") as archive:
        archive.add(corpus_dir, arcname="corpus", filter=_skip_pycache)
        archive.add(real_dir, arcname="real-seg2", filter=_skip_pycache)
        archive.add(pianocv_dir, arcname="pianocv", filter=_skip_pycache)
    return tar_path


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


def push(
    corpus_dir: Path,
    real_dir: Path,
    pianocv_dir: Path,
    version: str,
    *,
    alias: str,
    bucket: str = "datasets",
) -> str:
    key = f"keybed/seg2-{version}/"
    if _destination_taken(alias, bucket, key):
        raise FileExistsError(f"{alias}/{bucket}/{key} already has a bundle, pick a new version")
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        build_bundle(corpus_dir, real_dir, pianocv_dir, work / _BUNDLE_STEM)
        bundle_path = _compress(work / _BUNDLE_STEM, work)
        _mc_cp(bundle_path, f"{alias}/{bucket}/{key}{bundle_path.name}")
        name = bundle_path.name
    return f"s3://{bucket}/{key}{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pack and push the seg2 training bundle to an S3 bucket"
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--pianocv-dir", type=Path, default=DEFAULT_PIANOCV_DIR)
    parser.add_argument("--version", default=date.today().isoformat())
    parser.add_argument("--bucket", default="datasets")
    parser.add_argument("--alias", required=True, help="the mc alias of the S3 host")
    args = parser.parse_args()
    uri = push(
        args.corpus_dir,
        args.real_dir,
        args.pianocv_dir,
        args.version,
        alias=args.alias,
        bucket=args.bucket,
    )
    print(uri)


if __name__ == "__main__":
    main()
