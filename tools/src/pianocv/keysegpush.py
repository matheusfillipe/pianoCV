"""Pack what a training run adds to the synthetic bundle and upload it with the MinIO client: the
real frames the app labelled itself, rendered camera paths when there are any, the pianocv source
that trains on them, and the keyseg export training starts from."""

import argparse
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from pianocv.datasetpush import _compress, _destination_taken, _mc_cp, _skip_pycache

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REAL_DIR = _REPO_ROOT / "data" / "real-keys"
DEFAULT_INIT_ONNX = _REPO_ROOT / "web" / "public" / "keyseg.onnx"
DEFAULT_PIANOCV_DIR = Path(__file__).resolve().parent
DEFAULT_FIXED_DIR = _REPO_ROOT / "data" / "real-keys-fixed"
FIXED_ARCNAME = "real-keys-fixed"
_BUNDLE_STEM = "bundle.tar"


def build_keyseg_bundle(
    real_dir: Path,
    init_onnx: Path,
    pianocv_dir: Path,
    tar_path: Path,
    motion_dir: Path | None = None,
    fixed_dir: Path | None = None,
) -> Path:
    for name, path in (
        ("real-keys", real_dir),
        ("pianocv", pianocv_dir),
        *((("synth-motion", motion_dir),) if motion_dir is not None else ()),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {name} directory: {path}")
    if not init_onnx.is_file():
        raise FileNotFoundError(f"missing starting export: {init_onnx}")
    with tarfile.open(tar_path, mode="w") as archive:
        archive.add(real_dir, arcname="real-keys", filter=_skip_pycache)
        archive.add(pianocv_dir, arcname="pianocv", filter=_skip_pycache)
        archive.add(init_onnx, arcname="keyseg.onnx")
        if motion_dir is not None:
            archive.add(motion_dir, arcname="synth-motion", filter=_skip_pycache)
        if fixed_dir is not None and fixed_dir.is_dir():
            archive.add(fixed_dir, arcname=FIXED_ARCNAME, filter=_skip_pycache)
    return tar_path


def push_keyseg(
    real_dir: Path,
    init_onnx: Path,
    pianocv_dir: Path,
    version: str,
    *,
    alias: str,
    bucket: str = "datasets",
    motion_dir: Path | None = None,
    name: str = "keyseg-real",
    fixed_dir: Path | None = None,
) -> str:
    key = f"keybed/{name}-{version}/"
    if _destination_taken(alias, bucket, key):
        raise FileExistsError(f"{alias}/{bucket}/{key} already has a bundle, pick a new version")
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        build_keyseg_bundle(
            real_dir, init_onnx, pianocv_dir, work / _BUNDLE_STEM, motion_dir, fixed_dir
        )
        bundle_path = _compress(work / _BUNDLE_STEM, work)
        _mc_cp(bundle_path, f"{alias}/{bucket}/{key}{bundle_path.name}")
        name = bundle_path.name
    return f"s3://{bucket}/{key}{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pack and push the real frames and code a keyseg fine-tune needs"
    )
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--init-onnx", type=Path, default=DEFAULT_INIT_ONNX)
    parser.add_argument("--pianocv-dir", type=Path, default=DEFAULT_PIANOCV_DIR)
    parser.add_argument("--motion-dir", type=Path, help="rendered camera paths to pack too")
    parser.add_argument("--fixed-dir", type=Path, default=DEFAULT_FIXED_DIR)
    parser.add_argument("--name", default="keyseg-real", help="the bundle's key prefix")
    parser.add_argument("--version", default=date.today().isoformat())
    parser.add_argument("--bucket", default="datasets")
    parser.add_argument("--alias", required=True, help="the mc alias of the S3 host")
    args = parser.parse_args()
    uri = push_keyseg(
        args.real_dir,
        args.init_onnx,
        args.pianocv_dir,
        args.version,
        alias=args.alias,
        bucket=args.bucket,
        motion_dir=args.motion_dir,
        name=args.name,
        fixed_dir=args.fixed_dir,
    )
    print(uri)


if __name__ == "__main__":
    main()
