"""Pack the KeyMatch training inputs and upload them to an S3 bucket with the MinIO client."""

import argparse
import tarfile
import tempfile
from datetime import date
from pathlib import Path

from pianocv.datasetpush import _compress, _destination_taken, _mc_cp, _skip_pycache
from pianocv.keymatch import DEFAULT_DATA_DIR

DEFAULT_PIANOCV_DIR = Path(__file__).resolve().parent
_BUNDLE_STEM = "bundle.tar"


def build_keymatch_bundle(data_dir: Path, pianocv_dir: Path, tar_path: Path) -> Path:
    for name, path in (("synth-keys", data_dir), ("pianocv", pianocv_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"missing {name} directory: {path}")
    with tarfile.open(tar_path, mode="w") as archive:
        archive.add(data_dir, arcname="synth-keys", filter=_skip_pycache)
        archive.add(pianocv_dir, arcname="pianocv", filter=_skip_pycache)
    return tar_path


def push_keymatch(
    data_dir: Path,
    pianocv_dir: Path,
    version: str,
    *,
    alias: str,
    bucket: str = "datasets",
) -> str:
    key = f"keybed/keymatch-{version}/"
    if _destination_taken(alias, bucket, key):
        raise FileExistsError(f"{alias}/{bucket}/{key} already has a bundle, pick a new version")
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        build_keymatch_bundle(data_dir, pianocv_dir, work / _BUNDLE_STEM)
        bundle_path = _compress(work / _BUNDLE_STEM, work)
        _mc_cp(bundle_path, f"{alias}/{bucket}/{key}{bundle_path.name}")
        name = bundle_path.name
    return f"s3://{bucket}/{key}{name}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pack and push the keymatch training bundle to an S3 bucket"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--pianocv-dir", type=Path, default=DEFAULT_PIANOCV_DIR)
    parser.add_argument("--version", default=date.today().isoformat())
    parser.add_argument("--bucket", default="datasets")
    parser.add_argument("--alias", required=True, help="the mc alias of the S3 host")
    args = parser.parse_args()
    uri = push_keymatch(
        args.data_dir, args.pianocv_dir, args.version, alias=args.alias, bucket=args.bucket
    )
    print(uri)


if __name__ == "__main__":
    main()
