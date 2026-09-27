"""Bake the rendered corpus down to what the net consumes, for upload to a training host."""

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from pianocv.dataset import DEFAULT_SYNTH_DIR, Frame, load_synth
from pianocv.model import INPUT_SIZE

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "corpus"


def _load_all(synth_dirs: Sequence[Path]) -> list[Frame]:
    frames: list[Frame] = []
    stem_origin: dict[str, Path] = {}
    for synth_dir in synth_dirs:
        for frame in load_synth(synth_dir):
            origin = stem_origin.get(frame.source_stem)
            if origin is not None:
                raise ValueError(
                    f"frame stem {frame.source_stem!r} exists in both {origin} and {synth_dir}"
                )
            stem_origin[frame.source_stem] = synth_dir
            frames.append(frame)
    return frames


def bake(
    synth_dirs: Path | Sequence[Path] = DEFAULT_SYNTH_DIR,
    out_dir: Path = DEFAULT_CORPUS_DIR,
) -> int:
    dirs = [synth_dirs] if isinstance(synth_dirs, Path) else list(synth_dirs)
    frames = _load_all(dirs)
    images_dir = out_dir / "frames"
    # a stale frame here is a frame from a different corpus, and nothing downstream can tell
    shutil.rmtree(images_dir, ignore_errors=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    corners: dict[str, list[list[float]]] = {}
    for frame in frames:
        image = cv2.imread(str(frame.image_path))
        if image is None:
            raise ValueError(f"cannot read frame {frame.image_path}")
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images_dir / f"{frame.source_stem}.png"), resized)
        normalised = frame.corners_px / np.array([float(width), float(height)])
        corners[frame.source_stem] = normalised.tolist()
    (out_dir / "corners.json").write_text(json.dumps(corners))
    return len(corners)


def main() -> None:
    parser = argparse.ArgumentParser(description="bake the render corpus for upload")
    parser.add_argument(
        "--synth-dir",
        dest="synth_dirs",
        type=Path,
        action="append",
        default=None,
        help="directory to bake frames from; repeatable, defaults to data/synth",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    args = parser.parse_args()
    synth_dirs = args.synth_dirs or [DEFAULT_SYNTH_DIR]
    count = bake(synth_dirs, args.out_dir)
    print(f"baked {count} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
