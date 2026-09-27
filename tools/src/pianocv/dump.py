"""Render procedural samples to disk so a GPU host can train on them without the renderer.

The renderer is what carries transfer to real photographs, but it is CPU bound, which made it
the reason training could not move off this machine. Rendering once to files decouples the two.
"""

import argparse
import json
from multiprocessing.pool import Pool
from pathlib import Path

import cv2
import numpy as np

from pianocv.render import render_sample
from pianocv.segnet2 import resize_rgb

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DUMP_DIR = _REPO_ROOT / "data" / "procedural"
_WIDTH = 640
_HEIGHT = 480


def _one(task: tuple[int, int, str]) -> None:
    index, seed, out = task
    sample = render_sample(np.random.default_rng(seed + index))
    directory = Path(out)
    stem = f"proc-{index:06d}"
    cv2.imwrite(str(directory / f"{stem}.png"), resize_rgb(sample.image, _SIZE)[:, :, ::-1])
    corners = sample.quad_px / np.array([float(_WIDTH), float(_HEIGHT)])
    (directory / f"{stem}.json").write_text(
        json.dumps(
            {
                "kind": "synth",
                "startedAt": 0,
                "durationMs": 0,
                "corners": [{"x": float(x), "y": float(y)} for x, y in corners],
                "imageWidth": _SIZE,
                "imageHeight": _SIZE,
                "mimeType": "image/png",
            }
        )
    )


_SIZE = 288


def dump(count: int, out_dir: Path = DEFAULT_DUMP_DIR, seed: int = 0, workers: int = 4) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(i, seed, str(out_dir)) for i in range(count)]
    with Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(_one, tasks, chunksize=32):
            pass
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="render procedural frames to disk")
    parser.add_argument("--count", type=int, default=30_000)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DUMP_DIR)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    written = dump(args.count, args.out_dir, workers=args.workers)
    print(f"rendered {written} frames to {args.out_dir}")


if __name__ == "__main__":
    main()
