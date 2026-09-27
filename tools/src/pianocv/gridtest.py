"""Score the detector on the deterministic pose grid, so a regression names its pose.

The grid comes from gen.html's "render test grid": one frame per (elevation, azimuth,
distance), rendered under a fixed seed, with the pose written into each sidecar.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from pianocv.dataset import align, canonical_quad
from pianocv.jitter import DEFAULT_SEG2_PATH, constrain, quad_from_probability
from pianocv.pose import RENDER_DEPTH_UNITS, RENDER_WHITE_KEY_COUNT
from pianocv.segnet2 import load_seg2, predict_mask2

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GRID_DIR = _REPO_ROOT / "data" / "grid"


def _error(quad: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    """Mean corner error, then the near and far end separately (far is the thinner end)."""
    aligned = align(quad, truth)
    per_corner = np.linalg.norm(aligned - truth, axis=1)
    near_first = np.linalg.norm(truth[3] - truth[0]) >= np.linalg.norm(truth[2] - truth[1])
    near = per_corner[[0, 3]] if near_first else per_corner[[1, 2]]
    far = per_corner[[1, 2]] if near_first else per_corner[[0, 3]]
    return float(per_corner.mean()), float(near.mean()), float(far.mean())


def score_grid(
    model_path: Path, grid_dir: Path, constrained: bool = False
) -> list[dict[str, float | str]]:
    model = load_seg2(model_path)
    rows: list[dict[str, float | str]] = []
    for sidecar_path in sorted(grid_dir.glob("*.json")):
        meta = json.loads(sidecar_path.read_text())
        image = cv2.imread(str(sidecar_path.with_suffix(".png")))
        if image is None or "pose" not in meta:
            continue
        height, width = image.shape[:2]
        truth = canonical_quad(
            np.array([[c["x"] * width, c["y"] * height] for c in meta["corners"]])
        )
        small = predict_mask2(model, image)
        quad = quad_from_probability(image, small)
        if quad is not None and constrained:
            # the grid is rendered as an 88-key keybed, whatever instrument the camera sees
            quad, _ = constrain(
                image,
                small,
                quad,
                white_keys=RENDER_WHITE_KEY_COUNT,
                depth_units=RENDER_DEPTH_UNITS,
            )
        pose = meta["pose"]
        # a corner outside the frame is extrapolated, not observed, and is scored apart
        inside = bool(
            np.all(
                (truth[:, 0] >= 0)
                & (truth[:, 0] < width)
                & (truth[:, 1] >= 0)
                & (truth[:, 1] < height)
            )
        )
        row: dict[str, float | str] = {
            "elevation": float(pose["elevation"]),
            "azimuth": float(pose["azimuth"]),
            "distance": float(pose["distance"]),
            "inside": 1.0 if inside else 0.0,
        }
        if quad is None:
            row.update(error=float("nan"), near=float("nan"), far=float("nan"), found=0.0)
        else:
            error, near, far = _error(quad, truth)
            row.update(error=error, near=near, far=far, found=1.0)
        rows.append(row)
    return rows


def _by(rows: list[dict[str, float | str]], key: str) -> None:
    groups: dict[float, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        groups[float(row[key])].append(row)
    print(f"\nby {key:9s} {'n':>3s} {'found':>5s} {'median':>7s} {'near':>6s} {'far':>6s}")
    for value in sorted(groups):
        group = groups[value]
        hit = [r for r in group if r["found"] == 1.0]
        if not hit:
            print(f"   {value:8.0f} {len(group):3d} {0:5d}")
            continue
        errors = np.array([float(r["error"]) for r in hit])
        near = np.array([float(r["near"]) for r in hit])
        far = np.array([float(r["far"]) for r in hit])
        print(
            f"   {value:8.0f} {len(group):3d} {len(hit):5d} {np.median(errors):7.1f} "
            f"{np.median(near):6.1f} {np.median(far):6.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="score the detector on the pose grid")
    parser.add_argument("--model", type=Path, default=DEFAULT_SEG2_PATH)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument(
        "--constrained", action="store_true", help="refit as the keybed rectangle in 3D"
    )
    args = parser.parse_args()
    rows = score_grid(args.model, args.grid_dir, args.constrained)
    for label, flag in (("all corners in frame", 1.0), ("corners off frame", 0.0)):
        subset = [r for r in rows if r["inside"] == flag]
        hit = [r for r in subset if r["found"] == 1.0]
        if not hit:
            continue
        errors = np.array([float(r["error"]) for r in hit])
        print(
            f"{label}: {len(subset)} poses, found {len(hit)}, median {np.median(errors):.1f} px, "
            f"under 10 px {int((errors < 10).sum())}, worst {errors.max():.1f}"
        )
    for key in ("elevation", "azimuth", "distance"):
        _by([r for r in rows if r["inside"] == 1.0], key)


if __name__ == "__main__":
    main()
