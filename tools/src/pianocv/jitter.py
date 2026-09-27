"""Measure how much the detected quad moves on a clip where nothing moves.

A static recording carries one label for every frame, so any frame-to-frame motion in the
detection is noise the viewer sees as bouncing edges. This reports that noise per corner.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from pianocv.dataset import (
    DEFAULT_RECORDINGS_DIR,
    align,
    canonical_quad,
    orient_quad,
    parse_sidecar,
    scan_recordings,
)
from pianocv.fitquad import quad_from_mask
from pianocv.pose import DEPTH_UNITS, WHITE_KEY_COUNT
from pianocv.rectfit import _segment_distances, boundary_points, fit_rectangle, snap_to_gradient
from pianocv.refine_edges import refine_quad
from pianocv.segnet2 import KeybedSegNet2, load_seg2, predict_mask2

# the rectangle may explain the boundary this much worse than the free quad and still show:
# a smaller margin made the two trade places frame to frame, since both snap to the same gradients
_FIT_SLACK_PX = 2.0
# the camera's focal as a fraction of the frame width, a webcam's lens; the same constant
# the browser holds (web/src/detector.ts), since per-frame estimates wander
_CAMERA_FOCAL = 0.75
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEG2_PATH = _REPO_ROOT / "data" / "models" / "keybed_seg2.tuned.pt"
_MASK_THRESHOLD = 0.5
_JUMP_PX = 10.0
_FRAME_S = 1.0 / 30.0
# One-Euro (Casiez 2012): cutoff rises with speed, so still is smooth and motion is not laggy
_MIN_CUTOFF_HZ = 1.0
_BETA = 10.0
_DERIVATIVE_CUTOFF_HZ = 1.0
# frames of mask probability averaged before fitting when the scene is static
_ACCUMULATE = 8
# the same shape test the app applies before it will draw a quad (web/src/quad.ts)
_MIN_ASPECT = 1.8
_MAX_ASPECT = 30.0
# one end may be drawn this much wider than the other, the same as their distance ratio; the
# hand-labelled views span 1.54 to 2.65
_MAX_END_RATIO = 3.2
Filter = Literal["raw", "oneeuro", "accumulate", "both"]


def keybed_shaped(quad: np.ndarray) -> bool:
    edges = [float(np.linalg.norm(quad[(i + 1) % 4] - quad[i])) for i in range(4)]
    span = (edges[0] + edges[2]) / 2
    depth = max((edges[1] + edges[3]) / 2, 1e-6)
    signs = set()
    for i in range(4):
        a = quad[(i + 1) % 4] - quad[i]
        b = quad[(i + 2) % 4] - quad[(i + 1) % 4]
        signs.add(float(np.sign(a[0] * b[1] - a[1] * b[0])))
    ends = (edges[1], edges[3])
    ratio = max(ends) / max(min(ends), 1e-6)
    return (
        len(signs) == 1 and _MIN_ASPECT <= span / depth <= _MAX_ASPECT and ratio <= _MAX_END_RATIO
    )


@dataclass(frozen=True)
class ClipJitter:
    stem: str
    frames: int
    found: int
    accepted: int
    corner_std_px: np.ndarray
    step_median_px: float
    step_p95_px: float
    jumps: int
    error_px: float | None


def _alpha(cutoff_hz: float) -> float:
    return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff_hz * _FRAME_S))


class OneEuro:
    def __init__(self) -> None:
        self.value: np.ndarray | None = None
        self.rate = np.zeros((4, 2))

    def push(self, quad: np.ndarray) -> np.ndarray:
        if self.value is None:
            self.value = quad.copy()
            return quad
        rate = (quad - self.value) / _FRAME_S
        a_rate = _alpha(_DERIVATIVE_CUTOFF_HZ)
        self.rate = a_rate * rate + (1.0 - a_rate) * self.rate
        cutoff = _MIN_CUTOFF_HZ + _BETA * np.abs(self.rate)
        a = 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * _FRAME_S))
        self.value = a * quad + (1.0 - a) * self.value
        return np.asarray(self.value, dtype=np.float64)


def quad_from_probability(image_bgr: np.ndarray, small: np.ndarray) -> np.ndarray | None:
    height, width = image_bgr.shape[:2]
    quad = quad_from_mask((small > _MASK_THRESHOLD).astype(np.uint8) * 255)
    if quad is None:
        return None
    scaled = quad / float(small.shape[0] - 1) * np.array([float(width), float(height)])
    return orient_quad(image_bgr, refine_quad(image_bgr, scaled))


def constrain(
    image_bgr: np.ndarray,
    small: np.ndarray,
    quad: np.ndarray,
    focal: float | None = None,
    white_keys: float = WHITE_KEY_COUNT,
    depth_units: float = DEPTH_UNITS,
) -> tuple[np.ndarray, float | None]:
    """The quad refitted as the keybed rectangle against the whole mask boundary, and the
    focal the fit settled on, or None when the boundary gave nothing to fit."""
    height, width = image_bgr.shape[:2]
    points = boundary_points(small, width, height)
    if points is None:
        return quad, None
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    points = snap_to_gradient(gray, points, quad)
    fit = fit_rectangle(points, quad, width, height, focal, white_keys, depth_units, gray)
    if not np.isfinite(fit.cost):
        return quad, None
    # the rectangle earns its place by explaining the boundary at least as well as the free
    # quad; at a wrong focal it cannot, and the free quad is the better picture
    plain_residual = float(np.median(_segment_distances(points, canonical_quad(quad))[0]))
    fit_residual = float(np.median(_segment_distances(points, fit.quad_px)[0]))
    if fit_residual > plain_residual + _FIT_SLACK_PX:
        return quad, fit.focal
    return orient_quad(image_bgr, fit.quad_px), fit.focal


def detect_quad(model: KeybedSegNet2, image_bgr: np.ndarray) -> np.ndarray | None:
    return quad_from_probability(image_bgr, predict_mask2(model, image_bgr))


def _align(quads: list[np.ndarray]) -> np.ndarray:
    # a change of corner order between frames is a labelling miss, and would otherwise register
    # as the keybed's whole length of motion; every frame is put in the first frame's order
    return np.stack([quads[0]] + [align(quad, quads[0]) for quad in quads[1:]])


def measure_clip(
    model: KeybedSegNet2,
    media_path: Path,
    label_px: np.ndarray | None,
    mode: Filter = "raw",
    dump_dir: Path | None = None,
    constrained: bool = False,
) -> ClipJitter | None:
    capture = cv2.VideoCapture(str(media_path))
    quads: list[np.ndarray] = []
    frames = 0
    found = 0
    rejected: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]] = []
    recent: list[np.ndarray] = []
    smoother = OneEuro()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames += 1
        small = predict_mask2(model, frame)
        if mode in ("accumulate", "both"):
            recent.append(small)
            del recent[:-_ACCUMULATE]
            small = np.mean(recent, axis=0)
        quad = quad_from_probability(frame, small)
        if quad is None:
            continue
        if constrained:
            quad, _ = constrain(frame, small, quad, _CAMERA_FOCAL * frame.shape[1])
        found += 1
        if not keybed_shaped(quad):
            rejected.append((frames, frame, small, quad))
            continue
        if mode in ("oneeuro", "both"):
            if quads:
                flipped = np.roll(quad, 2, axis=0)
                if np.linalg.norm(flipped - quads[-1]) < np.linalg.norm(quad - quads[-1]):
                    quad = flipped
            quad = smoother.push(quad)
        quads.append(quad)
    capture.release()
    if dump_dir is not None and rejected:
        _dump(dump_dir, media_path.stem, rejected[:6])
    if len(quads) < 2:
        return None
    run = _align(quads)
    steps = np.linalg.norm(np.diff(run, axis=0), axis=2).mean(axis=1)
    error = None
    if label_px is not None:
        error = float(
            np.mean([np.linalg.norm(align(q, label_px) - label_px, axis=1).mean() for q in run])
        )
    return ClipJitter(
        stem=f"{media_path.stem} [{mode}{' +constrained' if constrained else ''}]",
        frames=frames,
        found=found,
        accepted=len(quads),
        corner_std_px=run.std(axis=0).mean(axis=1),
        step_median_px=float(np.median(steps)),
        step_p95_px=float(np.percentile(steps, 95)),
        jumps=int((steps > _JUMP_PX).sum()),
        error_px=error,
    )


def _dump(
    dump_dir: Path, stem: str, worst: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for index, frame, small, quad in worst:
        view = frame.copy()
        height, width = view.shape[:2]
        heat = cv2.resize((small * 255).astype(np.uint8), (width, height))
        view[..., 2] = np.maximum(view[..., 2], heat)
        cv2.polylines(view, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.putText(
            view, f"frame {index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        tiles.append(cv2.resize(view, (320, 240)))
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(dump_dir / f"{stem}-rejected.png"), np.vstack(rows))


def report(results: list[ClipJitter]) -> None:
    print(
        f"{'clip':36s} {'frames':>6s} {'found':>5s} {'kept':>5s} {'std/corner px':>28s} "
        f"{'step med':>8s} {'step p95':>8s} {'jumps':>5s} {'error':>6s}"
    )
    for r in results:
        std = " ".join(f"{v:5.1f}" for v in r.corner_std_px)
        error = f"{r.error_px:6.1f}" if r.error_px is not None else "     -"
        print(
            f"{r.stem:36s} {r.frames:6d} {r.found:5d} {r.accepted:5d} {std:>28s} "
            f"{r.step_median_px:8.2f} {r.step_p95_px:8.2f} {r.jumps:5d} {error}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="measure detection jitter on static clips")
    parser.add_argument("--model", type=Path, default=DEFAULT_SEG2_PATH)
    parser.add_argument("--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument(
        "--filter", action="append", choices=("raw", "oneeuro", "accumulate", "both")
    )
    parser.add_argument("--dump-dir", type=Path, default=None)
    parser.add_argument(
        "--constrained", action="store_true", help="refit as the keybed rectangle in 3D"
    )
    args = parser.parse_args()
    modes: list[Filter] = args.filter or ["raw"]
    model = load_seg2(args.model)
    results: list[ClipJitter] = []
    for recording in scan_recordings(args.recordings_dir):
        if recording.media_path.suffix != ".webm":
            continue
        sidecar = parse_sidecar(recording.sidecar_path, recording.kind)
        label = None
        if sidecar.corners is not None:
            label = sidecar.corners * np.array([float(sidecar.width), float(sidecar.height)])
        for mode in modes:
            measured = measure_clip(
                model,
                recording.media_path,
                label,
                mode,
                args.dump_dir,
                args.constrained,
            )
            if measured is not None:
                results.append(measured)
    report(results)


if __name__ == "__main__":
    main()
