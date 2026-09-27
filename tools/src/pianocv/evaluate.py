"""Evaluate the keybed detector against sidecar ground truth."""

import argparse
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np

from pianocv.dataset import Frame, load_frames, orient_quad
from pianocv.detect import Detection, find_keybed, find_keybed_pattern
from pianocv.fitquad import quad_from_mask
from pianocv.jitter import DEFAULT_SEG2_PATH, constrain
from pianocv.model import (
    KeybedNet,
    KeybedSegNet,
    load_model,
    load_seg_model,
    predict_corners,
    predict_mask,
)
from pianocv.refine_edges import refine_quad
from pianocv.segnet2 import KeybedSegNet2, load_seg2, predict_mask2

Method = Literal["v0", "pattern", "net", "seg", "seg2"]
Detector = Callable[[np.ndarray], Detection | None]
_DETECTORS: dict[str, Detector] = {"v0": find_keybed, "pattern": find_keybed_pattern}

_PRESENT_THRESHOLD = 0.5
_SUCCESS_RADIUS_PX = 15.0
_DUPLICATE_QUAD_PX = 1.0
_FINE_TUNE_KIND = "rec"
_MASK_THRESHOLD = 0.5
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FRAMES_DIR = _REPO_ROOT / "data" / "frames"
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "out" / "detect"
DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_net.pt"
_GREEN = (0, 255, 0)
_BLUE = (255, 0, 0)
_RED = (0, 0, 255)


def default_method() -> Method:
    return "net" if DEFAULT_MODEL_PATH.is_file() else "pattern"


def net_detector(model: KeybedNet) -> Detector:
    def detect(image_bgr: np.ndarray) -> Detection | None:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        quad_px, present = predict_corners(model, image_rgb)
        if present < _PRESENT_THRESHOLD:
            return None
        return Detection(quad_px=quad_px, confidence=present)

    return detect


def seg_detector(model: KeybedSegNet, threshold: float = _MASK_THRESHOLD) -> Detector:
    def detect(image_bgr: np.ndarray) -> Detection | None:
        probability = predict_mask(model, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        quad = quad_from_mask((probability > threshold).astype(np.uint8) * 255)
        if quad is None:
            return None
        refined = refine_quad(image_bgr, quad)
        return Detection(
            quad_px=orient_quad(image_bgr, refined), confidence=float(probability.max())
        )

    return detect


def seg2_detector(
    model: KeybedSegNet2, threshold: float = _MASK_THRESHOLD, constrained: bool = False
) -> Detector:
    def detect(image_bgr: np.ndarray) -> Detection | None:
        height, width = image_bgr.shape[:2]
        small = predict_mask2(model, image_bgr)
        quad = quad_from_mask((small > threshold).astype(np.uint8) * 255)
        if quad is None:
            return None
        scaled = quad / float(small.shape[0] - 1) * np.array([float(width), float(height)])
        refined = orient_quad(image_bgr, refine_quad(image_bgr, scaled))
        if constrained:
            refined, _ = constrain(image_bgr, small, refined)
        return Detection(quad_px=refined, confidence=float(small.max()))

    return detect


def _detector_for(
    method: Method, model_path: Path = DEFAULT_MODEL_PATH, constrained: bool = False
) -> Detector:
    if method == "seg2":
        seg2_path = DEFAULT_SEG2_PATH if model_path == DEFAULT_MODEL_PATH else model_path
        return seg2_detector(load_seg2(seg2_path), constrained=constrained)
    if method == "net":
        return net_detector(load_model(model_path))
    if method == "seg":
        return seg_detector(load_seg_model(model_path))
    return _DETECTORS[method]


@dataclass
class FrameResult:
    source_stem: str
    kind: str
    quad_px: np.ndarray | None
    locked: bool
    success: bool
    mean_error_px: float | None
    trained: bool = False


def evaluate_frame(
    frame: Frame,
    method: Method = "pattern",
    detector: Detector | None = None,
    trained: bool = False,
) -> FrameResult:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise ValueError(f"cannot read frame {frame.image_path}")
    if detector is None:
        detector = _detector_for(method)
    detection = detector(image)
    if detection is None:
        return FrameResult(
            source_stem=frame.source_stem,
            kind=frame.kind,
            quad_px=None,
            locked=False,
            success=False,
            mean_error_px=None,
            trained=trained,
        )
    quad_px = detection.quad_px
    if frame.corners_px is None:
        return FrameResult(
            source_stem=frame.source_stem,
            kind=frame.kind,
            quad_px=quad_px,
            locked=True,
            success=False,
            mean_error_px=None,
            trained=trained,
        )
    errors = np.linalg.norm(quad_px - frame.corners_px, axis=1)
    return FrameResult(
        source_stem=frame.source_stem,
        kind=frame.kind,
        quad_px=quad_px,
        locked=True,
        success=bool(float(errors.max()) <= _SUCCESS_RADIUS_PX),
        mean_error_px=float(errors.mean()),
        trained=trained,
    )


def fine_tuned_frames(frames: list[Frame]) -> list[bool]:
    # a snapshot that reuses a fine-tuned clip's corners is training data whatever its kind says
    quads = [
        frame.corners_px
        for frame in frames
        if frame.kind == _FINE_TUNE_KIND and frame.corners_px is not None
    ]
    return [
        frame.corners_px is not None
        and any(
            float(np.abs(frame.corners_px - quad).max()) <= _DUPLICATE_QUAD_PX for quad in quads
        )
        for frame in frames
    ]


def run(
    frames_dir: Path,
    out_dir: Path,
    method: Method = "pattern",
    model_path: Path = DEFAULT_MODEL_PATH,
    constrained: bool = False,
    source_stems: set[str] | None = None,
) -> list[FrameResult]:
    frames = load_frames(frames_dir)
    if source_stems is not None:
        frames = [frame for frame in frames if frame.source_stem in source_stems]
    out_dir.mkdir(parents=True, exist_ok=True)
    detector = _detector_for(method, model_path, constrained)
    trained = fine_tuned_frames(frames)
    results = [
        evaluate_frame(frame, method, detector, flag)
        for frame, flag in zip(frames, trained, strict=True)
    ]
    for frame, result in zip(frames, results, strict=True):
        _write_preview(frame, result, out_dir)
    _print_table(results)
    return results


@dataclass
class _GroupStats:
    label: str
    kind: str
    frames: int = 0
    locked: int = 0
    successes: int = 0
    errors: list[float] = field(default_factory=list)

    def add(self, result: FrameResult) -> None:
        self.frames += 1
        if result.locked:
            self.locked += 1
            if result.mean_error_px is not None:
                self.errors.append(result.mean_error_px)
        if result.success:
            self.successes += 1

    def merge(self, other: "_GroupStats") -> None:
        self.frames += other.frames
        self.locked += other.locked
        self.successes += other.successes
        self.errors += other.errors

    @property
    def lock_rate(self) -> float:
        return self.locked / self.frames if self.frames else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.frames if self.frames else 0.0

    @property
    def mean_error(self) -> float | None:
        return sum(self.errors) / len(self.errors) if self.errors else None

    @property
    def worst_error(self) -> float | None:
        return max(self.errors) if self.errors else None


def _print_table(results: list[FrameResult]) -> None:
    groups: dict[tuple[str, str], _GroupStats] = {}
    for result in results:
        groups.setdefault(
            (result.source_stem, result.kind),
            _GroupStats(result.source_stem, result.kind),
        ).add(result)
    by_kind: dict[str, _GroupStats] = {}
    for stats in groups.values():
        by_kind.setdefault(stats.kind, _GroupStats(stats.kind, stats.kind)).merge(stats)
    splits = {"fine-tuned on": _GroupStats("", ""), "held out": _GroupStats("", "")}
    for result in results:
        splits["fine-tuned on" if result.trained else "held out"].add(result)
    print(
        f"{'source':<28} {'kind':<7} {'frames':>6} {'lock':>6} {'success':>8} "
        f"{'mean_err_px':>12} {'worst_err_px':>13}"
    )
    for stats in groups.values():
        print(_format_row(stats.label, stats.kind, stats))
    for kind in sorted(by_kind):
        print(_format_row(f"{kind} (all)", kind, by_kind[kind]))
    for label, stats in splits.items():
        print(_format_row(label, "", stats))


def _format_row(label: str, kind: str, stats: _GroupStats) -> str:
    mean_error = "-" if stats.mean_error is None else f"{stats.mean_error:.1f}"
    worst_error = "-" if stats.worst_error is None else f"{stats.worst_error:.1f}"
    return (
        f"{label:<28} {kind:<7} {stats.frames:>6} {stats.lock_rate:>6.2f} "
        f"{stats.success_rate:>8.2f} {mean_error:>12} {worst_error:>13}"
    )


def _write_preview(frame: Frame, result: FrameResult, out_dir: Path) -> None:
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise ValueError(f"cannot read frame {frame.image_path}")
    if frame.corners_px is not None:
        color = _GREEN if result.locked else _RED
        cv2.polylines(image, [frame.corners_px.astype(np.int32)], True, color, 2)
    if result.quad_px is not None:
        cv2.polylines(image, [result.quad_px.astype(np.int32)], True, _BLUE, 2)
    cv2.imwrite(str(out_dir / frame.image_path.name), image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="evaluate keybed detection against sidecar ground truth"
    )
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--method", choices=("v0", "pattern", "net", "seg", "seg2"), default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--constrained", action="store_true", help="refit as the keybed rectangle in 3D"
    )
    parser.add_argument(
        "--source-stem",
        action="append",
        dest="source_stems",
        help="evaluate only frames from this source; repeat for multiple sources",
    )
    args = parser.parse_args()
    method = cast(Method, args.method) if args.method is not None else default_method()
    run(
        args.frames_dir,
        args.out_dir,
        method,
        args.model,
        args.constrained,
        set(args.source_stems) if args.source_stems is not None else None,
    )


if __name__ == "__main__":
    main()
