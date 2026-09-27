"""Compare ONNX keybed detectors on the synthetic pose grid and on real recordings.

Both paths run the same geometry the browser runtime does: quad_from_probability, then
constrain at the browser's fixed focal fraction of the frame width, with render geometry
for grid frames. Real recordings are tagged by their role in the current model's real-seg2
splits, so the summary can count only evidence the model never trained on.
"""

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import cv2
import numpy as np
import onnxruntime as ort

from pianocv.dataset import (
    DEFAULT_RECORDINGS_DIR,
    Recording,
    align,
    canonical_quad,
    parse_sidecar,
    scan_recordings,
)
from pianocv.gridtest import DEFAULT_GRID_DIR
from pianocv.jitter import constrain, quad_from_probability
from pianocv.pose import RENDER_DEPTH_UNITS, RENDER_WHITE_KEY_COUNT
from pianocv.segnet2 import preprocess_seg2

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS_PATH = _REPO_ROOT / "data" / "real-seg2" / "splits.json"
DEFAULT_KEYS_TRUTH_PATH = _REPO_ROOT / "data" / "recordings-keys-truth.json"
# the camera's focal as a fraction of the frame width, the same constant pianocv.jitter holds
_FOCAL_FRACTION = 0.75
_FAR_LIMIT_PX = 50.0

Role = Literal["train", "validation", "held_out", "unseen"]
_ROLES: tuple[Role, ...] = ("train", "validation", "held_out")
TruthSource = Literal["sidecar", "keys"]


class MaskSession(Protocol):
    def run(
        self, output_names: list[str] | None, _input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class FrameScore:
    iou: float
    near: float | None
    far: float | None
    quad: np.ndarray | None


@dataclass(frozen=True)
class Summary:
    n: int
    iou_median: float
    near_median: float | None
    far_median: float | None
    far_over_limit: int
    far_n: int


@dataclass(frozen=True)
class GridRow:
    elevation: float
    azimuth: float
    scores: dict[str, FrameScore]


@dataclass(frozen=True)
class ClipResult:
    stem: str
    kind: str
    role: Role
    frames: int
    scores: dict[str, list[FrameScore]]
    jitter: dict[str, np.ndarray]


def predict_mask(session: MaskSession, image_bgr: np.ndarray) -> np.ndarray:
    batch = preprocess_seg2(image_bgr)[None]
    output = session.run(None, {"image": batch})[0]
    return np.asarray(output[0, 0], dtype=np.float32)


def mask_iou(probability: np.ndarray, truth_px: np.ndarray, width: int, height: int) -> float:
    size = probability.shape[0]
    target = np.zeros((size, size), np.uint8)
    scaled = np.round(truth_px / np.array([width, height]) * (size - 1)).astype(np.int32)
    cv2.fillPoly(target, [scaled], 1)
    predicted = probability > 0.5
    truthy = target > 0
    union = int((predicted | truthy).sum())
    return float((predicted & truthy).sum()) / union if union else 0.0


def _thin_end(truth_px: np.ndarray) -> tuple[int, int]:
    """The far end's corner index and the near end's; the far end is the shorter edge."""
    edge_far = float(np.linalg.norm(truth_px[1] - truth_px[2]))
    edge_near = float(np.linalg.norm(truth_px[3] - truth_px[0]))
    return (1, 3) if edge_far < edge_near else (3, 1)


def score_frame(
    session: MaskSession, image_bgr: np.ndarray, truth_px: np.ndarray, render: bool
) -> FrameScore:
    height, width = image_bgr.shape[:2]
    probability = predict_mask(session, image_bgr)
    iou = mask_iou(probability, truth_px, width, height)
    quad = quad_from_probability(image_bgr, probability)
    visible = bool(np.all((truth_px >= 0) & (truth_px <= [width, height])))
    if quad is None or not visible:
        return FrameScore(iou=iou, near=None, far=None, quad=None)
    geometry: dict[str, float] = (
        {"white_keys": RENDER_WHITE_KEY_COUNT, "depth_units": RENDER_DEPTH_UNITS} if render else {}
    )
    fit, _ = constrain(image_bgr, probability, quad, _FOCAL_FRACTION * width, **geometry)
    aligned = align(canonical_quad(fit), truth_px)
    error = np.linalg.norm(aligned - truth_px, axis=1)
    far_index, near_index = _thin_end(truth_px)
    near = float((error[near_index] + error[(near_index + 1) % 4]) / 2)
    far = float((error[far_index] + error[(far_index + 1) % 4]) / 2)
    return FrameScore(iou=iou, near=near, far=far, quad=aligned)


def summarize(scores: list[FrameScore]) -> Summary:
    ious = [score.iou for score in scores]
    far = np.array([score.far for score in scores if score.far is not None])
    near = np.array([score.near for score in scores if score.near is not None])
    if far.size == 0:
        return Summary(len(scores), float(np.median(ious)), None, None, 0, 0)
    return Summary(
        len(scores),
        float(np.median(ious)),
        float(np.median(near)),
        float(np.median(far)),
        int((far > _FAR_LIMIT_PX).sum()),
        int(far.size),
    )


def format_summary(summary: Summary) -> str:
    if summary.far_n == 0:
        return f"IoU {summary.iou_median:.2f}   (no fully visible poses)"
    return (
        f"IoU {summary.iou_median:.2f}   near {summary.near_median:5.1f}px   "
        f"far {summary.far_median:5.1f}px   far>{_FAR_LIMIT_PX:.0f}px "
        f"{summary.far_over_limit}/{summary.far_n}"
    )


def jitter_steps(quads: list[np.ndarray]) -> np.ndarray:
    """Per-corner step in px between consecutive fitted quads, pooled over corners."""
    if len(quads) < 2:
        return np.empty(0)
    stacked = np.stack(quads)
    return np.asarray(np.linalg.norm(np.diff(stacked, axis=0), axis=2).ravel(), dtype=np.float64)


def format_jitter(steps: np.ndarray) -> str:
    if steps.size == 0:
        return "jitter      -"
    return f"jitter med {np.median(steps):5.2f}px p95 {np.percentile(steps, 95):5.2f}px"


def _label(path: Path) -> tuple[np.ndarray, float, float] | None:
    sidecar = json.loads(path.read_text())
    if "pose" not in sidecar:
        return None
    width, height = sidecar["imageWidth"], sidecar["imageHeight"]
    corners = np.array([[c["x"] * width, c["y"] * height] for c in sidecar["corners"]])
    pose = sidecar["pose"]
    return canonical_quad(corners), float(pose["elevation"]), float(pose["azimuth"])


def score_grid(sessions: Mapping[str, MaskSession], grid_dir: Path) -> list[GridRow]:
    rows: list[GridRow] = []
    for path in sorted(grid_dir.glob("*.json")):
        labelled = _label(path)
        if labelled is None:
            continue
        truth, elevation, azimuth = labelled
        image = cv2.imread(str(path.with_suffix(".png")))
        if image is None:
            continue
        scores = {
            name: score_frame(session, image, truth, render=True)
            for name, session in sessions.items()
        }
        rows.append(GridRow(elevation=elevation, azimuth=azimuth, scores=scores))
    return rows


def _print_group(label: str, rows: list[GridRow], model_names: list[str]) -> None:
    print(f"  {label}")
    for name in model_names:
        print(f"    {name}: {format_summary(summarize([row.scores[name] for row in rows]))}")


def print_grid_report(rows: list[GridRow], model_names: list[str]) -> None:
    print("SYNTHETIC GRID (render geometry, never trained on)")
    _print_group(f"all {len(rows)}", rows, model_names)
    for elevation in sorted({row.elevation for row in rows}):
        group = [row for row in rows if row.elevation == elevation]
        _print_group(f"elevation {elevation:>2.0f} deg", group, model_names)
    for azimuth in sorted({abs(row.azimuth) for row in rows}):
        group = [row for row in rows if abs(row.azimuth) == azimuth]
        _print_group(f"azimuth {azimuth:>2.0f} deg", group, model_names)


def load_splits(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {str(key): [str(name) for name in names] for key, names in data.items()}


def load_keys_truth(path: Path) -> dict[str, np.ndarray | None]:
    """Recording stem to its keys-only corners (normalized, uncanonicalised), or None when the
    relabel pipeline skipped that recording."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    truth: dict[str, np.ndarray | None] = {}
    for stem, entry in data.items():
        corners = entry.get("corners") if isinstance(entry, dict) else None
        truth[str(stem)] = (
            None
            if corners is None
            else np.array([[c["x"], c["y"]] for c in corners], dtype=np.float64)
        )
    return truth


def role_of(stem: str, kind: str, splits: dict[str, list[str]]) -> Role:
    def matches(name: str) -> bool:
        return name == f"{stem}.png" if kind == "snap" else name.startswith(f"{stem}.")

    for role in _ROLES:
        if any(matches(name) for name in splits.get(role, [])):
            return role
    return "unseen"


def read_recording_frames(recording: Recording, stride: int) -> list[np.ndarray] | None:
    if recording.kind == "snap":
        image = cv2.imread(str(recording.media_path))
        return None if image is None else [image]
    capture = cv2.VideoCapture(str(recording.media_path))
    if not capture.isOpened():
        capture.release()
        return None
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(image)
        index += 1
    capture.release()
    return frames if frames else None


def evaluate_recordings(
    sessions: Mapping[str, MaskSession],
    recordings_dir: Path,
    splits: dict[str, list[str]],
    stride: int,
    truth_source: TruthSource = "sidecar",
    keys_truth: dict[str, np.ndarray | None] | None = None,
) -> list[ClipResult]:
    results: list[ClipResult] = []
    for recording in scan_recordings(recordings_dir):
        frames = read_recording_frames(recording, stride)
        if frames is None:
            print(f"skipping unreadable recording {recording.stem}")
            continue
        sidecar = parse_sidecar(recording.sidecar_path, recording.kind)
        if truth_source == "keys":
            keys_corners = (keys_truth or {}).get(recording.stem)
            if keys_corners is None:
                print(f"skipping {recording.stem}: no keys-only truth")
                continue
            scale = np.array([float(sidecar.width), float(sidecar.height)])
            sidecar.corners = canonical_quad(keys_corners * scale) / scale
        truth = sidecar.corners * np.array([float(sidecar.width), float(sidecar.height)])
        scores: dict[str, list[FrameScore]] = {}
        jitter: dict[str, np.ndarray] = {}
        for name, session in sessions.items():
            frame_scores = [score_frame(session, frame, truth, render=False) for frame in frames]
            scores[name] = frame_scores
            jitter[name] = jitter_steps([s.quad for s in frame_scores if s.quad is not None])
        results.append(
            ClipResult(
                stem=recording.stem,
                kind=recording.kind,
                role=role_of(recording.stem, recording.kind, splits),
                frames=len(frames),
                scores=scores,
                jitter=jitter,
            )
        )
    return results


def _summary_columns(summary: Summary) -> str:
    near = f"{summary.near_median:7.1f}" if summary.near_median is not None else f"{'-':>7}"
    far = f"{summary.far_median:7.1f}" if summary.far_median is not None else f"{'-':>7}"
    far_over = f"{summary.far_over_limit}/{summary.far_n}" if summary.far_n else "-"
    return f"{summary.iou_median:>5.2f} {near} {far} {far_over:>8}"


def print_recording_table(results: list[ClipResult], model_names: list[str]) -> None:
    print(
        f"{'recording':<32} {'kind':<4} {'role':<10} {'frames':>6} {'model':<8} "
        f"{'iou':>5} {'near':>7} {'far':>7} {'far>50':>8}   jitter"
    )
    for result in results:
        for name in model_names:
            summary = summarize(result.scores[name])
            print(
                f"{result.stem:<32} {result.kind:<4} {result.role:<10} {result.frames:>6} "
                f"{name:<8} {_summary_columns(summary)}   {format_jitter(result.jitter[name])}"
            )


def print_evidence_summary(results: list[ClipResult], model_names: list[str]) -> None:
    evidence = [result for result in results if result.role in ("held_out", "unseen")]
    print(f"\nEVIDENCE (held_out + unseen only, {len(evidence)} recordings)")
    for name in model_names:
        scores = [score for result in evidence for score in result.scores[name]]
        steps = (
            np.concatenate([result.jitter[name] for result in evidence])
            if evidence
            else np.empty(0)
        )
        print(f"  {name}: {format_summary(summarize(scores))}   {format_jitter(steps)}")


def _parse_models(specs: list[str]) -> dict[str, ort.InferenceSession]:
    sessions: dict[str, ort.InferenceSession] = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator:
            raise ValueError(f"--model must be NAME=PATH, got {spec!r}")
        sessions[name] = ort.InferenceSession(path)
    return sessions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="compare onnx keybed detectors on the render grid and on real recordings"
    )
    parser.add_argument("--model", action="append", dest="models", metavar="NAME=PATH", default=[])
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--every", type=int, default=6)
    parser.add_argument(
        "--truth",
        choices=("sidecar", "keys"),
        default="sidecar",
        help="score against each recording's sidecar corners, or its keys-only truth",
    )
    args = parser.parse_args()
    if len(args.models) < 2:
        parser.error("pass at least two --model NAME=PATH")
    try:
        sessions = _parse_models(args.models)
    except ValueError as error:
        parser.error(str(error))
    model_names = list(sessions)

    print_grid_report(score_grid(sessions, args.grid), model_names)

    splits = load_splits(args.splits)
    keys_truth = load_keys_truth(DEFAULT_KEYS_TRUTH_PATH) if args.truth == "keys" else None
    results = evaluate_recordings(
        sessions, args.recordings, splits, args.every, args.truth, keys_truth
    )
    print("\nREAL RECORDINGS")
    print_recording_table(results, model_names)
    print_evidence_summary(results, model_names)


if __name__ == "__main__":
    main()
