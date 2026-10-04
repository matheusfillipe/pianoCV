"""Score a KeyNet export on labelled frames: how far each predicted keypoint lands from its label,
in white-key widths, so models and decoders compare on the same ruler.

Each frame is cropped the way the runtime tracks, around its labelled keyboard, so the score is
the detector's own precision and leaves acquisition and the fit out.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from pianocv.keymatch import SynthKeysFrame, load_synth_keys
from pianocv.keynet import (
    BACK_CHANNELS,
    BLACK_TOP_CHANNELS,
    IGNORE_THRESHOLD,
    STRIDE,
    TRACK_HEIGHT,
    TRACK_MARGIN_ACROSS,
    TRACK_MARGIN_ALONG,
    TRACK_WIDTH,
    Crop,
    KeyPoints,
    frame_points,
    oriented_crop,
    rectified_crop,
    shown_corners,
)
from pianocv.keyseg import preprocess_crop

PEAK_THRESHOLD = 0.3
# a peak farther than this from its label is a miss, not a large error: half a key is where a
# drawn key starts landing on its neighbour
MATCH_KEYS = 0.5
PCK_KEYS = (0.1, 0.25)

GROUPS: dict[str, tuple[int, ...]] = {
    "corners": (0, 1, 2, 3),
    "gaps": (4,),
    "black": (5, 6),
    "tops": BLACK_TOP_CHANNELS,
    "back": BACK_CHANNELS[:1],
    "backtops": BACK_CHANNELS[1:],
}
# a frame fails when a keybed corner is missed or half a key off, since a wrong end numbers
# every key wrong; a single missed gap or black corner leaves the fit standing
FAIL_GROUPS = ("corners",)

# a decoder places a peak inside its cell from the heatmap around it and, for models that have
# them, the predicted offsets at that cell
Decoder = Callable[[np.ndarray, np.ndarray | None, int, int], tuple[float, float]]
# heatmaps, and offsets for models with an offset head
Predictor = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray | None]]


def centroid(
    channel: np.ndarray, _offset: np.ndarray | None, i: int, j: int
) -> tuple[float, float]:
    """The runtime's 3x3 weighted centroid around a peak, in cells."""
    window = np.clip(_window(channel, i, j), 0.0, None)
    total = float(window.sum())
    if total <= 0:
        return float(i), float(j)
    offsets = np.arange(-1, 2)
    return (
        i + float((window.sum(axis=1) * offsets).sum()) / total,
        j + float((window.sum(axis=0) * offsets).sum()) / total,
    )


def parabola(
    channel: np.ndarray, _offset: np.ndarray | None, i: int, j: int
) -> tuple[float, float]:
    """A separable 3-point parabola through the log of the peak and its neighbours, in cells,
    which lands exactly on a Gaussian's centre."""
    window = np.log(np.clip(_window(channel, i, j), 1e-6, None))
    return i + _vertex(window[:, 1]), j + _vertex(window[1, :])


def _vertex(values: np.ndarray) -> float:
    before, peak, after = (float(v) for v in values)
    curvature = before - 2 * peak + after
    if curvature >= 0:
        return 0.0
    return float(np.clip(0.5 * (before - after) / curvature, -0.5, 0.5))


def _window(channel: np.ndarray, i: int, j: int) -> np.ndarray:
    padded = np.pad(channel, 1, mode="edge")
    return np.asarray(padded[i : i + 3, j : j + 3], dtype=np.float64)


def predicted(
    channel: np.ndarray, offset: np.ndarray | None, i: int, j: int
) -> tuple[float, float]:
    """The peak cell moved by the offset the model predicts there, x then y in cells; the
    centroid for a model without an offset head."""
    if offset is None:
        return centroid(channel, None, i, j)
    return i + float(offset[1, i, j]), j + float(offset[0, i, j])


DECODERS: dict[str, Decoder] = {"centroid": centroid, "parabola": parabola, "offset": predicted}


def peaks(
    channel: np.ndarray,
    decoder: Decoder,
    threshold: float = PEAK_THRESHOLD,
    offset: np.ndarray | None = None,
) -> np.ndarray:
    """Each local maximum above the threshold, refined, as (x, y) crop pixels."""
    grown = cv2.dilate(channel, np.ones((3, 3), np.uint8))
    rows, cols = np.nonzero((channel >= grown) & (channel >= threshold))
    found = []
    for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
        ci, cj = decoder(channel, offset, i, j)
        found.append((STRIDE * cj + 0.5, STRIDE * ci + 0.5))
    return np.array(found, dtype=np.float64).reshape(-1, 2)


@dataclass(frozen=True)
class GroupScore:
    labels: int
    recall: float
    median_keys: float
    mean_keys: float
    pck: dict[str, float]


@dataclass(frozen=True)
class Score:
    frames: int
    failure_rate: float
    groups: dict[str, GroupScore]


def _key_width(points: KeyPoints, crop: Crop) -> float:
    """One white key's width in crop pixels, the front edge as the crop shows it over the key
    count, which a rectified crop keeps equal along the whole edge."""
    front_high, front_low = crop.points(points.corners[2:4])
    front = float(np.linalg.norm(front_high - front_low))
    return front / (len(points.gaps) + 1)


def _ignored(frame: SynthKeysFrame) -> np.ndarray | None:
    if frame.ignore_mask is None:
        return None
    return cv2.imread(str(frame.ignore_mask), cv2.IMREAD_GRAYSCALE)


def _labels(points: KeyPoints, channel: int) -> np.ndarray:
    by_channel = [
        *shown_corners(points),
        points.gaps,
        points.black_low,
        points.black_high,
        points.black_top_low,
        points.black_top_high,
        points.back_gaps,
        points.black_back_low,
        points.black_back_high,
    ]
    return by_channel[channel][np.isfinite(by_channel[channel]).all(axis=1)]


def score(
    frames: list[SynthKeysFrame],
    predict: Predictor,
    decoder: Decoder,
    threshold: float = PEAK_THRESHOLD,
    fixed_dir: Path | None = None,
    rectified: bool = False,
) -> Score:
    """Errors in white-key widths for every labelled keypoint the crop shows and no hand covers,
    matched to the nearest peak of its own channel. A frame with a corrected label file is scored
    against it; one whose keybed corner is hidden has no crop and is left out."""
    errors: dict[str, list[float]] = {name: [] for name in GROUPS}
    labelled = dict.fromkeys(GROUPS, 0)
    failed = 0
    counted = 0
    for frame in frames:
        points = frame_points(frame, fixed_dir)
        image = cv2.imread(str(frame.image_path))
        if points is None or image is None or not np.isfinite(points.corners).all():
            continue
        counted += 1
        make_crop = rectified_crop if rectified else oriented_crop
        crop = make_crop(
            points.corners, TRACK_WIDTH, TRACK_HEIGHT, TRACK_MARGIN_ALONG, TRACK_MARGIN_ACROSS
        )
        pixels = cv2.cvtColor(
            cv2.warpPerspective(image, crop.to_crop, (TRACK_WIDTH, TRACK_HEIGHT)), cv2.COLOR_BGR2RGB
        )
        heat_batch, offset_batch = predict(preprocess_crop(pixels)[None])
        heat = heat_batch[0]
        offsets = None if offset_batch is None else offset_batch[0]
        key_crop_px = _key_width(points, crop)
        ignore = _ignored(frame)
        frame_failed = False
        for name, channels in GROUPS.items():
            for channel in channels:
                if channel >= heat.shape[0]:
                    continue
                found = peaks(
                    heat[channel],
                    decoder,
                    threshold,
                    None if offsets is None else offsets[2 * channel : 2 * channel + 2],
                )
                for label in _labels(points, channel):
                    # a keypoint past the frame's edge is one the camera never saw
                    if not (0 <= label[0] < image.shape[1] and 0 <= label[1] < image.shape[0]):
                        continue
                    if ignore is not None:
                        row, col = round(float(label[1])), round(float(label[0]))
                        inside = 0 <= row < ignore.shape[0] and 0 <= col < ignore.shape[1]
                        if inside and ignore[row, col] > IGNORE_THRESHOLD:
                            continue
                    at = crop.points(label[None])[0]
                    if not (0 <= at[0] < TRACK_WIDTH and 0 <= at[1] < TRACK_HEIGHT):
                        continue
                    labelled[name] += 1
                    distance = (
                        float(np.linalg.norm(found - at, axis=1).min()) / key_crop_px
                        if len(found)
                        else np.inf
                    )
                    if distance <= MATCH_KEYS:
                        errors[name].append(distance)
                    elif name in FAIL_GROUPS:
                        frame_failed = True
        failed += int(frame_failed)
    return Score(
        frames=counted,
        failure_rate=failed / counted if counted else 0.0,
        groups={name: _group(errors[name], labelled[name]) for name in GROUPS},
    )


# a white key is 23.5 mm wide and the keybed 150 mm deep
_KEY_WIDTH_MM = 23.5
_KEYBED_DEPTH_MM = 150.0
_ELEVATION_EDGES_DEG = (30.0, 60.0)
_ELEVATION_NAMES = ("low", "mid", "high")


def camera_elevation(points: KeyPoints) -> float:
    """The camera's elevation over the keybed in degrees, from how much the quad's depth is
    foreshortened against its width: 0 looks along the keys, 90 looks straight down."""
    corners = points.corners
    width = (np.linalg.norm(corners[1] - corners[0]) + np.linalg.norm(corners[2] - corners[3])) / 2
    depth = (np.linalg.norm(corners[3] - corners[0]) + np.linalg.norm(corners[2] - corners[1])) / 2
    true_ratio = _KEYBED_DEPTH_MM / (_KEY_WIDTH_MM * (len(points.gaps) + 1))
    return float(np.degrees(np.arcsin(np.clip(depth / width / true_ratio, 0.0, 1.0))))


def view_of(points: KeyPoints) -> tuple[str, str]:
    bucket = _ELEVATION_NAMES[int(np.searchsorted(_ELEVATION_EDGES_DEG, camera_elevation(points)))]
    return f"{len(points.gaps) + 1} white keys", f"elevation {bucket}"


def score_by_view(
    frames: list[SynthKeysFrame],
    predict: Predictor,
    decoder: Decoder,
    threshold: float = PEAK_THRESHOLD,
    fixed_dir: Path | None = None,
    rectified: bool = False,
) -> dict[str, Score]:
    """The overall score, then one per board size and per camera elevation bucket."""
    views: dict[str, list[SynthKeysFrame]] = {}
    for frame in frames:
        points = frame_points(frame, fixed_dir)
        if points is None or not np.isfinite(points.corners).all():
            continue
        for view in view_of(points):
            views.setdefault(view, []).append(frame)
    return {
        name: score(subset, predict, decoder, threshold, fixed_dir, rectified)
        for name, subset in {"all": frames, **dict(sorted(views.items()))}.items()
    }


def _group(errors: list[float], labelled: int) -> GroupScore:
    values = np.array(errors) if errors else np.array([np.nan])
    return GroupScore(
        labels=labelled,
        recall=len(errors) / labelled if labelled else 0.0,
        median_keys=float(np.nanmedian(values)),
        mean_keys=float(np.nanmean(values)),
        pck={f"{k}": float(np.mean(values <= k)) if errors else 0.0 for k in PCK_KEYS},
    )


def _json_points(points: np.ndarray) -> list[list[float] | None]:
    return [None if np.isnan(row).any() else row.tolist() for row in points]


def label_points(
    frames: list[SynthKeysFrame], fixed_dir: Path | None = None
) -> dict[str, dict[str, object]]:
    """Each frame's labelled keypoints in frame pixels, low key to high, for the browser to score
    the runtime's fitted board against; a hidden point is null."""
    exported: dict[str, dict[str, object]] = {}
    for frame in frames:
        points = frame_points(frame, fixed_dir)
        if points is None:
            continue
        exported[frame.image_path.stem] = {
            "width": frame.image_size[0],
            "height": frame.image_size[1],
            "whiteKeys": len(points.gaps) + 1,
            "corners": _json_points(points.corners),
            "gaps": _json_points(points.gaps),
            "blackLow": _json_points(points.black_low),
            "blackHigh": _json_points(points.black_high),
        }
    return exported


def onnx_predictor(model_path: Path) -> Predictor:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    names = [output.name for output in session.get_outputs()]

    def predict(batch: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        outputs = dict(
            zip(names, session.run(None, {"image": batch.astype(np.float32)}), strict=True)
        )
        return outputs["heatmaps"], outputs.get("offsets")

    return predict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="score a KeyNet export on labelled frames")
    parser.add_argument("--model", type=Path, action="append", default=[])
    parser.add_argument(
        "--points-out", type=Path, help="write the frames' labelled keypoints here as JSON"
    )
    parser.add_argument("--frames", type=Path, required=True, help="a real-keys directory")
    parser.add_argument(
        "--fixed-dir", type=Path, help="hand-corrected labels, one <frame stem>.json per frame"
    )
    parser.add_argument("--clips", default="", help="comma separated recordings to score on")
    parser.add_argument("--decoder", choices=sorted(DECODERS), action="append")
    parser.add_argument("--threshold", type=float, default=PEAK_THRESHOLD)
    parser.add_argument(
        "--rectified", action="store_true", help="crop with the rectifying homography"
    )
    parser.add_argument(
        "--by-view",
        action="store_true",
        help="also score per board size and camera elevation: low <30, mid 30-60, high >60 degrees",
    )
    parser.add_argument("--json", type=Path, help="also write the scores here")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    clips = tuple(clip for clip in args.clips.split(",") if clip)
    frames = [
        frame
        for frame in load_synth_keys(args.frames)
        if not clips or any(clip in frame.image_path.stem for clip in clips)
    ]
    if args.points_out is not None:
        args.points_out.write_text(json.dumps(label_points(frames, args.fixed_dir)))
    results = {}
    for model in args.model:
        predict = onnx_predictor(model)
        for name in args.decoder or ["centroid"]:
            call = (frames, predict, DECODERS[name], args.threshold, args.fixed_dir, args.rectified)
            scores = score_by_view(*call) if args.by_view else {"all": score(*call)}
            for view, result in scores.items():
                label = f"{model.name} {name}" + ("" if view == "all" else f" {view}")
                results[label] = asdict(result)
                print(f"{label}: {result.frames} frames, fail {result.failure_rate:.3f}")
                for group, value in result.groups.items():
                    print(
                        f"  {group:8} n {value.labels:5}  recall {value.recall:.3f}  median "
                        f"{value.median_keys:.3f}  mean {value.mean_keys:.3f}  "
                        + "  ".join(f"pck@{k} {v:.3f}" for k, v in value.pck.items())
                    )
    if args.json is not None:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
