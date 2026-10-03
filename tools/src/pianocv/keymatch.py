"""KeyMatchNet: finds white-key boundaries and black-key edges in a rectified keybed strip.

The strip is the same rectification the browser runtime produces from an imperfect mask, so
every function here that touches geometry works from a possibly-perturbed quad, never the true
one, except where a docstring says otherwise.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from pianocv.export import strip_source_paths
from pianocv.segnet2 import normalise
from pianocv.template import strip_destination

STRIP_HEIGHT = 64
STRIP_WIDTH = 768
WHITE_ROW_FRACTION = 0.88
BLACK_ROW_FRACTION = 0.27
GAUSSIAN_SIGMA_PX = 1.5
PEAK_TOLERANCE_PX = 2.0

_STRIP_DST = strip_destination(STRIP_WIDTH, STRIP_HEIGHT)
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "synth-keys"


@dataclass(frozen=True)
class SidecarKey:
    pitch: int
    black: bool
    top: np.ndarray
    front: np.ndarray | None


@dataclass(frozen=True)
class SynthKeysFrame:
    image_path: Path
    # None for a negative: no keyboard in view, "keys" is empty too
    corners_px: np.ndarray | None
    keys: list[SidecarKey]
    image_size: tuple[int, int]
    # white where something covers the keys, as a hand does on a real frame, which training
    # leaves out
    ignore_mask: Path | None = None
    # labelled by the app from its own fit on a real recording, rather than rendered
    real: bool = False
    # a synth-motion clip id and its frame's position in it, when the sidecar carries them
    sequence: str | None = None
    frame_index: int | None = None


def _parse_face(raw: object, scale: np.ndarray, what: str) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{what} must list exactly 4 corners")
    points = np.zeros((4, 2), dtype=np.float64)
    for i, corner in enumerate(raw):
        if not isinstance(corner, dict):
            raise ValueError(f"{what} corner {i} is not an object")
        x, y = corner.get("x"), corner.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"{what} corner {i} has invalid coordinates")
        points[i] = (float(x), float(y))
    return np.asarray(points * scale, dtype=np.float64)


def _parse_key(raw: object, scale: np.ndarray) -> SidecarKey:
    if not isinstance(raw, dict):
        raise ValueError("key entry must be an object")
    pitch, black, top, front = raw.get("pitch"), raw.get("black"), raw.get("top"), raw.get("front")
    if not isinstance(pitch, int) or not isinstance(black, bool):
        raise ValueError("key entry has invalid pitch or black flag")
    return SidecarKey(
        pitch=pitch,
        black=black,
        top=_parse_face(top, scale, "key top face"),
        front=None if front is None else _parse_face(front, scale, "key front face"),
    )


def parse_synth_keys_sidecar(sidecar_path: Path) -> SynthKeysFrame:
    data = json.loads(sidecar_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sidecar {sidecar_path} is not a json object")
    width, height = data.get("imageWidth"), data.get("imageHeight")
    corners, keys = data.get("corners"), data.get("keys")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"sidecar {sidecar_path} has invalid image dimensions")
    if not isinstance(keys, list):
        raise ValueError(f"sidecar {sidecar_path} must list keys")
    if corners is None and keys:
        raise ValueError(f"sidecar {sidecar_path} has keys but no corners")
    ignore_mask = data.get("ignoreMask")
    if ignore_mask is not None and not isinstance(ignore_mask, str):
        raise ValueError(f"sidecar {sidecar_path} ignoreMask must name a file")
    sequence = data.get("sequence")
    if sequence is not None and not isinstance(sequence, str):
        raise ValueError(f"sidecar {sidecar_path} sequence must be a string")
    frame_index = data.get("frameIndex")
    if frame_index is not None and not isinstance(frame_index, int):
        raise ValueError(f"sidecar {sidecar_path} frameIndex must be an int")
    scale = np.array([float(width), float(height)])
    return SynthKeysFrame(
        image_path=sidecar_path.with_suffix(".png"),
        corners_px=None if corners is None else _parse_face(corners, scale, "keybed quad"),
        keys=[_parse_key(key, scale) for key in keys],
        image_size=(width, height),
        ignore_mask=None if ignore_mask is None else sidecar_path.parent / ignore_mask,
        real=data.get("kind") == "real-keys",
        sequence=sequence,
        frame_index=frame_index,
    )


def load_synth_keys(data_dir: Path = DEFAULT_DATA_DIR) -> list[SynthKeysFrame]:
    if not data_dir.is_dir():
        return []
    frames: list[SynthKeysFrame] = []
    for sidecar_path in sorted(data_dir.glob("*.json")):
        frame = parse_synth_keys_sidecar(sidecar_path)
        if frame.image_path.is_file():
            frames.append(frame)
    return frames


@dataclass(frozen=True)
class PerturbConfig:
    end_shift_fraction: float = 0.12
    far_outward_max_fraction: float = 0.25
    far_tilt_fraction: float = 0.05
    near_shift_fraction: float = 0.05
    jitter_fraction: float = 0.01
    unperturbed_fraction: float = 0.15


def _unit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def quad_axes(corners_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The keyboard's left-right and far-near directions, from the true (unperturbed) quad.

    A face's own corners carry no order guarantee, so every consumer classifies them by
    projecting onto these two axes instead of trusting a corner index.
    """
    far_left, far_right, near_right, near_left = corners_px
    key_axis = (far_right - far_left) + (near_right - near_left)
    depth_axis = (near_left - far_left) + (near_right - far_right)
    return _unit(key_axis), _unit(depth_axis)


def perturb_quad(
    corners_px: np.ndarray, rng: np.random.Generator, config: PerturbConfig | None = None
) -> np.ndarray:
    """A quad a real segmentation mask could have produced from the true one.

    The far edge only ever moves outward (a mask undershoots the case, never the keys), while
    the near edge and the two ends can move either way, so the strip sometimes crops a key off
    and sometimes runs onto the case.
    """
    config = config or PerturbConfig()
    if rng.random() < config.unperturbed_fraction:
        return np.array(corners_px, dtype=np.float64, copy=True)
    far_left, far_right, near_right, near_left = corners_px.astype(np.float64)
    key_axis, depth_axis = quad_axes(corners_px)
    quad_length = 0.5 * (
        np.linalg.norm(far_right - far_left) + np.linalg.norm(near_right - near_left)
    )
    depth = 0.5 * (np.linalg.norm(near_left - far_left) + np.linalg.norm(near_right - far_right))

    left_shift = rng.uniform(-config.end_shift_fraction, config.end_shift_fraction) * quad_length
    right_shift = rng.uniform(-config.end_shift_fraction, config.end_shift_fraction) * quad_length
    far_outward = rng.uniform(0.0, config.far_outward_max_fraction) * depth
    far_tilt_left = rng.uniform(-config.far_tilt_fraction, config.far_tilt_fraction) * depth
    far_tilt_right = rng.uniform(-config.far_tilt_fraction, config.far_tilt_fraction) * depth
    near_shift = rng.uniform(-config.near_shift_fraction, config.near_shift_fraction) * depth

    far_left = far_left + left_shift * key_axis - (far_outward + far_tilt_left) * depth_axis
    far_right = far_right + right_shift * key_axis - (far_outward + far_tilt_right) * depth_axis
    near_left = near_left + left_shift * key_axis + near_shift * depth_axis
    near_right = near_right + right_shift * key_axis + near_shift * depth_axis

    jitter_scale = config.jitter_fraction * quad_length
    jitter = rng.normal(scale=jitter_scale, size=(4, 2))
    return np.stack([far_left, far_right, near_right, near_left]) + jitter


def _homography(quad_px: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(quad_px.astype(np.float32), _STRIP_DST)


def rectify_strip(image_bgr: np.ndarray, quad_px: np.ndarray) -> np.ndarray:
    homography = _homography(quad_px)
    warped_bgr = cv2.warpPerspective(image_bgr, homography, (STRIP_WIDTH, STRIP_HEIGHT))
    return np.asarray(cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _project(points_px: np.ndarray, homography: np.ndarray) -> np.ndarray:
    ones = np.ones((points_px.shape[0], 1))
    homogeneous = np.concatenate([points_px, ones], axis=1)
    projected = homogeneous @ homography.T
    return np.asarray(projected[:, :2] / projected[:, 2:3])


def _side_edge(face_strip: np.ndarray, side: str) -> np.ndarray:
    """A key face's left or right edge, far corner first, classified in strip space.

    In the strip, depth runs straight down the rows, so the face splits cleanly into its far
    and near corner pairs. Classifying in the camera frame instead picks the key's front edge
    whenever perspective tilts the depth direction along the keyboard, as oblique views do.
    """
    by_depth = face_strip[np.argsort(face_strip[:, 1])]
    far, near = by_depth[:2], by_depth[2:]
    choose = np.argmax if side == "right" else np.argmin
    return np.stack([far[choose(far[:, 0])], near[choose(near[:, 0])]])


def _row_crossing(p0: np.ndarray, p1: np.ndarray, row: float) -> float | None:
    if p0[1] == p1[1]:
        return None
    t = (row - p0[1]) / (p1[1] - p0[1])
    if t < 0.0 or t > 1.0:
        return None
    return float(p0[0] + t * (p1[0] - p0[0]))


def _crossing_in_strip(
    face_strip: np.ndarray,
    side: str,
    row: float,
    to_frame: np.ndarray,
    image_size: tuple[int, int],
) -> float | None:
    """Where a face's edge crosses `row` of the strip, if that spot is in the strip and was
    actually in the camera's picture; an edge the camera never saw is not a target."""
    edge = _side_edge(face_strip, side)
    x = _row_crossing(edge[0], edge[1], row)
    if x is None or not (0.0 <= x < STRIP_WIDTH):
        return None
    frame_x, frame_y = _project(np.array([[x, row]]), to_frame)[0]
    width, height = image_size
    if not (0.0 <= frame_x < width and 0.0 <= frame_y < height):
        return None
    return x


def _rasterize(positions: Sequence[float]) -> np.ndarray:
    axis = np.arange(STRIP_WIDTH, dtype=np.float64)
    channel = np.zeros(STRIP_WIDTH, dtype=np.float64)
    for x in positions:
        channel = np.maximum(channel, np.exp(-((axis - x) ** 2) / (2.0 * GAUSSIAN_SIGMA_PX**2)))
    return channel


@dataclass(frozen=True)
class KeyMatchTargets:
    heatmaps: np.ndarray  # (3, STRIP_WIDTH) float32: white boundary, black left, black right
    white_positions: list[float] = field(default_factory=list)
    black_left_positions: list[float] = field(default_factory=list)
    black_right_positions: list[float] = field(default_factory=list)


def compute_targets(frame: SynthKeysFrame, quad_px: np.ndarray) -> KeyMatchTargets:
    """Where every key edge lands in the strip once it is rectified from `quad_px`.

    Both faces and the strip's own homography are built from frame-pixel coordinates, so a
    quad that crops a key off simply maps that key's edge outside the strip and it is dropped,
    rather than needing a separate cut-off check.
    """
    homography = _homography(quad_px)
    to_frame = np.linalg.inv(homography)
    white_row = WHITE_ROW_FRACTION * (STRIP_HEIGHT - 1)
    black_row = BLACK_ROW_FRACTION * (STRIP_HEIGHT - 1)

    def crossing(key: SidecarKey, side: str, row: float) -> float | None:
        face = _project(key.top, homography)
        return _crossing_in_strip(face, side, row, to_frame, frame.image_size)

    white_keys = sorted((key for key in frame.keys if not key.black), key=lambda key: key.pitch)
    white_positions: list[float] = []
    for left_key, right_key in pairwise(white_keys):
        # the boundary is the middle of the gap, between one key's right edge and the next's left
        sides = [
            x
            for x in (
                crossing(left_key, "right", white_row),
                crossing(right_key, "left", white_row),
            )
            if x is not None
        ]
        if sides:
            white_positions.append(sum(sides) / len(sides))

    black_left_positions: list[float] = []
    black_right_positions: list[float] = []
    for key in frame.keys:
        if not key.black:
            continue
        for side, bucket in (
            ("left", black_left_positions),
            ("right", black_right_positions),
        ):
            x = crossing(key, side, black_row)
            if x is not None:
                bucket.append(x)

    heatmaps = np.stack(
        [
            _rasterize(white_positions),
            _rasterize(black_left_positions),
            _rasterize(black_right_positions),
        ]
    ).astype(np.float32)
    return KeyMatchTargets(heatmaps, white_positions, black_left_positions, black_right_positions)


_HEIGHT_CHANNELS = (16, 24, 32, 32, 32, 32)
_WIDTH_CHANNELS = 48
_DILATIONS = (1, 2, 4, 8)


class KeyMatchNet(nn.Module):
    """Collapses the strip's height, then reads along its width with dilated 1D convolutions.

    The dilations give each output position a view of three to four keys, wide enough to use
    the two-and-three black-key pattern to disambiguate a blurred or occluded edge.
    """

    def __init__(self) -> None:
        super().__init__()
        height_layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in _HEIGHT_CHANNELS:
            height_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=(2, 1), padding=1),
                nn.GroupNorm(4, out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
        self.height_body = nn.Sequential(*height_layers)

        width_layers: list[nn.Module] = []
        in_channels = _HEIGHT_CHANNELS[-1]
        for dilation in _DILATIONS:
            width_layers += [
                nn.Conv1d(
                    in_channels, _WIDTH_CHANNELS, kernel_size=3, padding=dilation, dilation=dilation
                ),
                nn.GroupNorm(4, _WIDTH_CHANNELS),
                nn.ReLU(inplace=True),
            ]
            in_channels = _WIDTH_CHANNELS
        width_layers += [
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        self.width_body = nn.Sequential(*width_layers)
        self.head = nn.Conv1d(32, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.height_body(x)
        features = features.squeeze(2)
        features = self.width_body(features)
        logits: torch.Tensor = self.head(features)
        return logits


def focal_heatmap_loss(
    logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0
) -> torch.Tensor:
    """Quality focal loss: a BCE weighted by how far the prediction is from a continuous target.

    Peaks are sparse and most targets are not exactly 0 or 1 (a Gaussian tail), so we weight by
    |target - prediction| rather than the usual positive/negative split, which needs a hard
    label CornerNet-style focal loss does not have here.
    """
    probability = torch.sigmoid(logits)
    weight = (target - probability).abs().pow(gamma)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (weight * bce).mean()


def find_peaks(
    values: np.ndarray, threshold: float = 0.5, min_distance_px: float = 3.0
) -> list[float]:
    order = np.argsort(values)[::-1]
    picked: list[float] = []
    for index in order:
        if values[index] < threshold:
            break
        if all(abs(float(index) - p) >= min_distance_px for p in picked):
            picked.append(float(index))
    return sorted(picked)


def match_peaks(
    predicted: Sequence[float], truth: Sequence[float], tolerance_px: float
) -> tuple[int, int, int, list[float]]:
    remaining = list(truth)
    errors: list[float] = []
    true_positives = 0
    for p in predicted:
        if not remaining:
            break
        distances = [abs(p - t) for t in remaining]
        best = int(np.argmin(distances))
        if distances[best] <= tolerance_px:
            true_positives += 1
            errors.append(distances[best])
            remaining.pop(best)
    false_positives = len(predicted) - true_positives
    false_negatives = len(remaining)
    return true_positives, false_positives, false_negatives, errors


@dataclass(frozen=True)
class ChannelMetrics:
    precision: float
    recall: float
    mean_error_px: float


def channel_metrics(
    predicted: Sequence[float], truth: Sequence[float], tolerance_px: float = PEAK_TOLERANCE_PX
) -> ChannelMetrics:
    true_positives, false_positives, false_negatives, errors = match_peaks(
        predicted, truth, tolerance_px
    )
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    mean_error = float(np.mean(errors)) if errors else 0.0
    return ChannelMetrics(precision, recall, mean_error)


def preprocess_strip(strip_rgb_uint8: np.ndarray) -> np.ndarray:
    return np.asarray(normalise(strip_rgb_uint8[None])[0], dtype=np.float32)


class _Sigmoid(nn.Module):
    def __init__(self, model: KeyMatchNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def load_keymatch(path: Path) -> KeyMatchNet:
    model = KeyMatchNet()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def export_keymatch_onnx(model_path: Path, onnx_path: Path) -> Path:
    """Mirrors pianocv.export's `_write`: same export call, reusing its doc-string scrub."""
    model = load_keymatch(model_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        _Sigmoid(model).eval(),
        (torch.zeros(1, 3, STRIP_HEIGHT, STRIP_WIDTH),),
        str(onnx_path),
        dynamo=True,
        optimize=True,
        external_data=False,
        opset_version=20,
        input_names=["strip"],
        output_names=["heatmaps"],
    )
    return strip_source_paths(onnx_path)
