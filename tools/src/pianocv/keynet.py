"""KeyNet's label side: the crop geometry, keypoints and heatmap targets the keypoint model
trains against.

Every role (low/high, back/front) is read off the keys' own geometry, never off a sidecar's
corner order, since that order is a convention some sources keep and others do not.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pianocv.keymatch import SidecarKey, SynthKeysFrame, quad_axes

TRACK_WIDTH = 768
TRACK_HEIGHT = 160
TRACK_MARGIN_ALONG = 0.08
TRACK_MARGIN_ACROSS = 0.15
SEARCH_SIZE = 256

STRIDE = 2
GAUSSIAN_SIGMA_CELLS = 1.0
IGNORE_RADIUS_CELLS = 3
CHANNELS = 12
PRESENCE_MIN_VISIBLE_FRACTION = 0.2
IGNORE_THRESHOLD = 127


@dataclass(frozen=True)
class Crop:
    """The 3x3 matrix that takes frame pixels to crop pixels; a 2x3 affine is given the row
    [0, 0, 1]."""

    to_crop: np.ndarray

    def __post_init__(self) -> None:
        if self.to_crop.shape == (2, 3):
            object.__setattr__(self, "to_crop", np.vstack([self.to_crop, [0.0, 0.0, 1.0]]))

    def points(self, frame_px: np.ndarray) -> np.ndarray:
        ones = np.ones((len(frame_px), 1))
        projected = np.hstack([frame_px, ones]) @ self.to_crop.T
        return np.asarray(projected[:, :2] / projected[:, 2:3], dtype=np.float64)


def oriented_crop(
    quad_px: np.ndarray,
    width: int,
    height: int,
    margin_along: float,
    margin_across: float,
) -> Crop:
    """An oriented crop with the keys running left to right and the player's edge at the bottom.

    Scale is chosen so both the quad's length and its depth fit with their margins, the same
    way the browser runtime crops, which is what lets the model run on a live frame.
    """
    key_axis, depth_axis = quad_axes(quad_px)
    across = np.array([-key_axis[1], key_axis[0]])
    if float(across @ depth_axis) < 0:
        across = -across
    centre = quad_px.mean(axis=0)
    along_extent = np.ptp((quad_px - centre) @ key_axis) * (1 + 2 * margin_along)
    across_extent = np.ptp((quad_px - centre) @ across) + along_extent * margin_across / (
        1 + 2 * margin_along
    )
    scale = max(along_extent / width, across_extent / height, 1e-6)
    rows = np.stack([key_axis / scale, across / scale])
    offset = np.array([width / 2, height / 2]) - rows @ centre
    return Crop(to_crop=np.hstack([rows, offset[:, None]]))


def rectified_crop(
    quad_px: np.ndarray,
    width: int,
    height: int,
    margin_along: float,
    margin_across: float,
) -> Crop:
    """The homography that sends the quad (back-low, back-high, front-high, front-low) onto a
    fixed rectangle, so every key is equally wide in the crop. Each margin is a share of the
    rectangle's extent on both sides: back-low goes to (width * ma / (1 + 2 ma), height * mx /
    (1 + 2 mx)) and front-high to the mirrored corner."""
    left = width * margin_along / (1 + 2 * margin_along)
    top = height * margin_across / (1 + 2 * margin_across)
    rectangle = np.array(
        [(left, top), (width - left, top), (width - left, height - top), (left, height - top)],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad_px.astype(np.float32), rectangle)
    return Crop(to_crop=np.asarray(matrix, dtype=np.float64))


def squash_crop(frame_size: tuple[int, int], width: int, height: int) -> Crop:
    """The whole frame resized to `width` x `height`, aspect ratio dropped, for search mode."""
    frame_width, frame_height = frame_size
    to_crop = np.array([[width / frame_width, 0.0, 0.0], [0.0, height / frame_height, 0.0]])
    return Crop(to_crop=to_crop)


@dataclass(frozen=True)
class KeyPoints:
    """Every keypoint in frame pixels.

    `corners` is (4, 2): back-low, back-high, front-high, front-low. `gaps` is (n - 1, 2), the
    white-key boundaries at the front edge. `black_low` and `black_high` are (m, 2) each, one
    entry per black key with a front face, low-pitch and high-pitch side, where the front face
    meets the keybed; `black_top_low` and `black_top_high` are where it meets the key's top.
    `back_gaps` is (n - 1, 2), the white-key boundaries at the back edge, and `black_back_low`
    and `black_back_high` are (m, 2) each, the black keys' top-face corners at their back end.
    """

    corners: np.ndarray
    gaps: np.ndarray
    black_low: np.ndarray
    black_high: np.ndarray
    black_top_low: np.ndarray
    black_top_high: np.ndarray
    back_gaps: np.ndarray
    black_back_low: np.ndarray
    black_back_high: np.ndarray
    # per corner: True when the case hides it, so it has no peak and no loss
    corner_hidden: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=bool))


def _unit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def _split_by_depth(
    corners: np.ndarray, cross_axis: np.ndarray, black_reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """A white key's 4 top-face corners, split into its back pair and its front pair.

    The two pairs are told apart by which one sits closer to the black keys along the axis
    across the keyboard, per the spec: the front edge is farther from the black keys.
    """
    projection = corners @ cross_axis
    order = np.argsort(projection)
    first_pair, second_pair = corners[order[:2]], corners[order[2:]]
    black_projection = float(black_reference @ cross_axis)
    first_distance = abs(float(projection[order[:2]].mean()) - black_projection)
    second_distance = abs(float(projection[order[2:]].mean()) - black_projection)
    if first_distance <= second_distance:
        return first_pair, second_pair
    return second_pair, first_pair


def _low_high(pair: np.ndarray, key_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projection = pair @ key_axis
    order = np.argsort(projection)
    return pair[order[0]], pair[order[1]]


@dataclass(frozen=True)
class _KeyCorners:
    back_low: np.ndarray
    back_high: np.ndarray
    front_low: np.ndarray
    front_high: np.ndarray


def _white_key_corners(
    whites: list[SidecarKey],
    key_axis: np.ndarray,
    cross_axis: np.ndarray,
    black_reference: np.ndarray,
) -> list[_KeyCorners]:
    corners = []
    for white in whites:
        back_pair, front_pair = _split_by_depth(white.top, cross_axis, black_reference)
        back_low, back_high = _low_high(back_pair, key_axis)
        front_low, front_high = _low_high(front_pair, key_axis)
        corners.append(_KeyCorners(back_low, back_high, front_low, front_high))
    return corners


@dataclass(frozen=True)
class _BlackFront:
    bottom_low: np.ndarray
    bottom_high: np.ndarray
    top_low: np.ndarray
    top_high: np.ndarray
    back_low: np.ndarray
    back_high: np.ndarray


def _black_key_front(key: SidecarKey, key_axis: np.ndarray) -> _BlackFront | None:
    """A black key's front face corners, low-pitch and high-pitch side.

    Two of the front face's corners sit on the top face (the drop's upper edge) and the other
    two on the keybed plane. The pair on the top face says how far the key stands up in this
    view, which the runtime cannot know from the plane alone.
    """
    if key.front is None:
        return None
    shared = [c for c in key.front if np.min(np.linalg.norm(key.top - c, axis=1)) < 1e-6]
    lower = [c for c in key.front if np.min(np.linalg.norm(key.top - c, axis=1)) >= 1e-6]
    if len(shared) != 2 or len(lower) != 2:
        return None
    bottom_low, bottom_high = _low_high(np.array(lower), key_axis)
    top_low, top_high = _low_high(np.array(shared), key_axis)
    back = [c for c in key.top if np.min(np.linalg.norm(np.array(shared) - c, axis=1)) >= 1e-6]
    back_low, back_high = _low_high(np.array(back), key_axis)
    return _BlackFront(bottom_low, bottom_high, top_low, top_high, back_low, back_high)


def keypoints(frame: SynthKeysFrame) -> KeyPoints | None:
    """The frame's keypoints, derived from key geometry rather than sidecar corner order.

    None when the frame has no keys, or too few white keys to define a keyboard axis (a single
    white key gives no meaningful low/high end or front/back split).
    """
    whites = sorted((key for key in frame.keys if not key.black), key=lambda key: key.pitch)
    if len(whites) < 2:
        return None

    key_axis = _unit(whites[-1].top.mean(axis=0) - whites[0].top.mean(axis=0))
    cross_axis = np.array([-key_axis[1], key_axis[0]])
    black_tops = [key.top for key in frame.keys if key.black]
    black_reference = (
        np.vstack(black_tops).mean(axis=0) if black_tops else whites[0].top.mean(axis=0)
    )

    white_corners = _white_key_corners(whites, key_axis, cross_axis, black_reference)
    corners = np.array(
        [
            white_corners[0].back_low,
            white_corners[-1].back_high,
            white_corners[-1].front_high,
            white_corners[0].front_low,
        ]
    )
    gaps = np.array(
        [
            (white_corners[i].front_high + white_corners[i + 1].front_low) / 2.0
            for i in range(len(white_corners) - 1)
        ]
    )

    back_gaps = np.array(
        [
            (white_corners[i].back_high + white_corners[i + 1].back_low) / 2.0
            for i in range(len(white_corners) - 1)
        ]
    )

    fronts = [
        front
        for key in sorted((key for key in frame.keys if key.black), key=lambda key: key.pitch)
        if (front := _black_key_front(key, key_axis)) is not None
    ]
    return KeyPoints(
        corners=corners,
        corner_hidden=_hidden_corners(frame, corners),
        gaps=gaps,
        black_low=_stacked([front.bottom_low for front in fronts]),
        black_high=_stacked([front.bottom_high for front in fronts]),
        black_top_low=_stacked([front.top_low for front in fronts]),
        black_top_high=_stacked([front.top_high for front in fronts]),
        back_gaps=back_gaps,
        black_back_low=_stacked([front.back_low for front in fronts]),
        black_back_high=_stacked([front.back_high for front in fronts]),
    )


def _hidden_corners(frame: SynthKeysFrame, corners: np.ndarray) -> np.ndarray:
    """The roles' hidden flags: each role corner takes the flag of the sidecar corner nearest it."""
    if frame.corner_visible is None or frame.corners_px is None:
        return np.zeros(4, dtype=bool)
    nearest = np.linalg.norm(corners[:, None] - frame.corners_px[None], axis=2).argmin(axis=1)
    return np.asarray(~np.array(frame.corner_visible)[nearest])


def _stacked(points: list[np.ndarray]) -> np.ndarray:
    return np.array(points) if points else np.empty((0, 2))


def _nan_rows(entries: list[list[float] | None]) -> np.ndarray:
    if not entries:
        return np.empty((0, 2))
    return np.array(
        [[np.nan, np.nan] if entry is None else entry for entry in entries], dtype=float
    )


def _back_rows(
    data: dict[str, list[list[float] | None]], key: str, count: int
) -> list[list[float] | None]:
    """The file's rows for a back label, or `count` hidden rows when the file predates it."""
    rows = data.get(key)
    return [None] * count if rows is None else rows


def has_back_labels(path: Path | None) -> bool:
    return path is not None and "backGaps" in json.loads(path.read_text())


def load_fixed_points(path: Path) -> KeyPoints:
    """A hand-corrected label file as keypoints, with every hidden point a NaN row so the array
    shapes stay fixed."""
    data = json.loads(path.read_text())
    return KeyPoints(
        corners=_nan_rows(data["corners"]),
        gaps=_nan_rows(data["gaps"]),
        black_low=_nan_rows(data["blackLow"]),
        black_high=_nan_rows(data["blackHigh"]),
        black_top_low=_nan_rows(data["blackTopLow"]),
        black_top_high=_nan_rows(data["blackTopHigh"]),
        back_gaps=_nan_rows(_back_rows(data, "backGaps", len(data["gaps"]))),
        black_back_low=_nan_rows(_back_rows(data, "blackBackLow", len(data["blackLow"]))),
        black_back_high=_nan_rows(_back_rows(data, "blackBackHigh", len(data["blackHigh"]))),
    )


def fixed_path(frame: SynthKeysFrame, fixed_dir: Path | None) -> Path | None:
    if fixed_dir is None:
        return None
    path = fixed_dir / f"{frame.image_path.stem}.json"
    return path if path.is_file() else None


def frame_points(frame: SynthKeysFrame, fixed_dir: Path | None) -> KeyPoints | None:
    """The corrected points when the fixed directory has this frame, else the derived ones."""
    path = fixed_path(frame, fixed_dir)
    return keypoints(frame) if path is None else load_fixed_points(path)


def presence_target(frame: SynthKeysFrame) -> float:
    """1 when the frame has keys and at least 20% of the keybed quad lies inside the frame."""
    if not frame.keys or frame.corners_px is None:
        return 0.0
    width, height = frame.image_size
    quad = frame.corners_px.astype(np.float32).reshape(-1, 1, 2)
    frame_rect = np.array(
        [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)], dtype=np.float32
    ).reshape(-1, 1, 2)
    quad_area = float(cv2.contourArea(quad))
    if quad_area <= 0.0:
        return 0.0
    visible_area, _ = cv2.intersectConvexConvex(quad, frame_rect)
    return 1.0 if visible_area / quad_area >= PRESENCE_MIN_VISIBLE_FRACTION else 0.0


def _bounds(centre: float, radius: float, size: int) -> tuple[int, int]:
    """A [low, high) index range around `centre`, clipped so it never wraps via a negative
    index when `centre` sits far outside the array."""
    low = int(np.clip(np.floor(centre - radius), 0, size))
    high = int(np.clip(np.ceil(centre + radius) + 1, 0, size))
    return low, high


def _add_gaussian(channel: np.ndarray, cell: np.ndarray, sigma: float) -> None:
    height, width = channel.shape
    cx, cy = cell
    radius = 3 * sigma
    x0, x1 = _bounds(cx, radius, width)
    y0, y1 = _bounds(cy, radius, height)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    gaussian = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2)))
    channel[y0:y1, x0:x1] = np.maximum(channel[y0:y1, x0:x1], gaussian)


def _zero_radius(channel: np.ndarray, cell: np.ndarray, radius: int) -> None:
    height, width = channel.shape
    cx, cy = cell
    x0, x1 = _bounds(cx, radius, width)
    y0, y1 = _bounds(cy, radius, height)
    channel[y0:y1, x0:x1] = 0.0


def _splat(
    heat_channel: np.ndarray,
    weight_channel: np.ndarray,
    frame_point: np.ndarray,
    crop: Crop,
    input_width: int,
    input_height: int,
    frame_width: int,
    frame_height: int,
    ignore: np.ndarray | None,
) -> None:
    crop_xy = crop.points(frame_point[None])[0]
    # a cell covers input pixels 2j and 2j + 1, so its centre is input pixel 2j + 0.5, which is
    # where the browser's decoder reads it back
    cell = crop_xy / STRIDE - 0.25
    visible = (
        0.0 <= frame_point[0] < frame_width
        and 0.0 <= frame_point[1] < frame_height
        and 0.0 <= crop_xy[0] < input_width
        and 0.0 <= crop_xy[1] < input_height
    )
    if visible and ignore is not None:
        row, col = round(float(frame_point[1])), round(float(frame_point[0]))
        if 0 <= row < ignore.shape[0] and 0 <= col < ignore.shape[1]:
            visible = ignore[row, col] <= IGNORE_THRESHOLD
    if not visible:
        _zero_radius(weight_channel, cell, IGNORE_RADIUS_CELLS)
        return
    _add_gaussian(heat_channel, cell, GAUSSIAN_SIGMA_CELLS)


BLACK_TOP_CHANNELS = (7, 8)
BACK_CHANNELS = (9, 10, 11)
# the cells around a point that learn its offset: any of them can be the peak a decoder picks
OFFSET_RADIUS_CELLS = 1


def shown_corners(points: KeyPoints) -> list[np.ndarray]:
    """Each corner as a one-row array, empty when the case hides it."""
    return [
        np.empty((0, 2)) if points.corner_hidden[k] else points.corners[k : k + 1] for k in range(4)
    ]


def _channel_points(points: KeyPoints) -> list[tuple[int, np.ndarray]]:
    """Each channel's visible points; a hidden (NaN) point gets no peak."""
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
    return [
        (channel, rows[np.isfinite(rows).all(axis=1)]) for channel, rows in enumerate(by_channel)
    ]


def offset_targets(
    points: KeyPoints | None,
    crop: Crop,
    input_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """For each channel, how far its point sits from each nearby cell's centre (x then y, in
    cells, shape (2 * 12, H/2, W/2)), and how much each cell's offset counts: the point's
    Gaussian there, so the cells a decoder is likely to pick count most."""
    input_height, input_width = input_size
    rows, cols = input_height // STRIDE, input_width // STRIDE
    offset = np.zeros((2 * CHANNELS, rows, cols), dtype=np.float32)
    weight = np.zeros((CHANNELS, rows, cols), dtype=np.float32)
    if points is None:
        return offset, weight
    for channel, frame_points in _channel_points(points):
        for point in frame_points:
            cx, cy = crop.points(point[None])[0] / STRIDE - 0.25
            for i in range(round(cy) - OFFSET_RADIUS_CELLS, round(cy) + OFFSET_RADIUS_CELLS + 1):
                for j in range(
                    round(cx) - OFFSET_RADIUS_CELLS, round(cx) + OFFSET_RADIUS_CELLS + 1
                ):
                    if not (0 <= i < rows and 0 <= j < cols):
                        continue
                    near = float(
                        np.exp(-((j - cx) ** 2 + (i - cy) ** 2) / (2 * GAUSSIAN_SIGMA_CELLS**2))
                    )
                    if near > weight[channel, i, j]:
                        weight[channel, i, j] = near
                        offset[2 * channel, i, j] = cx - j
                        offset[2 * channel + 1, i, j] = cy - i
    return offset, weight


def heatmap_targets(
    points: KeyPoints | None,
    crop: Crop,
    input_size: tuple[int, int],
    frame_size: tuple[int, int],
    ignore: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """The (12, H/2, W/2) heatmap and per-cell loss weight for one crop, at stride 2."""
    input_height, input_width = input_size
    heat = np.zeros((CHANNELS, input_height // STRIDE, input_width // STRIDE), dtype=np.float32)
    weight = np.ones_like(heat)
    if points is None:
        return heat, weight

    frame_width, frame_height = frame_size
    for hidden in np.flatnonzero(points.corner_hidden):
        cell = crop.points(points.corners[hidden][None])[0] / STRIDE - 0.25
        _zero_radius(weight[hidden], cell, IGNORE_RADIUS_CELLS)
    for channel, frame_points in _channel_points(points):
        for point in frame_points:
            _splat(
                heat[channel],
                weight[channel],
                point,
                crop,
                input_width,
                input_height,
                frame_width,
                frame_height,
                ignore,
            )
    return heat.astype(np.float32), weight.astype(np.float32)
