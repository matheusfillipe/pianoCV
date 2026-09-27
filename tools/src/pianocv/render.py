"""Synthetic keybed sample renderer for detector training."""

import math
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pianocv.dataset import DEFAULT_FRAMES_DIR, canonical_quad, load_frames

_WHITE_COUNT = 52.0
_BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
_BLACK_WIDTH = 0.58
_STRIP_DEPTH = 6.38
_BLACK_DEPTH_FRACTION = 0.6
_PIXELS_PER_UNIT = 20.0

_ABSENT_PROBABILITY = 0.1
_REAL_BACKGROUND_PROBABILITY = 0.5
_REAL_NEGATIVE_PROBABILITY = 0.5
_INPAINT_RADIUS = 6
_INPAINT_MARGIN_PX = 8
_SCALE_RANGE = (0.6, 1.4)
_MIN_VISIBLE_SPAN = 0.7
_JITTER_FRACTION = 0.06

_WHITE_BRIGHTNESS_RANGE = (170.0, 235.0)
_BLACK_BRIGHTNESS_RANGE = (15.0, 50.0)
_KEY_SEAM_BRIGHTNESS_RANGE = (60.0, 100.0)
_BLUR_SIGMA_RANGE = (0.5, 1.5)
_GLARE_PROBABILITY = 0.3
_GLARE_AMPLITUDE_RANGE = (30.0, 90.0)
_OCCLUDER_PROBABILITY = 0.6
_OCCLUDER_MAX_AREA_FRACTION = 0.4
_NOISE_SIGMA_RANGE = (2.0, 6.0)

_FOCAL_RANGE = (0.6, 2.4)
_SPAN_RANGE = (0.35, 1.6)
_ELEVATION_RANGE = (18.0, 88.0)
_AZIMUTH_RANGE = (-70.0, 70.0)
_ROLL_RANGE = (-30.0, 30.0)
_PRINCIPAL_RANGE = (0.1, 0.9)
_MIN_VISIBLE_FRACTION = 0.45
_QUAD_ATTEMPTS = 24

# white keys on the boards people own: 25, 37, 49, 61, 73, 76 and 88 key instruments
_WHITE_COUNTS = (15, 22, 29, 36, 43, 45, 52)


def _world_corners(white_count: float) -> np.ndarray:
    half = white_count / 2.0
    depth = _STRIP_DEPTH / 2.0
    return np.array(
        [[-half, -depth, 0.0], [half, -depth, 0.0], [half, depth, 0.0], [-half, depth, 0.0]]
    )


@dataclass
class RenderSample:
    image: np.ndarray
    quad_px: np.ndarray
    present: bool
    white_count: float = _WHITE_COUNT

    @property
    def mask(self) -> np.ndarray:
        # the mask covers the whole keybed even where a hand hides it, so the net learns to fill in
        height, width = self.image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        if self.present:
            cv2.fillPoly(mask, [self.quad_px.astype(np.int32)], 255)
        return mask


def render_sample(
    rng: np.random.Generator,
    width: int = 640,
    height: int = 480,
    background: np.ndarray | None = None,
    quad_px: np.ndarray | None = None,
) -> RenderSample:
    if background is not None and quad_px is not None:
        return _render_with_background(rng, background, quad_px, width, height)
    present = bool(rng.random() >= _ABSENT_PROBABILITY)
    real_probability = _REAL_BACKGROUND_PROBABILITY if present else _REAL_NEGATIVE_PROBABILITY
    sampled = (
        _sample_real_background(rng, width, height) if rng.random() < real_probability else None
    )
    if sampled is not None:
        real_background, real_quad = sampled
        return _render_with_background(rng, real_background, real_quad, width, height, present)
    white_count = float(rng.choice(_WHITE_COUNTS))
    quad = _sample_quad(rng, width, height, white_count)
    canvas = _render_background(rng, width, height)
    if present:
        _maybe_add_control_panel(canvas, quad, rng)
        _draw_keybed(canvas, quad, rng, white_count)
        _maybe_add_glare(canvas, rng)
        _maybe_add_occluders(canvas, quad, rng)
    return _finalize(canvas, quad, present, rng, white_count)


def _finalize(
    canvas: np.ndarray,
    quad: np.ndarray,
    present: bool,
    rng: np.random.Generator,
    white_count: float = _WHITE_COUNT,
) -> RenderSample:
    sigma = float(rng.uniform(*_NOISE_SIGMA_RANGE))
    noisy = canvas + rng.normal(0.0, sigma, canvas.shape)
    return RenderSample(
        image=np.asarray(np.clip(noisy, 0.0, 255.0), dtype=np.float32),
        quad_px=quad,
        present=present,
        white_count=white_count,
    )


_REAL_FRAMES: list[tuple[Path, np.ndarray]] | None = None


# Real recordings are the backgrounds these samples are composited onto, so a recording left in
# here is training data no matter what a caller passes for real_fraction. Anything being held
# out for evaluation has to be named here too, or its score is measured on frames it has seen.
BACKGROUND_EXCLUDE_ENV = "PIANOCV_BACKGROUND_EXCLUDE"


def _excluded() -> tuple[str, ...]:
    raw = os.environ.get(BACKGROUND_EXCLUDE_ENV, "")
    return tuple(part for part in (p.strip() for p in raw.split(",")) if part)


def _real_frames() -> list[tuple[Path, np.ndarray]]:
    global _REAL_FRAMES
    if _REAL_FRAMES is None:
        excluded = _excluded()
        _REAL_FRAMES = [
            (frame.image_path, frame.corners_px)
            for frame in load_frames(DEFAULT_FRAMES_DIR)
            if frame.kind == "rec"
            and frame.corners_px is not None
            and not any(mark in frame.source_stem for mark in excluded)
        ]
    return _REAL_FRAMES


def _sample_real_background(
    rng: np.random.Generator, width: int, height: int
) -> tuple[np.ndarray, np.ndarray] | None:
    frames = _real_frames()
    if not frames:
        return None
    image_path, quad = frames[int(rng.integers(0, len(frames)))]
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    background = np.asarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dtype=np.float32)
    return background, quad.astype(np.float64)


def _render_with_background(
    rng: np.random.Generator,
    background: np.ndarray,
    quad_px: np.ndarray,
    width: int,
    height: int,
    present: bool | None = None,
) -> RenderSample:
    image = np.asarray(background, dtype=np.float32).copy()
    source_height, source_width = image.shape[:2]
    base = canonical_quad(quad_px.astype(np.float64))
    if (source_width, source_height) != (width, height):
        image = np.asarray(
            cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), dtype=np.float32
        )
        base = base * np.array([width / source_width, height / source_height])
    canvas = _inpaint_quad(image, base)
    if present is None:
        present = bool(rng.random() >= _ABSENT_PROBABILITY)
    quad = _perturb_quad(rng, base, width, height)
    if present:
        _draw_keybed(canvas, quad, rng, float(rng.choice(_WHITE_COUNTS)))
        _maybe_add_glare(canvas, rng)
        _maybe_add_occluders(canvas, quad, rng)
    return _finalize(canvas, quad, present, rng)


def _inpaint_quad(image: np.ndarray, quad_px: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(int(np.floor(quad_px[:, 0].min())) - _INPAINT_MARGIN_PX, 0)
    y0 = max(int(np.floor(quad_px[:, 1].min())) - _INPAINT_MARGIN_PX, 0)
    x1 = min(int(np.ceil(quad_px[:, 0].max())) + _INPAINT_MARGIN_PX, width)
    y1 = min(int(np.ceil(quad_px[:, 1].max())) + _INPAINT_MARGIN_PX, height)
    if x1 <= x0 or y1 <= y0:
        return image
    crop = image[y0:y1, x0:x1]
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    offset = np.array([float(x0), float(y0)])
    cv2.fillPoly(mask, [(quad_px - offset).astype(np.int32)], 255)
    inpainted = cv2.inpaint(
        np.clip(crop, 0.0, 255.0).astype(np.uint8), mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA
    )
    image[y0:y1, x0:x1] = np.asarray(inpainted, dtype=np.float32)
    return image


def _sample_quad(
    rng: np.random.Generator, width: int, height: int, white_count: float = _WHITE_COUNT
) -> np.ndarray:
    best = _project_keybed(rng, width, height, white_count)
    # a keybed that missed the frame teaches nothing, so keep drawing until enough of it lands
    for _ in range(_QUAD_ATTEMPTS):
        if _visible_fraction(best, width, height) >= _MIN_VISIBLE_FRACTION:
            return best
        candidate = _project_keybed(rng, width, height, white_count)
        if _visible_fraction(candidate, width, height) > _visible_fraction(best, width, height):
            best = candidate
    return best


def _visible_fraction(quad: np.ndarray, width: int, height: int) -> float:
    edge = np.linspace(quad[0], quad[1], 64)
    inside = (
        (edge[:, 0] >= 0.0) & (edge[:, 0] < width) & (edge[:, 1] >= 0.0) & (edge[:, 1] < height)
    )
    return float(inside.mean())


def _project_keybed(
    rng: np.random.Generator, width: int, height: int, white_count: float
) -> np.ndarray:
    focal = float(rng.uniform(*_FOCAL_RANGE)) * width
    span = float(rng.uniform(*_SPAN_RANGE)) * width
    # keeping the camera outside the keybed's bounding sphere keeps every corner in front of it
    distance = max(white_count, focal * white_count / span)
    elevation = math.radians(float(rng.uniform(*_ELEVATION_RANGE)))
    azimuth = math.radians(float(rng.uniform(*_AZIMUTH_RANGE)))
    position = distance * np.array(
        [
            math.cos(elevation) * math.sin(azimuth),
            math.cos(elevation) * math.cos(azimuth),
            math.sin(elevation),
        ]
    )
    rotation = _look_at_origin(position, math.radians(float(rng.uniform(*_ROLL_RANGE))))
    camera = (_world_corners(white_count) - position) @ rotation.T
    principal = np.array([float(width), float(height)]) * rng.uniform(*_PRINCIPAL_RANGE, size=2)
    return canonical_quad(focal * camera[:, :2] / camera[:, 2:3] + principal)


def _look_at_origin(position: np.ndarray, roll: float) -> np.ndarray:
    forward = -position / float(np.linalg.norm(position))
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right = right / float(np.linalg.norm(right))
    down = np.cross(forward, right)
    cos_roll = math.cos(roll)
    sin_roll = math.sin(roll)
    return np.stack(
        [cos_roll * right + sin_roll * down, cos_roll * down - sin_roll * right, forward]
    )


def _perturb_quad(
    rng: np.random.Generator, base: np.ndarray, width: int, height: int
) -> np.ndarray:
    centroid = base.mean(axis=0)
    scale = float(rng.uniform(*_SCALE_RANGE))
    quad = centroid + (base - centroid) * scale
    xmin, ymin = quad.min(axis=0)
    xmax, ymax = quad.max(axis=0)
    span = xmax - xmin
    dx_low = _MIN_VISIBLE_SPAN * span - xmax
    dx_high = max(dx_low, float(width) - xmin - _MIN_VISIBLE_SPAN * span)
    dx = float(rng.uniform(dx_low, dx_high))
    dy_low = -ymin
    dy_high = float(height) - ymax
    dy = 0.5 * (dy_low + dy_high) if dy_high < dy_low else float(rng.uniform(dy_low, dy_high))
    quad = quad + np.array([dx, dy])
    jitter = rng.uniform(-_JITTER_FRACTION, _JITTER_FRACTION, size=(4, 2)) * np.array(
        [float(width), float(height)]
    )
    return np.asarray(quad + jitter, dtype=np.float64)


def _render_background(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    kind = int(rng.integers(0, 3))
    if kind == 0:
        return _bright_wood(rng, width, height)
    if kind == 1:
        return _dark_room(width, height)
    return _mid_clutter(rng, width, height)


def _bright_wood(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    grid_y = np.arange(height, dtype=np.float64)[:, None]
    grid_x = np.arange(width, dtype=np.float64)[None, :]
    base = 130.0 + 50.0 * grid_y / height + 25.0 * grid_x / width
    warm = np.stack([base * 0.95, base * 0.78, base * 0.55], axis=-1)
    canvas = warm.astype(np.float32)
    for _ in range(120):
        y = int(rng.integers(0, height))
        x0 = int(rng.integers(0, width))
        length = int(rng.integers(40, 400))
        amplitude = float(rng.uniform(-25.0, 25.0))
        canvas[y, x0 : min(width, x0 + length)] += amplitude
    return np.asarray(cv2.GaussianBlur(np.clip(canvas, 0.0, 255.0), (9, 3), 0), dtype=np.float32)


def _dark_room(width: int, height: int) -> np.ndarray:
    grid_y = np.arange(height, dtype=np.float64)[:, None]
    base = 15.0 + 35.0 * grid_y / height
    dim = np.stack([base * 0.85, base * 0.95, base * 1.1], axis=-1)
    return np.broadcast_to(dim, (height, width, 3)).astype(np.float32).copy()


def _mid_clutter(rng: np.random.Generator, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), float(rng.uniform(100.0, 150.0)), dtype=np.float32)
    for _ in range(int(rng.integers(6, 15))):
        x0, x1 = sorted((int(rng.integers(0, width)), int(rng.integers(0, width))))
        y0, y1 = sorted((int(rng.integers(0, height)), int(rng.integers(0, height))))
        canvas[y0 : y1 + 1, x0 : x1 + 1] = float(rng.uniform(50.0, 210.0))
    return canvas


def _draw_keybed(
    canvas: np.ndarray, quad: np.ndarray, rng: np.random.Generator, white_count: float
) -> None:
    sheet = _keybed_sheet(rng, white_count)
    sheet_h, sheet_w = sheet.shape[:2]
    src = np.array(
        [
            [0.0, 0.0],
            [float(sheet_w), 0.0],
            [float(sheet_w), float(sheet_h)],
            [0.0, float(sheet_h)],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    height, width = canvas.shape[:2]
    warped = cv2.warpPerspective(
        sheet,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [quad.astype(np.int32)], 255)
    inside = mask > 0
    canvas[inside] = warped[inside]
    sigma = float(rng.uniform(*_BLUR_SIGMA_RANGE))
    canvas[:] = cv2.GaussianBlur(canvas, (0, 0), sigma)


def _maybe_add_control_panel(
    canvas: np.ndarray, quad: np.ndarray, rng: np.random.Generator
) -> None:
    """Draw the dark button-filled body behind a keyboard without labelling it as keybed."""
    if rng.random() >= 0.7:
        return
    panel = np.array(
        [
            [-0.08, -0.95],
            [1.08, -0.95],
            [1.08, 0.04],
            [-0.08, 0.04],
        ],
        dtype=np.float32,
    )
    source = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    panel_quad = cv2.perspectiveTransform(
        panel[None, :, :], cv2.getPerspectiveTransform(source, quad.astype(np.float32))
    )[0]
    cv2.fillPoly(canvas, [panel_quad.astype(np.int32)], (22.0, 25.0, 28.0))
    for _ in range(int(rng.integers(18, 46))):
        u = float(rng.uniform(0.03, 0.97))
        v = float(rng.uniform(0.08, 0.88))
        button = np.array(
            [
                [u - 0.012, v - 0.025],
                [u + 0.012, v - 0.025],
                [u + 0.012, v + 0.025],
                [u - 0.012, v + 0.025],
            ],
            dtype=np.float32,
        )
        points = cv2.perspectiveTransform(
            button[None, :, :], cv2.getPerspectiveTransform(source, panel_quad.astype(np.float32))
        )[0]
        shade = float(rng.uniform(95.0, 190.0))
        cv2.fillPoly(canvas, [points.astype(np.int32)], (shade, shade, shade))


def _keybed_sheet(rng: np.random.Generator, white_count: float) -> np.ndarray:
    unit = _PIXELS_PER_UNIT
    sheet_w = round(white_count * unit)
    sheet_h = round(_STRIP_DEPTH * unit)
    bar_h = round(sheet_h * _BLACK_DEPTH_FRACTION)
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.float32)
    for key in range(int(white_count)):
        x0 = round(key * unit)
        x1 = round((key + 1) * unit)
        sheet[:, x0:x1] = float(rng.uniform(*_WHITE_BRIGHTNESS_RANGE))
        if key > 0:
            sheet[:, max(0, x0 - 1) : x0] = float(rng.uniform(*_KEY_SEAM_BRIGHTNESS_RANGE))
    # an 88 key board starts on A and a 61 key one on C, so the 2-3 grouping is not edge aligned
    phase = float(rng.integers(0, 7))
    black = float(rng.uniform(*_BLACK_BRIGHTNESS_RANGE))
    for octave in range(-1, int(white_count // 7) + 2):
        for offset in _BLACK_OFFSETS:
            u0 = 7.0 * octave + offset - phase
            u1 = u0 + _BLACK_WIDTH
            if u1 <= 0.0 or u0 >= white_count:
                continue
            x0 = max(0, round(u0 * unit))
            x1 = min(sheet_w, round(u1 * unit))
            if x1 > x0:
                sheet[:bar_h, x0:x1] = black
    return sheet


def _maybe_add_glare(canvas: np.ndarray, rng: np.random.Generator) -> None:
    if rng.random() >= _GLARE_PROBABILITY:
        return
    height, width = canvas.shape[:2]
    center = float(rng.uniform(0.0, float(width)))
    half = float(rng.uniform(0.05, 0.25) * width)
    xs = np.arange(width, dtype=np.float64)
    profile = np.clip(1.0 - np.abs(xs - center) / half, 0.0, 1.0)
    falloff = np.linspace(float(rng.uniform(0.2, 0.6)), 1.0, height)
    if rng.random() < 0.5:
        falloff = falloff[::-1].copy()
    amplitude = float(rng.uniform(*_GLARE_AMPLITUDE_RANGE))
    canvas += (amplitude * profile[None, :] * falloff[:, None])[:, :, None]


def _maybe_add_occluders(canvas: np.ndarray, quad: np.ndarray, rng: np.random.Generator) -> None:
    if rng.random() >= _OCCLUDER_PROBABILITY:
        return
    max_axis = (_OCCLUDER_MAX_AREA_FRACTION * _quad_area(quad) / np.pi) ** 0.5
    for _ in range(int(rng.integers(1, 4))):
        weights = rng.uniform(0.15, 1.0, size=4)
        center = (weights[:, None] * quad).sum(axis=0) / weights.sum()
        axes = (float(rng.uniform(0.1, 1.0) * max_axis), float(rng.uniform(0.1, 1.0) * max_axis))
        color = (
            float(rng.uniform(170.0, 220.0)),
            float(rng.uniform(120.0, 170.0)),
            float(rng.uniform(90.0, 140.0)),
        )
        cv2.ellipse(
            canvas,
            center=(round(float(center[0])), round(float(center[1]))),
            axes=(max(1, round(axes[0])), max(1, round(axes[1]))),
            angle=float(rng.uniform(0.0, 180.0)),
            startAngle=0.0,
            endAngle=360.0,
            color=color,
            thickness=-1,
        )


def _quad_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
