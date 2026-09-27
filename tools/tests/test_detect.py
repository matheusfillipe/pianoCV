import cv2
import numpy as np

from pianocv.detect import find_keybed, find_keybed_pattern
from pianocv.homography import find_homography

BLACK_OFFSETS = (0.60, 1.75, 3.60, 4.63, 5.66)
BLACK_WIDTH = 0.58
WHITE_COUNT = 52
KEYBED_WIDTH = 52.0
KEYBED_HEIGHT = 8.0
RENDER_PX_PER_UNIT = 16
CANVAS_SIZE = (640, 400)
TRUE_QUAD = np.array(
    [[55.0, 108.0], [595.0, 100.0], [600.0, 258.0], [50.0, 268.0]], dtype=np.float64
)
BACKGROUND_VALUE = 25.0
PATTERN_DEPTH = 6.38
PATTERN_MARGIN = 8.0
PATTERN_QUAD = np.array(
    [[55.0, 168.0], [595.0, 160.0], [595.0, 226.3], [55.0, 234.3]], dtype=np.float64
)
WOOD_BASE = (90, 90, 90)


def _render_sheet() -> np.ndarray:
    height = int(KEYBED_HEIGHT * RENDER_PX_PER_UNIT)
    width = int(KEYBED_WIDTH * RENDER_PX_PER_UNIT)
    sheet = np.full((height, width, 3), 235, dtype=np.uint8)
    bar_height = int(height * 0.6)
    for octave in range(8):
        for offset in BLACK_OFFSETS:
            u0 = 7 * octave + offset
            u1 = u0 + BLACK_WIDTH
            if u1 > KEYBED_WIDTH:
                continue
            x0 = round(u0 / KEYBED_WIDTH * width)
            x1 = round(u1 / KEYBED_WIDTH * width)
            sheet[0:bar_height, x0:x1] = 15
    return sheet


def _render_keybed(seed: int) -> np.ndarray:
    sheet = _render_sheet()
    src = np.array(
        [
            [0.0, 0.0],
            [float(sheet.shape[1]), 0.0],
            [float(sheet.shape[1]), float(sheet.shape[0])],
            [0.0, float(sheet.shape[0])],
        ],
        dtype=np.float32,
    )
    matrix = find_homography(src, TRUE_QUAD.astype(np.float32))
    canvas: np.ndarray = cv2.warpPerspective(
        sheet,
        matrix.astype(np.float32),
        CANVAS_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(BACKGROUND_VALUE,) * 3,
    )
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 6.0, canvas.shape)
    return np.asarray(np.clip(canvas.astype(np.float64) + noise, 0.0, 255.0), dtype=np.uint8)


def _corner_errors(detection_quad: np.ndarray, true_quad: np.ndarray) -> np.ndarray:
    return np.asarray(np.linalg.norm(detection_quad - true_quad, axis=1), dtype=np.float64)


def _render_pattern_sheet() -> np.ndarray:
    sheet_w = round((WHITE_COUNT + 2 * PATTERN_MARGIN) * RENDER_PX_PER_UNIT)
    sheet_h = round(PATTERN_DEPTH * RENDER_PX_PER_UNIT)
    sheet = np.full((sheet_h, sheet_w, 3), 235, dtype=np.uint8)
    bar_h = round(sheet_h * 0.6)
    starts = [5.66 - 5.0]
    for octave in range(7):
        for offset in BLACK_OFFSETS:
            starts.append(2.0 + 7.0 * octave + float(offset))
    for u0 in starts:
        if u0 + BLACK_WIDTH > WHITE_COUNT:
            continue
        x0 = round((u0 + PATTERN_MARGIN) / (WHITE_COUNT + 2 * PATTERN_MARGIN) * sheet_w)
        x1 = round(
            (u0 + BLACK_WIDTH + PATTERN_MARGIN) / (WHITE_COUNT + 2 * PATTERN_MARGIN) * sheet_w
        )
        sheet[0:bar_h, x0:x1] = 15
    return sheet


def _render_pattern_keybed(bg: tuple[int, int, int], seed: int, wood: bool) -> np.ndarray:
    sheet = _render_pattern_sheet()
    sheet_h = sheet.shape[0]
    sheet_w = sheet.shape[1]
    left = round(PATTERN_MARGIN / (WHITE_COUNT + 2 * PATTERN_MARGIN) * sheet_w)
    right = round((WHITE_COUNT + PATTERN_MARGIN) / (WHITE_COUNT + 2 * PATTERN_MARGIN) * sheet_w)
    src = np.array(
        [[left, 0.0], [right, 0.0], [right, float(sheet_h)], [left, float(sheet_h)]],
        dtype=np.float32,
    )
    matrix = find_homography(src, PATTERN_QUAD.astype(np.float32))
    canvas: np.ndarray = cv2.warpPerspective(
        sheet,
        matrix.astype(np.float32),
        CANVAS_SIZE,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=bg,
    )
    rng = np.random.default_rng(seed)
    if wood:
        grid_x, grid_y = np.meshgrid(np.arange(CANVAS_SIZE[0]), np.arange(CANVAS_SIZE[1]))
        base = 150.0 + 50.0 * grid_x / CANVAS_SIZE[0] + 25.0 * grid_y / CANVAS_SIZE[1]
        warm = np.stack([base * 0.55, base * 0.78, base], axis=-1)
        floor: np.ndarray = np.clip(warm + rng.normal(0.0, 8.0, warm.shape), 0.0, 255.0)
        for _ in range(80):
            y0 = int(rng.integers(0, CANVAS_SIZE[1]))
            amp = float(rng.uniform(-30.0, 30.0))
            x0 = int(rng.integers(0, CANVAS_SIZE[0]))
            length = int(rng.integers(60, 400))
            floor[y0, x0 : min(CANVAS_SIZE[0], x0 + length), :] += amp
        floor = cv2.GaussianBlur(np.clip(floor, 0.0, 255.0), (9, 3), 0)
        edge = np.all(canvas == np.array(bg, dtype=canvas.dtype), axis=-1)
        canvas[edge] = floor[edge].astype(np.uint8)
    noise = rng.normal(0.0, 6.0, canvas.shape)
    return np.asarray(np.clip(canvas.astype(np.float64) + noise, 0.0, 255.0), dtype=np.uint8)


def test_pattern_locks_on_dark_background() -> None:
    detection = find_keybed_pattern(_render_pattern_keybed((25, 25, 25), 7, wood=False))
    assert detection is not None
    assert detection.confidence > 0.35
    assert _corner_errors(detection.quad_px, PATTERN_QUAD).max() <= 15.0


def test_pattern_locks_on_bright_wood_floor() -> None:
    detection = find_keybed_pattern(_render_pattern_keybed(WOOD_BASE, 7, wood=True))
    assert detection is not None
    assert detection.confidence > 0.35
    assert _corner_errors(detection.quad_px, PATTERN_QUAD).max() <= 15.0


def test_pattern_occlusion_never_returns_wild_quad() -> None:
    image = _render_pattern_keybed((25, 25, 25), 11, wood=False)
    cv2.ellipse(
        image,
        center=(320, 193),
        axes=(80, 55),
        angle=10.0,
        startAngle=0.0,
        endAngle=360.0,
        color=(10, 10, 10),
        thickness=-1,
    )
    detection = find_keybed_pattern(image)
    if detection is None:
        return
    assert _corner_errors(detection.quad_px, PATTERN_QUAD).max() < 60.0


def test_pattern_returns_none_without_bar_structures() -> None:
    blank = np.full((480, 640, 3), 200, dtype=np.uint8)
    assert find_keybed_pattern(blank) is None
    assert find_keybed_pattern(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_locks_synthetic_keybed_within_8px() -> None:
    detection = find_keybed(_render_keybed(seed=7))
    assert detection is not None
    assert detection.confidence > 0.35
    assert _corner_errors(detection.quad_px, TRUE_QUAD).max() <= 8.0


def test_blank_and_noise_return_no_confident_detection() -> None:
    blank = np.full((480, 640, 3), 128, dtype=np.uint8)
    assert find_keybed(blank) is None
    rng = np.random.default_rng(3)
    noise = rng.integers(0, 256, size=(480, 640, 3)).astype(np.uint8)
    detection = find_keybed(noise)
    assert detection is None or detection.confidence <= 0.35


def test_occlusion_never_returns_wild_quad() -> None:
    image = _render_keybed(seed=11)
    cv2.ellipse(
        image,
        center=(320, 184),
        axes=(140, 60),
        angle=15.0,
        startAngle=0.0,
        endAngle=360.0,
        color=(10, 10, 10),
        thickness=-1,
    )
    detection = find_keybed(image)
    if detection is None:
        return
    assert _corner_errors(detection.quad_px, TRUE_QUAD).max() < 60.0


def test_empty_image_returns_none() -> None:
    assert find_keybed(np.zeros((0, 0, 3), dtype=np.uint8)) is None


def test_square_blob_fails_aspect_filter() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[140:340, 220:420] = 235
    assert find_keybed(image) is None


def test_hollow_frame_fails_fill_filter() -> None:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 150), (520, 330), 235, thickness=6)
    assert find_keybed(image) is None
