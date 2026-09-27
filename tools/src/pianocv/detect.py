"""Classical keybed quad detector: geometry only, zero training."""

from dataclasses import dataclass

import cv2
import numpy as np

from pianocv.dataset import canonical_quad, orient_quad

_TARGET_WIDTH = 320
_STRIP_SIZE = (416, 64)
_STRIP_DST = np.array([[0.0, 0.0], [415.0, 0.0], [415.0, 63.0], [0.0, 63.0]], dtype=np.float32)
_MIN_CONFIDENCE = 0.35
_MIN_AREA_FRACTION = 0.005
_MAX_AREA_FRACTION = 0.4
_MIN_ASPECT = 2.2
_MIN_FILL = 0.5
_MAX_CANDIDATES = 5
_WHITE_ZONE = slice(4, 38)
_PROFILE_ZONE = slice(0, 40)
_WHITENESS_MIN = 100.0
_MIN_PEAKS = 14
_MAX_PEAKS = 40
_SPACING_REGULARITY_MAX = 0.6
_MIN_PEAK_WIDTH = 2.0
_MAX_PEAK_WIDTH = 12.0
_REFINE_STEP = 2.0
_REFINE_FRACTION = 0.15
_EDGE_BAND = 2
_PATTERN_WIDTH = 480
_PATTERN_BLUR_LENGTH = 41
_PATTERN_MARGIN = 12.0
_PATTERN_OPEN = np.ones((2, 2), dtype=np.uint8)
_PATTERN_MIN_AREA = 4
_PATTERN_MAX_AREA = 800
_PATTERN_MIN_ASPECT = 0.10
_PATTERN_MAX_ASPECT = 6.0
_PATTERN_MIN_FILL = 0.4
_RANSAC_DISTANCE = 3.0
_RANSAC_ITERATIONS = 2000
_RANSAC_MIN_INLIERS = 10
_RANSAC_MIN_BASELINE = 10.0
_RANSAC_SEED = 11
_OCTAVE_MIN = 21.0
_OCTAVE_MAX = 84.0
_SLOT_TOLERANCE = 1.5
_MIN_PATTERN_SCORE = 0.6
_MIN_PATTERN_HITS = 10
_MIN_HITS_PER_OCTAVE = 3.0
_BLACK_OFFSETS = np.array([0.60, 1.75, 3.60, 4.63, 5.66])
_BAR_CENTER_BIAS = 0.29
_WHITE_COUNT = 52.0
_C_OFFSET = 2.0
_STRIP_DEPTH_UNITS = 6.38
_BACK_FRACTION = 0.3
_FRONT_FRACTION = 0.7
_EXTENT_LOW = 0.5
_EXTENT_HIGH = 1.3
_ANCHOR_MARGIN = 0.5
_VISIBILITY_FREE = 0.3


@dataclass
class Detection:
    quad_px: np.ndarray
    confidence: float


def find_keybed(image_bgr: np.ndarray) -> Detection | None:
    height, width = image_bgr.shape[:2]
    if height == 0 or width == 0:
        return None
    scale = _TARGET_WIDTH / width
    small_height = max(1, round(height * scale))
    small = cv2.resize(image_bgr, (_TARGET_WIDTH, small_height), interpolation=cv2.INTER_AREA)
    mask = _bright_mask(small)
    best: Detection | None = None
    for _, quad in _candidate_quads(mask, scale, small_height):
        detection = _verify(image_bgr, quad)
        if detection is not None and (best is None or detection.confidence > best.confidence):
            best = detection
    if best is None or best.confidence <= _MIN_CONFIDENCE:
        return None
    return _refine(image_bgr, best)


def _bright_mask(small: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)


def _candidate_quads(
    mask: np.ndarray, scale: float, small_height: int
) -> list[tuple[float, np.ndarray]]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total = float(_TARGET_WIDTH * small_height)
    scored: list[tuple[float, np.ndarray]] = []
    for index in range(1, int(count)):
        area = float(stats[index, cv2.CC_STAT_AREA])
        box_width = float(stats[index, cv2.CC_STAT_WIDTH])
        box_height = float(stats[index, cv2.CC_STAT_HEIGHT])
        fraction = area / total
        if not _MIN_AREA_FRACTION <= fraction <= _MAX_AREA_FRACTION:
            continue
        if min(box_width, box_height) <= 0.0:
            continue
        aspect = max(box_width, box_height) / min(box_width, box_height)
        if aspect < _MIN_ASPECT:
            continue
        fill = area / (box_width * box_height)
        if fill < _MIN_FILL:
            continue
        points = np.column_stack(np.nonzero(labels == index)).astype(np.float32)
        points = np.ascontiguousarray(points[:, ::-1])
        box = np.asarray(cv2.boxPoints(cv2.minAreaRect(points)), dtype=np.float32)
        scored.append((area * fill, _order_quad(box) / scale))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:_MAX_CANDIDATES]


def _order_quad(box: np.ndarray) -> np.ndarray:
    total = box.sum(axis=1)
    diagonal = box[:, 1] - box[:, 0]
    # the strip warp reads the keys along edge 0->1, so a vertical keybed has to be rolled first
    return canonical_quad(
        np.array(
            [
                box[int(np.argmin(total))],
                box[int(np.argmin(diagonal))],
                box[int(np.argmax(total))],
                box[int(np.argmax(diagonal))],
            ],
            dtype=np.float64,
        )
    )


def _verify(image_bgr: np.ndarray, quad_px: np.ndarray) -> Detection | None:
    quad_px = orient_quad(image_bgr, quad_px)
    strip = cv2.warpPerspective(
        image_bgr,
        cv2.getPerspectiveTransform(quad_px.astype(np.float32), _STRIP_DST),
        _STRIP_SIZE,
    )
    confidence = _strip_confidence(strip)
    if confidence is None:
        return None
    return Detection(quad_px=quad_px.astype(np.float64), confidence=confidence)


def _strip_confidence(strip_bgr: np.ndarray) -> float | None:
    gray = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    whiteness = float(gray[_WHITE_ZONE, :].mean())
    if whiteness <= _WHITENESS_MIN:
        return None
    smoothed = _smooth(gray[_PROFILE_ZONE, :].mean(axis=0))
    threshold = float(smoothed.mean() - 0.25 * smoothed.std())
    peaks = _dark_peaks(smoothed, threshold)
    widths = [_peak_width(smoothed, threshold, peak) for peak in peaks]
    earned = 0
    if _MIN_PEAKS <= len(peaks) <= _MAX_PEAKS:
        earned += 1
    if len(peaks) >= 3:
        spacings = np.diff(np.asarray(peaks, dtype=np.float64))
        mean_spacing = float(spacings.mean())
        if mean_spacing > 0.0 and float(spacings.std()) / mean_spacing < _SPACING_REGULARITY_MAX:
            earned += 1
    median_width = float(np.median(widths)) if widths else 0.0
    if _MIN_PEAK_WIDTH <= median_width <= _MAX_PEAK_WIDTH:
        earned += 1
    margin = min(1.0, (whiteness - _WHITENESS_MIN) / 100.0)
    confidence = (earned / 3.0) * 0.5 + margin * 0.5
    return float(min(1.0, max(0.0, confidence)))


def _smooth(profile: np.ndarray) -> np.ndarray:
    kernel = np.full(3, 1.0 / 3.0)
    return np.convolve(np.pad(profile, 1, mode="edge"), kernel, mode="valid")


def _dark_peaks(profile: np.ndarray, threshold: float) -> list[int]:
    peaks: list[int] = []
    for i in range(1, len(profile) - 1):
        if profile[i] < profile[i - 1] and profile[i] <= profile[i + 1] and profile[i] < threshold:
            peaks.append(i)
    return peaks


def _peak_width(profile: np.ndarray, threshold: float, peak: int) -> float:
    left = peak
    while left > 0 and profile[left - 1] < threshold:
        left -= 1
    right = peak
    while right < len(profile) - 1 and profile[right + 1] < threshold:
        right += 1
    return float(right - left + 1)


def _refine(image_bgr: np.ndarray, detection: Detection) -> Detection:
    quad = detection.quad_px
    left = _walk_edge(image_bgr, quad, side=0)
    right = _walk_edge(image_bgr, quad, side=1)
    result = _verify(image_bgr, _shift_edges(quad, left, right))
    if result is not None and result.confidence >= detection.confidence:
        return result
    return detection


def _walk_edge(image_bgr: np.ndarray, quad: np.ndarray, side: int) -> float:
    length = float(np.linalg.norm(quad[1] - quad[0]))
    limit = length * _REFINE_FRACTION
    extension = 0.0
    while extension + _REFINE_STEP <= limit and _edge_bright(
        image_bgr, quad, side, extension + _REFINE_STEP
    ):
        extension += _REFINE_STEP
    if not _edge_bright(image_bgr, quad, side, extension):
        while extension >= -limit and not _edge_bright(
            image_bgr, quad, side, extension - _REFINE_STEP
        ):
            extension -= _REFINE_STEP
    return extension


def _edge_bright(image_bgr: np.ndarray, quad: np.ndarray, side: int, extension: float) -> bool:
    shifted = _shift_edges(quad, extension if side == 0 else 0.0, extension if side == 1 else 0.0)
    strip = cv2.warpPerspective(
        image_bgr,
        cv2.getPerspectiveTransform(shifted.astype(np.float32), _STRIP_DST),
        _STRIP_SIZE,
    )
    gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float64)
    columns = gray[_WHITE_ZONE, :].mean(axis=0)
    band = columns[:_EDGE_BAND] if side == 0 else columns[-_EDGE_BAND:]
    return float(band.mean()) > _WHITENESS_MIN


def _shift_edges(quad: np.ndarray, left: float, right: float) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = quad
    top_length = float(np.linalg.norm(top_right - top_left))
    if top_length == 0.0:
        return quad.copy()
    unit_top = (top_right - top_left) / top_length
    bottom_length = float(np.linalg.norm(bottom_right - bottom_left))
    unit_bottom = (bottom_right - bottom_left) / bottom_length if bottom_length > 0.0 else unit_top
    return np.array(
        [
            top_left - unit_top * left,
            top_right + unit_top * right,
            bottom_right + unit_bottom * right,
            bottom_left - unit_bottom * left,
        ],
        dtype=np.float64,
    )


def _pattern_boxes(small: np.ndarray, ksize: tuple[int, int]) -> np.ndarray:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    blur = cv2.blur(gray, ksize)
    mask: np.ndarray = (gray < blur - _PATTERN_MARGIN).astype(np.uint8)
    mask *= 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, _PATTERN_OPEN)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, int(count)):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < _PATTERN_MIN_AREA or area > _PATTERN_MAX_AREA:
            continue
        box_w = float(stats[i, cv2.CC_STAT_WIDTH])
        box_h = float(stats[i, cv2.CC_STAT_HEIGHT])
        if box_w <= 0.0 or box_h <= 0.0:
            continue
        aspect = box_w / box_h
        fill = area / (box_w * box_h)
        if (
            aspect < _PATTERN_MIN_ASPECT
            or aspect > _PATTERN_MAX_ASPECT
            or fill <= _PATTERN_MIN_FILL
        ):
            continue
        center = centroids[i]
        boxes.append((float(center[0]), float(center[1]), box_w, box_h))
    if not boxes:
        return np.zeros((0, 4), dtype=np.float64)
    return np.array(boxes, dtype=np.float64)


def _dominant_line(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    points = boxes[:, :2]
    rng = np.random.default_rng(_RANSAC_SEED)
    best_count = 0
    best: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(_RANSAC_ITERATIONS):
        i, j = rng.choice(len(points), 2, replace=False)
        direction = points[j] - points[i]
        norm = float(np.linalg.norm(direction))
        if norm < _RANSAC_MIN_BASELINE:
            continue
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]])
        distances = np.abs((points - points[i]) @ normal)
        count = int((distances < _RANSAC_DISTANCE).sum())
        if count > best_count:
            best_count = count
            best = (points[i].copy(), direction.copy())
    if best is None or best_count < _RANSAC_MIN_INLIERS:
        return None
    point, direction = best
    normal = np.array([-direction[1], direction[0]])
    inliers = points[np.abs((points - point) @ normal) < _RANSAC_DISTANCE]
    mean = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - mean)
    axis = vt[0]
    normal = np.array([-axis[1], axis[0]])
    idx = np.flatnonzero(np.abs((points - mean) @ normal) < _RANSAC_DISTANCE)
    if len(idx) < _RANSAC_MIN_INLIERS:
        return None
    return mean, axis, idx


def _slot_score(
    residuals: np.ndarray, period: float, phase: float, slots: np.ndarray
) -> tuple[int, float]:
    offset = np.mod(residuals - phase, period)[:, None]
    diff = np.abs(np.mod(offset - slots[None, :], period))
    distance = np.minimum(diff, period - diff).min(axis=1)
    hits = distance < _SLOT_TOLERANCE
    return int(hits.sum()), float(distance[hits].sum())


def _fit_pattern(s: np.ndarray) -> tuple[float, float, float, int, float] | None:
    s_ref = float(s.min())
    best: tuple[int, float, float, float, float] | None = None
    for period in np.arange(_OCTAVE_MIN, _OCTAVE_MAX + 1e-9, 0.5):
        for sign in (1.0, -1.0):
            residuals = np.mod(sign * (s - s_ref), period)
            slots = np.mod(_BLACK_OFFSETS * period / 7.0, period)
            phases = np.mod(residuals[:, None] - slots[None, :], period).ravel()
            for phase in phases:
                hits, spread = _slot_score(residuals, period, phase, slots)
                candidate = (hits, -spread, period, sign, phase)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
    if best is None:
        return None
    hits, neg_spread, period, sign, phase = best
    for period_now in np.arange(
        max(_OCTAVE_MIN, period - 0.5), min(_OCTAVE_MAX, period + 0.5) + 1e-9, 0.05
    ):
        residuals = np.mod(sign * (s - s_ref), period_now)
        slots = np.mod(_BLACK_OFFSETS * period_now / 7.0, period_now)
        phases = np.mod(residuals[:, None] - slots[None, :], period_now).ravel()
        for phase_now in phases:
            now, spread = _slot_score(residuals, period_now, phase_now, slots)
            if now > hits or (now == hits and spread < -neg_spread):
                hits, neg_spread, period, phase = now, -spread, period_now, phase_now
    score = hits / len(s)
    if score <= _MIN_PATTERN_SCORE or hits < _MIN_PATTERN_HITS:
        return None
    return period, sign, phase, hits, score


def _solve_pattern(
    values: np.ndarray, period: float, sign: float, phase: float, s_ref: float
) -> tuple[float, float, np.ndarray, np.ndarray, float] | None:
    w = period / 7.0
    signed = sign * (values - s_ref)
    anchor = phase
    j = np.zeros(len(values), dtype=int)
    k = np.zeros(len(values), dtype=float)
    for _ in range(3):
        e = np.mod(signed - anchor, period)
        slots = np.mod(_BLACK_OFFSETS * w, period)
        diff = np.abs(np.mod(e[:, None] - slots[None, :], period))
        j = np.argmin(np.minimum(diff, period - diff), axis=1)
        k = np.round(((signed - anchor) / w - _BLACK_OFFSETS[j]) / 7.0)
        design = np.column_stack([np.ones(len(values)), _BLACK_OFFSETS[j] + 7.0 * k])
        solution, *_ = np.linalg.lstsq(design, signed, rcond=None)
        if not np.isfinite(solution).all():
            return None
        anchor, w = float(solution[0]), float(solution[1])
        if not _OCTAVE_MIN <= w * 7.0 <= _OCTAVE_MAX:
            return None
    e = np.mod(signed - anchor, period)
    slots = np.mod(_BLACK_OFFSETS * w, period)
    diff = np.abs(np.mod(e[:, None] - slots[None, :], period))
    j = np.argmin(np.minimum(diff, period - diff), axis=1)
    k = np.round(((signed - anchor) / w - _BLACK_OFFSETS[j]) / 7.0)
    residual_sum = float(np.abs(signed - (anchor + w * (_BLACK_OFFSETS[j] + 7.0 * k))).sum())
    return anchor, w, j, k, residual_sum


def _anchor_shift(u: np.ndarray) -> int | None:
    best_count = 0
    best: int | None = None
    for m in range(-8, 9):
        shifted = u + 7.0 * m
        count = int(
            ((shifted >= -_ANCHOR_MARGIN) & (shifted <= _WHITE_COUNT + _ANCHOR_MARGIN)).sum()
        )
        if count > best_count:
            best_count = count
            best = m
    return best


def _front_side_sign(
    small: np.ndarray,
    point: np.ndarray,
    axis: np.ndarray,
    normal: np.ndarray,
    depth_px: float,
    inlier_pts: np.ndarray,
) -> float:
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    projected = (inlier_pts - point) @ axis
    samples = np.linspace(float(projected.min()), float(projected.max()), 60)
    scores = []
    for sigma in (1.0, -1.0):
        total = 0.0
        count = 0
        for t in samples:
            for f in (0.7, 0.8, 0.9):
                q = point + axis * t + sigma * normal * f * depth_px
                x, y = round(float(q[0])), round(float(q[1]))
                if 0 <= x < width and 0 <= y < height:
                    total += float(gray[y, x])
                    count += 1
        scores.append(total / max(1, count))
    return 1.0 if scores[0] >= scores[1] else -1.0


def _order_pattern_quad(points: np.ndarray) -> np.ndarray:
    total = points.sum(axis=1)
    diagonal = points[:, 1] - points[:, 0]
    return canonical_quad(
        np.array(
            [
                points[int(np.argmin(total))],
                points[int(np.argmin(diagonal))],
                points[int(np.argmax(total))],
                points[int(np.argmax(diagonal))],
            ],
            dtype=np.float64,
        )
    )


def find_keybed_pattern(image_bgr: np.ndarray) -> Detection | None:
    height, width = image_bgr.shape[:2]
    if height == 0 or width == 0:
        return None
    scale = _PATTERN_WIDTH / width
    small = cv2.resize(
        image_bgr, (_PATTERN_WIDTH, max(1, round(height * scale))), interpolation=cv2.INTER_AREA
    )
    best: Detection | None = None
    for ksize in ((_PATTERN_BLUR_LENGTH, 1), (1, _PATTERN_BLUR_LENGTH)):
        detection = _find_pattern_pass(image_bgr, small, scale, ksize)
        if detection is not None and (best is None or detection.confidence > best.confidence):
            best = detection
    return best


def _find_pattern_pass(
    image_bgr: np.ndarray, small: np.ndarray, scale: float, ksize: tuple[int, int]
) -> Detection | None:
    boxes = _pattern_boxes(small, ksize)
    if len(boxes) < _RANSAC_MIN_INLIERS:
        return None
    line = _dominant_line(boxes)
    if line is None:
        return None
    point, axis, idx = line
    inliers = boxes[idx]
    normal = np.array([-axis[1], axis[0]])
    extent = np.abs(normal[0]) * inliers[:, 2] + np.abs(normal[1]) * inliers[:, 3]
    median = float(np.median(extent))
    inliers = inliers[(extent >= _EXTENT_LOW * median) & (extent <= _EXTENT_HIGH * median)]
    if len(inliers) < _RANSAC_MIN_INLIERS:
        return None
    s = (inliers[:, :2] - point) @ axis
    fit = _fit_pattern(s)
    if fit is None:
        return None
    period, sign, phase, hits, score = fit
    solves = []
    for flipped in (False, True):
        values = -s if flipped else s
        s_ref = float(values.min())
        solved = _solve_pattern(values, period, sign, phase, s_ref)
        if solved is not None:
            solves.append((solved[4], flipped, values, s_ref, solved))
    if not solves:
        return None
    solves.sort(key=lambda item: item[0])
    _, mirrored, values, s_ref, solved = solves[0]
    anchor, w, j, k = solved[:4]
    if hits / max(1, len(np.unique(k))) < _MIN_HITS_PER_OCTAVE:
        return None
    u = _C_OFFSET + _BAR_CENTER_BIAS + _BLACK_OFFSETS[j] + 7.0 * k
    m = _anchor_shift(u)
    if m is None:
        return None
    u = u + 7.0 * m
    in_range = (u >= -_ANCHOR_MARGIN) & (u <= _WHITE_COUNT + _ANCHOR_MARGIN)
    if int(in_range.sum()) < _MIN_PATTERN_HITS:
        return None
    if int(in_range.sum()) < len(values):
        values = values[in_range]
        resolved = _solve_pattern(values, period, sign, phase, s_ref)
        if resolved is None:
            return None
        anchor, w, j, k = resolved[:4]
        u = _C_OFFSET + _BAR_CENTER_BIAS + _BLACK_OFFSETS[j] + 7.0 * k
        m = _anchor_shift(u)
        if m is None:
            return None
        u = u + 7.0 * m
        if u.min() < -_ANCHOR_MARGIN or u.max() > _WHITE_COUNT + _ANCHOR_MARGIN:
            return None
    inlier_pts = inliers[in_range, :2]
    depth_px = _STRIP_DEPTH_UNITS * w
    sigma = _front_side_sign(small, point, axis, normal, depth_px, inlier_pts)
    t_back = s_ref + sign * (anchor + w * (0.0 - _C_OFFSET - _BAR_CENTER_BIAS - 7.0 * m))
    t_front = s_ref + sign * (anchor + w * (_WHITE_COUNT - _C_OFFSET - _BAR_CENTER_BIAS - 7.0 * m))
    if mirrored:
        t_back = -t_back
        t_front = -t_front
    p_back_a = point + axis * t_back - sigma * normal * _BACK_FRACTION * depth_px
    p_front_a = point + axis * t_back + sigma * normal * _FRONT_FRACTION * depth_px
    p_front_b = point + axis * t_front + sigma * normal * _FRONT_FRACTION * depth_px
    p_back_b = point + axis * t_front - sigma * normal * _BACK_FRACTION * depth_px
    quad = _order_pattern_quad(np.array([p_back_a, p_front_a, p_front_b, p_back_b]))
    quad = quad / scale
    edge = np.linspace(quad[0], quad[3], 100)
    full_h, full_w = image_bgr.shape[:2]
    inside = (edge[:, 0] >= 0) & (edge[:, 0] < full_w) & (edge[:, 1] >= 0) & (edge[:, 1] < full_h)
    frac_out = 1.0 - float(inside.mean())
    visibility = min(1.0, max(0.0, (1.0 - frac_out) / (1.0 - _VISIBILITY_FREE)))
    confidence = score * min(1.0, hits / 20.0) * visibility
    return Detection(quad_px=quad, confidence=float(confidence))
