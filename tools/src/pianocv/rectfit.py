"""Fit the keybed as what it is: a known rectangle seen through a pinhole camera.

A free quad has eight degrees of freedom and nothing stops its far end from wandering. The
keybed rectangle under a rotation, a translation and a focal length has six, and every quad
that shape can produce is a legal keybed. The six are fitted to the mask's whole boundary at
once, so the far end is placed by the outline plus the known proportions, never by the few
pixels at the thin end.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from pianocv.dataset import canonical_quad
from pianocv.fitquad import _largest_contour
from pianocv.pose import DEPTH_UNITS, WHITE_KEY_COUNT
from pianocv.refine_edges import _BLUR_SIGMA, _MIN_RESPONSE, _SEARCH_STEP, SEARCH_PX, _sample

_ITERATIONS = 40
_ASSIGNMENT_ROUNDS = 3
_HUBER_PX = 3.0
_MAX_POINTS = 400
_BORDER_PX = 2
_STEP = 1e-4
# closer than one key width to the camera the projection is meaningless
_MIN_DEPTH_UNITS = 1.0
# one mask cell on a 640 px frame: a fit the boundary sits further from than that is not
# explaining it
_MAX_RESIDUAL_PX = 4.5
_MIN_EDGE_POINTS = 3
_MIN_EDGE_VOTE = 24.0
# one end may be drawn this much wider than the other, the same as their distance ratio; the
# hand-labelled views span 1.54 to 2.65 and a sliding rectangle scores 4.0 and up
_MAX_END_RATIO = 3.2
# an end is re-drawn where the picture shows the keys stopping
_END_SAMPLES = 48
_END_TRIM = 0.08
_END_SEARCH_PX = 25.0
_MIN_END_SAMPLES = 10
_MAX_END_SHIFT_PX = 30.0
_END_CONTRAST_PX = 4.0
_MIN_KEY_BRIGHTNESS = 110.0
_MIN_END_CONTRAST = 30.0
_MIN_END_ALIGNMENT = 0.9
_MIN_END_SPAN = 0.5
# focal lengths as fractions of the frame width, from a wide phone lens to a long zoom
_FOCAL_SCAN = (0.5, 0.75, 1.1, 1.6, 2.4, 3.5)
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
_GOLDEN_ITERATIONS = 8


@dataclass(frozen=True)
class RectFit:
    quad_px: np.ndarray
    cost: float
    focal: float


def _world(white_keys: float, depth_units: float) -> np.ndarray:
    """The keybed rectangle in white-key widths, corner 0 at the origin, span along x."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [white_keys, 0.0, 0.0],
            [white_keys, depth_units, 0.0],
            [0.0, depth_units, 0.0],
        ]
    )


def project(
    params: np.ndarray, focal: float, cx: float, cy: float, world: np.ndarray
) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(params[:3])
    camera = world @ rotation.T + params[3:]
    z = np.maximum(camera[:, 2], 1e-6)
    return np.column_stack([cx + focal * camera[:, 0] / z, cy + focal * camera[:, 1] / z])


def _in_front(params: np.ndarray, world: np.ndarray) -> bool:
    rotation, _ = cv2.Rodrigues(params[:3])
    depths = (world @ rotation.T + params[3:])[:, 2]
    return bool(np.all(depths > _MIN_DEPTH_UNITS))


def _clockwise(quad: np.ndarray) -> np.ndarray:
    # the world corners project clockwise on screen under any rotation; a quad wound the
    # other way is the mirror image, which no pose reaches, so we reflect it along the span
    x, y = quad[:, 0], quad[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        return np.asarray(quad[[1, 0, 3, 2]])
    return quad


def _initial_pose(
    quad_px: np.ndarray, focal: float, cx: float, cy: float, world: np.ndarray
) -> np.ndarray:
    """Rotation and translation that put the rectangle on the quad, from planar PnP (IPPE)."""
    camera_matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    ordered = _clockwise(canonical_quad(quad_px)).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(world, ordered, camera_matrix, None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise ValueError("planar pose could not be initialised from the coarse quad")
    return np.concatenate([np.asarray(rvec).ravel(), np.asarray(tvec).ravel()])


def _segment_distances(points: np.ndarray, quad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance from every point to the nearest of the four edges, and which edge that is."""
    best = np.full(len(points), np.inf)
    which = np.zeros(len(points), dtype=np.int64)
    for i in range(4):
        a = quad[i]
        d = quad[(i + 1) % 4] - a
        length_sq = float(np.dot(d, d))
        if length_sq < 1e-9:
            continue
        t = np.clip(((points - a) @ d) / length_sq, 0.0, 1.0)
        foot = a + t[:, None] * d
        distance = np.linalg.norm(points - foot, axis=1)
        closer = distance < best
        best = np.where(closer, distance, best)
        which = np.where(closer, i, which)
    return best, which


def _assigned_distances(points: np.ndarray, quad: np.ndarray, which: np.ndarray) -> np.ndarray:
    """Distance from every point to the edge it is assigned to."""
    a = quad[which]
    d = quad[(which + 1) % 4] - a
    length_sq = np.maximum(np.einsum("ij,ij->i", d, d), 1e-9)
    t = np.clip(np.einsum("ij,ij->i", points - a, d) / length_sq, 0.0, 1.0)
    return np.asarray(np.linalg.norm(points - (a + t[:, None] * d), axis=1), dtype=np.float64)


def _edge_weights(which: np.ndarray) -> np.ndarray:
    # each edge is one line measurement however many boundary points lie on it; unweighted,
    # the hundreds of points on the long edges outvote the dozen on the thin far end
    counts = np.bincount(which, minlength=4).astype(np.float64)
    # an edge with few points gets a proportionally smaller vote: un-normalized, four far-end
    # crossings swung the rigid rectangle 3-5 px per frame on sub-pixel noise
    per_edge = np.sqrt(len(which) / (4.0 * np.maximum(counts, _MIN_EDGE_VOTE)))
    return np.asarray(per_edge[which], dtype=np.float64)


def _huber(distances: np.ndarray) -> np.ndarray:
    small = distances <= _HUBER_PX
    beyond = np.sqrt(np.maximum(2.0 * _HUBER_PX * distances - _HUBER_PX**2, 0.0))
    return np.where(small, distances, beyond)


def boundary_points(probability: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Boundary of the keybed in frame pixels: the half-probability line of the mask
    upsampled to the frame, so the edge is placed between cells, never on one."""
    field = cv2.resize(
        probability.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    contour = _largest_contour((field > 0.5).astype(np.uint8))
    if contour is None:
        return None
    # where the keybed leaves the frame the contour runs along the frame border, which says
    # nothing about the keybed, so those points are dropped
    inside = (
        (contour[:, 0] > _BORDER_PX)
        & (contour[:, 0] < width - 1 - _BORDER_PX)
        & (contour[:, 1] > _BORDER_PX)
        & (contour[:, 1] < height - 1 - _BORDER_PX)
    )
    contour = contour[inside]
    if len(contour) < 8:
        return None
    if len(contour) > _MAX_POINTS:
        contour = contour[:: int(np.ceil(len(contour) / _MAX_POINTS))]
    return np.asarray(contour, dtype=np.float64)


def _fit_at_focal(
    points: np.ndarray,
    coarse: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    world: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Levenberg-Marquardt over (rotation, translation) with the focal held fixed."""
    params = _initial_pose(coarse, focal, cx, cy, world)
    if not _in_front(params, world):
        return params, float("inf")

    # assignment is redone once per solve, not continuously: every step would let the
    # rectangle slide onto a wrong, lower-cost fit, while never redoing it locks in a bad start
    _, which = _segment_distances(points, _assignment_box(points, coarse))
    cost = float("inf")
    for _ in range(_ASSIGNMENT_ROUNDS):
        params, cost = _solve_assigned(points, which, params, focal, cx, cy, world)
        _, reassigned = _segment_distances(points, project(params, focal, cx, cy, world))
        if np.array_equal(reassigned, which):
            break
        which = reassigned
    return params, cost


def _solve_assigned(
    points: np.ndarray,
    which: np.ndarray,
    params: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    world: np.ndarray,
) -> tuple[np.ndarray, float]:
    weights = _edge_weights(which)

    def residuals(p: np.ndarray) -> np.ndarray:
        distances = _assigned_distances(points, project(p, focal, cx, cy, world), which)
        return np.asarray(_huber(distances) * weights, dtype=np.float64)

    current = residuals(params)
    cost = float(np.sum(current**2))
    damping = 1e-2
    for _ in range(_ITERATIONS):
        jacobian = np.empty((len(current), 6))
        for k in range(6):
            bumped = params.copy()
            bumped[k] += _STEP
            jacobian[:, k] = (residuals(bumped) - current) / _STEP
        gram = jacobian.T @ jacobian
        gradient = jacobian.T @ current
        step = np.linalg.solve(gram + damping * np.diag(np.diag(gram) + 1e-9), -gradient)
        candidate = params + step
        # a step that swings the plane through the camera is a fold of the cost, never a fit
        trial, trial_cost = current, float("inf")
        if _in_front(candidate, world):
            trial = residuals(candidate)
            trial_cost = float(np.sum(trial**2))
        if trial_cost < cost:
            params, current, cost = candidate, trial, trial_cost
            damping = max(damping / 3.0, 1e-6)
            if np.linalg.norm(step) < 1e-5:
                break
        else:
            damping = min(damping * 5.0, 1e6)
    return params, cost


def _scan_focal(
    points: np.ndarray,
    coarse: np.ndarray,
    width: int,
    cx: float,
    cy: float,
    world: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Focal, pose and cost at the boundary-cost minimum over focal: a scan, then golden
    section on the logarithm around the best sample."""
    tried: dict[float, tuple[np.ndarray, float]] = {}

    def at(log_focal: float) -> float:
        if log_focal not in tried:
            focal = float(np.exp(log_focal))
            tried[log_focal] = _fit_at_focal(points, coarse, focal, cx, cy, world)
        return tried[log_focal][1]

    scan = np.log(np.array(_FOCAL_SCAN) * width)
    costs = [at(float(f)) for f in scan]
    best = int(np.argmin(costs))
    a = float(scan[max(best - 1, 0)])
    b = float(scan[min(best + 1, len(scan) - 1)])
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    for _ in range(_GOLDEN_ITERATIONS):
        if at(c) < at(d):
            b, d = d, c
            c = b - _GOLDEN * (b - a)
        else:
            a, c = c, d
            d = a + _GOLDEN * (b - a)
    log_focal, (params, cost) = min(tried.items(), key=lambda item: item[1][1])
    return float(np.exp(log_focal)), params, cost


def snap_to_gradient(gray: np.ndarray, points: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Each boundary point moved along its edge's normal onto the strongest brightness step
    nearby, and dropped when there is none. On real frames the mask boundary sits a few pixels
    off the keybed's edge and a fit that trusts it inherits the bias; the image gradient is
    where the edge really is, the same evidence the coarse quad's edge refinement uses."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), _BLUR_SIGMA)
    _, which = _segment_distances(points, quad)
    edges = quad[(np.arange(4) + 1) % 4] - quad
    lengths = np.linalg.norm(edges, axis=1)
    normals = np.column_stack([-edges[:, 1], edges[:, 0]]) / np.maximum(lengths, 1e-9)[:, None]
    mids = (quad + quad[(np.arange(4) + 1) % 4]) / 2
    facing = np.sign(np.einsum("ij,ij->i", normals, mids - quad.mean(axis=0)))
    outward = normals * np.where(facing == 0, 1.0, facing)[:, None]
    offsets = np.arange(-SEARCH_PX, SEARCH_PX + 1e-6, _SEARCH_STEP)
    samples = points[:, None, :] + offsets[None, :, None] * outward[which][:, None, :]
    values = _sample(blurred, samples[:, :, 0].ravel(), samples[:, :, 1].ravel())
    values = values.reshape(len(points), len(offsets))
    # the edge is a step from key-bright to dark, not the strongest gradient, which instead
    # follows the case's lip beyond it; with no key-bright reach the gradient decides anyway
    reach = round(_END_CONTRAST_PX / _SEARCH_STEP)
    contrast = np.full_like(values, -np.inf)
    for k in range(reach, len(offsets) - reach):
        inside = values[:, k - reach : k].mean(axis=1)
        contrast[:, k] = np.where(
            inside >= _MIN_KEY_BRIGHTNESS,
            inside - values[:, k + 1 : k + 1 + reach].mean(axis=1),
            -np.inf,
        )
    by_contrast = np.argmax(contrast, axis=1)
    keyed = contrast[np.arange(len(points)), by_contrast] > _MIN_END_CONTRAST
    # otherwise the nearest real step to the boundary, not the strongest: the strongest in
    # reach was a blinking panel light beside the black keys, and the edge followed it
    gradient = np.abs(np.gradient(values, _SEARCH_STEP, axis=1))
    padded = np.pad(gradient, ((0, 0), (1, 1)), constant_values=0.0)
    local_max = (gradient >= padded[:, :-2]) & (gradient >= padded[:, 2:])
    candidate = np.where(local_max & (gradient > _MIN_RESPONSE), np.abs(offsets)[None, :], np.inf)
    by_gradient = np.argmin(candidate, axis=1)
    stepped = np.isfinite(candidate[np.arange(len(points)), by_gradient])
    best = np.where(keyed, by_contrast, by_gradient)
    keep = keyed | stepped
    return np.asarray(points[keep] + offsets[best[keep]][:, None] * outward[which[keep]])


def _principal_box(points: np.ndarray) -> np.ndarray:
    """The boundary's own box: centred on the points, along their principal axis, clockwise."""
    centre = points.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((points - centre).T))
    along, across = vectors[:, 1], vectors[:, 0]
    u = (points - centre) @ along
    v = (points - centre) @ across
    corners = np.array(
        [
            centre + u.min() * along + v.min() * across,
            centre + u.max() * along + v.min() * across,
            centre + u.max() * along + v.max() * across,
            centre + u.min() * along + v.max() * across,
        ]
    )
    return _clockwise(canonical_quad(corners))


def _assignment_box(points: np.ndarray, coarse: np.ndarray) -> np.ndarray:
    """The boundary's principal box in the coarse quad's corner order: its ends cut the keys
    square, where the coarse quad's own ends can be a diagonal spike that drags points of the
    long edges onto an end."""
    box = _principal_box(points)
    if np.linalg.norm(box[0] - coarse[0]) > np.linalg.norm(box[2] - coarse[0]):
        box = np.roll(box, 2, axis=0)
    return box


def fit_rectangle(
    points: np.ndarray,
    coarse_quad_px: np.ndarray,
    width: int,
    height: int,
    focal: float | None = None,
    white_keys: float = WHITE_KEY_COUNT,
    depth_units: float = DEPTH_UNITS,
    gray: np.ndarray | None = None,
) -> RectFit:
    """The rectangle pose whose outline best explains the boundary points.

    With no focal given the boundary chooses it too; the orthonormality residual of a coarse
    quad is flat in focal on frontal and end-on views and returns whatever it likes. On one
    image the focal and the far end trade off, so a caller with a series of frames holds a
    focal settled over many of them.
    """
    cx, cy = width / 2.0, height / 2.0
    world = _world(white_keys, depth_units)
    held_focal = focal is not None
    unfitted = RectFit(
        quad_px=canonical_quad(coarse_quad_px), cost=float("inf"), focal=focal or 0.0
    )
    # every start is solved and the closest-explaining outline wins: a spiky-mask start can
    # land 2 px from the boundary but 270 px off at the far end; the box lands within 7 px
    best: tuple[float, np.ndarray, float, float, np.ndarray] | None = None
    for coarse in (_clockwise(canonical_quad(coarse_quad_px)), _principal_box(points)):
        if focal is None:
            focal_used, params, cost = _scan_focal(points, coarse, width, cx, cy, world)
        else:
            focal_used = focal
            params, cost = _fit_at_focal(points, coarse, focal, cx, cy, world)
        if not np.isfinite(cost):
            continue
        fitted = project(params, focal_used, cx, cy, world)
        distances, _ = _segment_distances(points, fitted)
        residual = float(np.median(distances))
        if best is None or residual < best[0]:
            best = (residual, fitted, cost, focal_used, coarse)
    # the fit answers to the boundary, not the coarse quad: a coarse quad from a sliver of mask
    # is garbage exactly when the fit matters most, while the boundary is real evidence
    if best is None or best[0] > _MAX_RESIDUAL_PX:
        return unfitted
    _, fitted, cost, focal, coarse = best
    # both long edges plus one end are enough once the focal is held, since the known length
    # places the missing end; unheld, every edge must be seen or focal and end trade off freely
    _, which = _segment_distances(points, _assignment_box(points, coarse))
    seen = np.bincount(which, minlength=4) >= _MIN_EDGE_POINTS
    ends_needed = 1 if held_focal else 2
    if not (seen[0] and seen[2]) or int(seen[1]) + int(seen[3]) < ends_needed:
        return unfitted
    if _end_ratio(fitted) > _MAX_END_RATIO:
        return unfitted
    quad_px = _refine_ends(gray, fitted) if gray is not None else fitted
    return RectFit(quad_px=quad_px, cost=cost, focal=focal)


def _end_ratio(quad: np.ndarray) -> float:
    """How many times wider one end of the keybed is drawn than the other."""
    near = float(np.linalg.norm(quad[3] - quad[0]))
    far = float(np.linalg.norm(quad[1] - quad[2]))
    return max(near, far) / max(min(near, far), 1e-6)


def _line_through(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centre = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - centre)
    return centre, np.asarray(vt[0])


def _intersection(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> np.ndarray:
    t = np.linalg.solve(np.array([d1, -d2]).T, c2 - c1)
    return np.asarray(c1 + t[0] * d1)


def _end_from_pixels(
    gray: np.ndarray, quad: np.ndarray, end: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """The line of the keys' edge at one end, read from the picture: along the fitted end,
    the brightness steps down from the white keys to whatever lies beyond, and that step
    is found within a wide band across the end. Measured on the user's recordings the
    mask's end evidence covers only part of the end, so a line through it extrapolates
    wrongly to the far corner; the picture has the whole edge."""
    a, b = quad[end], quad[(end + 1) % 4]
    along = b - a
    length = float(np.linalg.norm(along))
    if length < 4.0:
        return None
    outward = np.array([-along[1], along[0]]) / length
    if np.dot(outward, (a + b) / 2 - quad.mean(axis=0)) < 0:
        outward = -outward
    ts = np.linspace(_END_TRIM, 1.0 - _END_TRIM, _END_SAMPLES)
    offsets = np.arange(-_END_SEARCH_PX, _END_SEARCH_PX + 1e-6, _SEARCH_STEP)
    base = a[None, :] + ts[:, None] * along[None, :]
    samples = base[:, None, :] + offsets[None, :, None] * outward[None, None, :]
    values = _sample(gray, samples[:, :, 0].ravel(), samples[:, :, 1].ravel())
    values = values.reshape(len(ts), len(offsets))
    # the edge is a step from key-bright to dark, not a plain derivative: the case's own edge
    # just beyond is a sharper step but dark on both sides, and a plain derivative picked it instead
    reach = round(_END_CONTRAST_PX / _SEARCH_STEP)
    inside = np.full_like(values, np.nan)
    outside = np.full_like(values, np.nan)
    for k in range(reach, len(offsets) - reach):
        inside[:, k] = values[:, k - reach : k].mean(axis=1)
        outside[:, k] = values[:, k + 1 : k + 1 + reach].mean(axis=1)
    contrast = np.where(inside >= _MIN_KEY_BRIGHTNESS, inside - outside, -np.inf)
    best = np.nanargmax(contrast, axis=1)
    strong = contrast[np.arange(len(ts)), best] > _MIN_END_CONTRAST
    if int(strong.sum()) < _MIN_END_SAMPLES:
        return None
    found = base[strong] + offsets[best[strong]][:, None] * outward[None, :]
    # the picture shifts the end to the keys' edge always, but sets its tilt only when samples
    # span most of it; the far end's short bright span otherwise swung the line frame to frame
    shifted = (
        a + float(ts[strong].mean()) * along + float(np.median(offsets[best[strong]])) * outward
    )
    parallel = (np.asarray(shifted), along / length)
    if float(ts[strong].max() - ts[strong].min()) < _MIN_END_SPAN:
        return parallel
    centre, direction = _line_through(found)
    residual = np.abs((found - centre) @ np.array([-direction[1], direction[0]]))
    kept = residual <= max(2.0 * float(np.median(residual)), 1.0)
    if int(kept.sum()) < _MIN_END_SAMPLES:
        return parallel
    centre, direction = _line_through(found[kept])
    if abs(float(np.dot(direction, along / length))) < _MIN_END_ALIGNMENT:
        return parallel
    return centre, direction


def _refine_ends(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """The ends re-drawn where the picture shows the keys stopping, cut by the rectangle's
    long edges. The rectangle's geometry sets the ends' tilt from the focal and the long
    edges' convergence, and on real frames that tilt disagrees with the keys' actual front;
    the long edges are the rectangle's strength and stay."""
    blurred = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), _BLUR_SIGMA)
    out = quad.copy()
    long_edges = {0: (quad[0], quad[1] - quad[0]), 2: (quad[3], quad[2] - quad[3])}
    for end, corners in ((1, {1: 0, 2: 2}), (3, {0: 0, 3: 2})):
        line = _end_from_pixels(blurred, quad, end)
        if line is None:
            continue
        centre, direction = line
        for corner, long_edge in corners.items():
            moved = _intersection(centre, direction, *long_edges[long_edge])
            if float(np.linalg.norm(moved - quad[corner])) <= _MAX_END_SHIFT_PX:
                out[corner] = moved
    return out
