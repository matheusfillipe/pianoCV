import { grayscale } from "./detector";
import type { Point } from "./homography";
import type { Size } from "./keyspace";
import { pointInQuad } from "./stillness";

declare global {
  interface Window {
    // recent update() costs in ms, for the lab to read off a live page rather than guess
    kvtFollowMs?: number[];
  }
}
const FOLLOW_MS_HISTORY = 300;

export interface GrayFrame {
  readonly data: Float32Array;
  readonly width: number;
  readonly height: number;
}

export interface Similarity {
  readonly scale: number;
  readonly rotation: number;
  readonly tx: number;
  readonly ty: number;
}

export interface MotionEstimate {
  readonly transform: Similarity;
  /** Where every point that still matched anything landed, so the caller can
   * keep tracking them next frame without re-picking features. */
  readonly trackedPoints: readonly Point[];
  readonly inliers: number;
}

const WORK_WIDTH = 320;
const PATCH_HALF = 3;
const SEARCH_RADIUS = 6;
// the patch is a float grayscale in 0..1, so this is a mean squared error per pixel: a real
// match is near zero, an unrelated patch (occlusion, noise) lands an order of magnitude higher
const MAX_SSD_PER_PIXEL = 0.02;
const MIN_INLIERS = 6;
const MIN_INLIER_FRACTION = 0.5;
const MAX_FIT_RESIDUAL_PX = 2;
const FEATURE_GRID_COLS = 6;
const FEATURE_GRID_ROWS = 4;

function clampFrame(value: number, size: number): number {
  return Math.max(0, Math.min(size - 1, value));
}

function patchSsd(
  source: GrayFrame,
  sx: number,
  sy: number,
  target: GrayFrame,
  tx: number,
  ty: number,
): number | null {
  let sum = 0;
  let count = 0;
  for (let dy = -PATCH_HALF; dy <= PATCH_HALF; dy += 1) {
    const sy2 = sy + dy;
    const ty2 = ty + dy;
    if (sy2 < 0 || sy2 >= source.height || ty2 < 0 || ty2 >= target.height) {
      return null;
    }
    for (let dx = -PATCH_HALF; dx <= PATCH_HALF; dx += 1) {
      const sx2 = sx + dx;
      const tx2 = tx + dx;
      if (sx2 < 0 || sx2 >= source.width || tx2 < 0 || tx2 >= target.width) {
        return null;
      }
      const diff =
        source.data[sy2 * source.width + sx2] -
        target.data[ty2 * target.width + tx2];
      sum += diff * diff;
      count += 1;
    }
  }
  return sum / count;
}

// a parabola through the best candidate and its neighbours reads the true minimum between grid
// steps, so a slow pan does not look like a staircase
function subpixelPeak(before: number, center: number, after: number): number {
  const denom = before - 2 * center + after;
  if (Math.abs(denom) < 1e-9) {
    return 0;
  }
  return (0.5 * (before - after)) / denom;
}

interface Matched {
  readonly to: Point;
  readonly error: number;
}

function trackPoint(
  source: GrayFrame,
  from: Point,
  target: GrayFrame,
): Matched | null {
  const ox = Math.round(from.x);
  const oy = Math.round(from.y);
  let best: { x: number; y: number; error: number } | null = null;
  for (let dy = -SEARCH_RADIUS; dy <= SEARCH_RADIUS; dy += 1) {
    for (let dx = -SEARCH_RADIUS; dx <= SEARCH_RADIUS; dx += 1) {
      const x = ox + dx;
      const y = oy + dy;
      const error = patchSsd(source, ox, oy, target, x, y);
      if (error !== null && (best === null || error < best.error)) {
        best = { x, y, error };
      }
    }
  }
  if (best === null || best.error > MAX_SSD_PER_PIXEL) {
    return null;
  }
  const left = patchSsd(source, ox, oy, target, best.x - 1, best.y);
  const right = patchSsd(source, ox, oy, target, best.x + 1, best.y);
  const up = patchSsd(source, ox, oy, target, best.x, best.y - 1);
  const down = patchSsd(source, ox, oy, target, best.x, best.y + 1);
  const sx =
    left !== null && right !== null ? subpixelPeak(left, best.error, right) : 0;
  const sy =
    up !== null && down !== null ? subpixelPeak(up, best.error, down) : 0;
  return { to: { x: best.x + sx, y: best.y + sy }, error: best.error };
}

export function applySimilarity(s: Similarity, p: Point): Point {
  const c = Math.cos(s.rotation) * s.scale;
  const n = Math.sin(s.rotation) * s.scale;
  return { x: c * p.x - n * p.y + s.tx, y: n * p.x + c * p.y + s.ty };
}

/** Least-squares similarity (uniform scale, rotation, translation) between two point sets,
 * the closed-form solution to minimizing total squared point-to-point distance. */
export function fitSimilarity(
  from: readonly Point[],
  to: readonly Point[],
): Similarity | null {
  const n = from.length;
  let meanFromX = 0;
  let meanFromY = 0;
  let meanToX = 0;
  let meanToY = 0;
  for (let i = 0; i < n; i += 1) {
    meanFromX += from[i].x;
    meanFromY += from[i].y;
    meanToX += to[i].x;
    meanToY += to[i].y;
  }
  meanFromX /= n;
  meanFromY /= n;
  meanToX /= n;
  meanToY /= n;
  let numerator = 0;
  let cross = 0;
  let denom = 0;
  for (let i = 0; i < n; i += 1) {
    const ux = from[i].x - meanFromX;
    const uy = from[i].y - meanFromY;
    const vx = to[i].x - meanToX;
    const vy = to[i].y - meanToY;
    numerator += ux * vx + uy * vy;
    cross += ux * vy - uy * vx;
    denom += ux * ux + uy * uy;
  }
  if (denom < 1e-6) {
    return null;
  }
  const a = numerator / denom;
  const b = cross / denom;
  return {
    scale: Math.hypot(a, b),
    rotation: Math.atan2(b, a),
    tx: meanToX - (a * meanFromX - b * meanFromY),
    ty: meanToY - (b * meanFromX + a * meanFromY),
  };
}

function residual(transform: Similarity, from: Point, to: Point): number {
  const p = applySimilarity(transform, from);
  return Math.hypot(p.x - to.x, p.y - to.y);
}

/** How the picture moved between two frames, read off a handful of tracked points, or null when
 * too few of them still match anything or the motion they agree on does not fit them well. */
export function estimateMotion(
  source: GrayFrame,
  points: readonly Point[],
  target: GrayFrame,
): MotionEstimate | null {
  const matched: { at: Point; to: Point }[] = [];
  for (const at of points) {
    const found = trackPoint(source, at, target);
    if (found !== null) {
      matched.push({ at, to: found.to });
    }
  }
  if (
    matched.length < MIN_INLIERS ||
    matched.length < points.length * MIN_INLIER_FRACTION
  ) {
    return null;
  }
  const initial = fitSimilarity(
    matched.map((m) => m.at),
    matched.map((m) => m.to),
  );
  if (initial === null) {
    return null;
  }
  // a cheap stand-in for RANSAC: one pass dropping the worst quarter survives a few bad
  // matches (a hand over a key, a repeated pattern) without iterating to convergence
  const ranked = matched
    .map((m) => ({ m, error: residual(initial, m.at, m.to) }))
    .sort((a, b) => a.error - b.error);
  const kept = ranked
    .slice(0, Math.max(MIN_INLIERS, Math.floor(ranked.length * 0.75)))
    .map((r) => r.m);
  if (kept.length < MIN_INLIERS) {
    return null;
  }
  const fit = fitSimilarity(
    kept.map((m) => m.at),
    kept.map((m) => m.to),
  );
  if (fit === null) {
    return null;
  }
  const meanResidual =
    kept.reduce((sum, m) => sum + residual(fit, m.at, m.to), 0) / kept.length;
  if (meanResidual > MAX_FIT_RESIDUAL_PX) {
    return null;
  }
  return {
    transform: fit,
    trackedPoints: matched.map((m) => m.to),
    inliers: kept.length,
  };
}

function gradientMagnitude(frame: GrayFrame, x: number, y: number): number {
  const i = y * frame.width + x;
  const gx = frame.data[i + 1] - frame.data[i - 1];
  const gy = frame.data[i + frame.width] - frame.data[i - frame.width];
  return Math.abs(gx) + Math.abs(gy);
}

/** High-contrast points inside the quad, one per grid cell so they spread across it rather than
 * clumping on whichever edge is sharpest; a keybed's black-key edges dominate the picks. */
export function pickFeatures(
  frame: GrayFrame,
  quad: readonly Point[],
): Point[] {
  const xs = quad.map((p) => p.x);
  const ys = quad.map((p) => p.y);
  const minX = clampFrame(Math.floor(Math.min(...xs)), frame.width);
  const maxX = clampFrame(Math.ceil(Math.max(...xs)), frame.width);
  const minY = clampFrame(Math.floor(Math.min(...ys)), frame.height);
  const maxY = clampFrame(Math.ceil(Math.max(...ys)), frame.height);
  const cellW = (maxX - minX) / FEATURE_GRID_COLS;
  const cellH = (maxY - minY) / FEATURE_GRID_ROWS;
  const points: Point[] = [];
  if (cellW < 1 || cellH < 1) {
    return points;
  }
  for (let row = 0; row < FEATURE_GRID_ROWS; row += 1) {
    for (let col = 0; col < FEATURE_GRID_COLS; col += 1) {
      const x0 = minX + col * cellW;
      const y0 = minY + row * cellH;
      let best: Point | null = null;
      let bestScore = 0;
      const fromY = Math.max(1, Math.ceil(y0));
      const toY = Math.min(frame.height - 2, Math.floor(y0 + cellH));
      const fromX = Math.max(1, Math.ceil(x0));
      const toX = Math.min(frame.width - 2, Math.floor(x0 + cellW));
      for (let y = fromY; y <= toY; y += 1) {
        for (let x = fromX; x <= toX; x += 1) {
          if (!pointInQuad(quad, { x, y })) {
            continue;
          }
          const score = gradientMagnitude(frame, x, y);
          if (score > bestScore) {
            bestScore = score;
            best = { x, y };
          }
        }
      }
      if (best !== null) {
        points.push(best);
      }
    }
  }
  return points;
}

export interface Follower {
  /** Anchors the follower to the tracker's own corners, discarding whatever it had estimated
   * before. Call this whenever the tracker (re)confirms the keybed. */
  readonly reset: (
    quad: readonly Point[],
    frame: CanvasImageSource,
    size: Size,
  ) => void;
  /** The followed quad for this frame, in the same frame-fraction coordinates as reset()'s
   * quad. Returns null until reset() has been called at least once. */
  readonly update: (
    frame: CanvasImageSource,
    size: Size,
  ) => readonly Point[] | null;
}

function workSize(size: Size): Size {
  const width = Math.max(1, Math.min(WORK_WIDTH, size.width));
  const height = Math.max(1, Math.round((width / size.width) * size.height));
  return { width, height };
}

export function createFollower(): Follower {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const extract = (frame: CanvasImageSource, size: Size): GrayFrame => {
    if (canvas.width !== size.width || canvas.height !== size.height) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
    if (ctx === null) {
      return { data: new Float32Array(size.width * size.height), ...size };
    }
    ctx.drawImage(frame, 0, 0, size.width, size.height);
    const pixels = ctx.getImageData(0, 0, size.width, size.height).data;
    const data = new Float32Array(size.width * size.height);
    grayscale(pixels, data);
    return { data, ...size };
  };

  let workAt: Size = { width: 0, height: 0 };
  let previous: GrayFrame | null = null;
  let points: Point[] = [];
  let quadPx: Point[] = [];
  let quadFraction: Point[] | null = null;

  return {
    reset: (quad, frame, size) => {
      workAt = workSize(size);
      const gray = extract(frame, workAt);
      previous = gray;
      quadPx = quad.map((p) => ({
        x: p.x * workAt.width,
        y: p.y * workAt.height,
      }));
      points = pickFeatures(gray, quadPx);
      quadFraction = quad.map((p) => ({ x: p.x, y: p.y }));
    },
    update: (frame, size) => {
      if (previous === null || quadFraction === null) {
        return null;
      }
      const at = workSize(size);
      // a resolution change (rare: only the video's own size can change it) is not worth
      // resampling every tracked point for, so just hold position until the next reset
      if (at.width !== workAt.width || at.height !== workAt.height) {
        return quadFraction;
      }
      const started = performance.now();
      const gray = extract(frame, workAt);
      const estimate = estimateMotion(previous, points, gray);
      window.kvtFollowMs ??= [];
      const history = window.kvtFollowMs;
      history.push(performance.now() - started);
      if (history.length > FOLLOW_MS_HISTORY) {
        history.shift();
      }
      if (estimate === null) {
        return quadFraction;
      }
      previous = gray;
      points = [...estimate.trackedPoints];
      quadPx = quadPx.map((p) => applySimilarity(estimate.transform, p));
      quadFraction = quadPx.map((p) => ({
        x: p.x / workAt.width,
        y: p.y / workAt.height,
      }));
      return quadFraction;
    },
  };
}
