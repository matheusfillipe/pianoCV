// Fit the keybed as what it is: a known rectangle seen through a pinhole camera.
//
// A free quad has eight degrees of freedom and nothing stops its far end from wandering. The
// keybed rectangle under a rotation and a translation has six, and every quad that shape can
// produce is a legal keybed. The six are fitted to the mask's whole boundary at once, so the
// far end is placed by the outline plus the known proportions, never by the few pixels at the
// thin end. Port of tools/src/pianocv/rectfit.py, measured there on the pose grid and the clips.

import { findHomography, type Point } from "./homography";
import { canonicalQuad, keybedDepth, WHITE_KEY_COUNT } from "./pose";
import {
  bilinear,
  blur,
  MIN_RESPONSE,
  SEARCH_PX,
  SEARCH_STEP,
} from "./refineedges";

// the focal at which the quad is a rectangle of the measured shape: the homography's
// columns must be orthogonal and equal length; flat on a top view, sharp on an oblique one
export function estimateFocalFraction(
  quad: Point[],
  width: number,
  height: number,
): number {
  const h = findHomography(world2d(), clockwise(canonicalQuad(quad)));
  const cx = width / 2;
  const cy = height / 2;
  const residual = (fraction: number): number => {
    const focal = fraction * width;
    const column = (col: number): number[] => [
      (h[col] - cx * h[col + 6]) / focal,
      (h[col + 3] - cy * h[col + 6]) / focal,
      h[col + 6],
    ];
    const c0 = column(0);
    const c1 = column(1);
    const n0 = Math.hypot(...c0);
    const n1 = Math.hypot(...c1);
    return Math.abs(dot(c0, c1)) / (n0 * n1) + Math.abs(1 - n1 / n0);
  };
  let best = 0.4;
  let bestValue = residual(best);
  for (let fraction = 0.42; fraction <= 2.5; fraction += 0.02) {
    const value = residual(fraction);
    if (value < bestValue) {
      best = fraction;
      bestValue = value;
    }
  }
  return best;
}

function world(): number[][] {
  return [
    [0, 0, 0],
    [WHITE_KEY_COUNT, 0, 0],
    [WHITE_KEY_COUNT, keybedDepth(), 0],
    [0, keybedDepth(), 0],
  ];
}

function world2d(): Point[] {
  return world().map(([x, y]) => ({ x, y }));
}

// the keybed's depth read off a quad at a known focal: the homography's two columns are
// the rotation's columns scaled by the world axes, so their length ratio is the depth
export function keybedDepthFromQuad(
  quad: Point[],
  focal: number,
  cx: number,
  cy: number,
): number {
  const unitWorld: Point[] = [
    { x: 0, y: 0 },
    { x: WHITE_KEY_COUNT, y: 0 },
    { x: WHITE_KEY_COUNT, y: 1 },
    { x: 0, y: 1 },
  ];
  const h = findHomography(unitWorld, clockwise(canonicalQuad(quad)));
  const column = (col: number): number[] => [
    (h[col] - cx * h[col + 6]) / focal,
    (h[col + 3] - cy * h[col + 6]) / focal,
    h[col + 6],
  ];
  return Math.hypot(...column(1)) / Math.hypot(...column(0));
}
const ITERATIONS = 40;
const ASSIGNMENT_ROUNDS = 3;
const HUBER_PX = 3;
const MAX_POINTS = 400;
const BORDER_PX = 2;
const STEP = 1e-4;
// closer than one key width to the camera the projection is meaningless
const MIN_DEPTH_UNITS = 1;
// one mask cell on a 640 px frame: a fit the boundary sits further from than that is not
// explaining it
const MAX_RESIDUAL_PX = 4.5;
const MIN_EDGE_POINTS = 3;
const MIN_EDGE_VOTE = 24;
// one end may be drawn this much wider than the other, the same as their distance ratio; the
// hand-labelled views span 1.54 to 2.65 and a sliding rectangle scores 4.0 and up
const MAX_END_RATIO = 3.2;
// an end is re-drawn where the picture shows the keys stopping
const END_SAMPLES = 48;
const END_TRIM = 0.08;
const END_SEARCH_PX = 25;
const MIN_END_SAMPLES = 10;
const MAX_END_SHIFT_PX = 30;
const END_CONTRAST_PX = 4;
const MIN_KEY_BRIGHTNESS = 110;
const MIN_END_CONTRAST = 30;
const MIN_END_ALIGNMENT = 0.9;
const MIN_END_SPAN = 0.5;
// a boundary crossing further than this fraction of the span from the coarse quad belongs
// to some other blob in the mask
const GATE = 0.15;
// focal lengths as fractions of the frame width, from a wide phone lens to a long zoom
export const FOCAL_SCAN = [0.5, 0.65, 0.8, 1.0, 1.3, 1.7, 2.4, 3.5];
const GOLDEN = (Math.sqrt(5) - 1) / 2;
const GOLDEN_ITERATIONS = 8;

// why the last fitRectangle call declined, for a lab session watching a recording
export let lastDecline = "";

export interface RectFit {
  // the rectangle with its ends moved to where the picture shows the keys stopping
  quad: Point[];
  // the rectangle the solver found, the start for the next frame's solve: seeding it with
  // the picture-moved ends made the solver alternate between two minima frame to frame
  rectangle: Point[];
  cost: number;
  focal: number;
}

type Mat3 = number[][];

interface Pose {
  base: Mat3;
  params: number[];
}

function rodrigues(w: number[]): Mat3 {
  const angle = Math.hypot(w[0], w[1], w[2]);
  if (angle < 1e-12) {
    return [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
  }
  const [x, y, z] = [w[0] / angle, w[1] / angle, w[2] / angle];
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const t = 1 - c;
  return [
    [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
    [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
    [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
  ];
}

function multiply(a: Mat3, b: Mat3): Mat3 {
  return a.map((row) =>
    [0, 1, 2].map(
      (j) => row[0] * b[0][j] + row[1] * b[1][j] + row[2] * b[2][j],
    ),
  );
}

function rotation(pose: Pose): Mat3 {
  return multiply(pose.base, rodrigues(pose.params.slice(0, 3)));
}

function camera(pose: Pose): number[][] {
  const r = rotation(pose);
  const t = pose.params.slice(3);
  return world().map((p) =>
    [0, 1, 2].map(
      (i) => r[i][0] * p[0] + r[i][1] * p[1] + r[i][2] * p[2] + t[i],
    ),
  );
}

function project(pose: Pose, focal: number, cx: number, cy: number): Point[] {
  return camera(pose).map((c) => ({
    x: cx + (focal * c[0]) / Math.max(c[2], 1e-6),
    y: cy + (focal * c[1]) / Math.max(c[2], 1e-6),
  }));
}

function inFront(pose: Pose): boolean {
  return camera(pose).every((c) => c[2] > MIN_DEPTH_UNITS);
}

function signedArea(quad: Point[]): number {
  let area = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    area += a.x * b.y - a.y * b.x;
  }
  return area;
}

// the world corners project clockwise on screen under any rotation; a quad wound the other
// way is the mirror image, which no pose reaches, so it is reflected along the span
function clockwise(quad: Point[]): Point[] {
  return signedArea(quad) >= 0 ? quad : [quad[1], quad[0], quad[3], quad[2]];
}

function dot(a: number[], b: number[]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function unit(v: number[]): number[] {
  const n = Math.hypot(v[0], v[1], v[2]);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

// the homography from the rectangle to the quad, with the calibration undone, is the
// rotation's first two columns and the translation up to one scale
function initialPose(
  quad: Point[],
  focal: number,
  cx: number,
  cy: number,
): Pose {
  const h = findHomography(world2d(), quad);
  const b: number[][] = [[], [], []];
  for (let col = 0; col < 3; col += 1) {
    b[0][col] = (h[col] - cx * h[col + 6]) / focal;
    b[1][col] = (h[col + 3] - cy * h[col + 6]) / focal;
    b[2][col] = h[col + 6];
  }
  const sign = b[2][2] < 0 ? -1 : 1;
  const c0 = [b[0][0], b[1][0], b[2][0]].map((v) => v * sign);
  const c1 = [b[0][1], b[1][1], b[2][1]].map((v) => v * sign);
  const c2 = [b[0][2], b[1][2], b[2][2]].map((v) => v * sign);
  const scale = (Math.hypot(...c0) + Math.hypot(...c1)) / 2;
  const r1 = unit(c0);
  const along = dot(r1, c1);
  const r2 = unit([
    c1[0] - along * r1[0],
    c1[1] - along * r1[1],
    c1[2] - along * r1[2],
  ]);
  const r3 = cross(r1, r2);
  return {
    base: [
      [r1[0], r2[0], r3[0]],
      [r1[1], r2[1], r3[1]],
      [r1[2], r2[2], r3[2]],
    ],
    params: [0, 0, 0, c2[0] / scale, c2[1] / scale, c2[2] / scale],
  };
}

interface Distances {
  distance: Float64Array;
  which: Int32Array;
}

function segmentDistances(points: Point[], quad: Point[]): Distances {
  const distance = new Float64Array(points.length).fill(Infinity);
  const which = new Int32Array(points.length);
  for (let i = 0; i < 4; i += 1) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq < 1e-9) {
      continue;
    }
    for (let k = 0; k < points.length; k += 1) {
      const p = points[k];
      const t = Math.min(
        1,
        Math.max(0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq),
      );
      const d = Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
      if (d < distance[k]) {
        distance[k] = d;
        which[k] = i;
      }
    }
  }
  return { distance, which };
}

function huber(d: number): number {
  return d <= HUBER_PX ? d : Math.sqrt(2 * HUBER_PX * d - HUBER_PX * HUBER_PX);
}

// each edge is one line measurement however many boundary points lie on it; unweighted,
// the hundreds of points on the long edges outvote the dozen on the thin far end
function edgeWeights(which: Int32Array, count: number): number[] {
  const counts = [0, 0, 0, 0];
  for (const edge of which) {
    counts[edge] += 1;
  }
  // an edge with few points gets a proportionally smaller vote: un-normalized, four far-end
  // crossings swung the rigid rectangle 3-5 px per frame on sub-pixel noise
  return counts.map((n) => Math.sqrt(count / (4 * Math.max(n, MIN_EDGE_VOTE))));
}

function assignedDistance(p: Point, quad: Point[], edge: number): number {
  const a = quad[edge];
  const b = quad[(edge + 1) % 4];
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const lengthSq = Math.max(dx * dx + dy * dy, 1e-9);
  const t = Math.min(
    1,
    Math.max(0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSq),
  );
  return Math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy);
}

function residuals(
  points: Point[],
  quad: Point[],
  which: Int32Array,
  weights: number[],
): Float64Array {
  const out = new Float64Array(points.length);
  for (let k = 0; k < points.length; k += 1) {
    out[k] =
      huber(assignedDistance(points[k], quad, which[k])) * weights[which[k]];
  }
  return out;
}

function median(values: Float64Array): number {
  const sorted = Float64Array.from(values).sort();
  return sorted.length ? sorted[sorted.length >> 1] : Infinity;
}

// how far the boundary sits from a quad, the yardstick a fit is judged by: at the wrong
// focal a rectangle explains it 1-2 px worse than the free quad, with corners off by tens
export function boundaryResidual(points: Point[], quad: Point[]): number {
  return median(segmentDistances(points, quad).distance);
}

function sumSquares(values: Float64Array): number {
  let total = 0;
  for (const v of values) {
    total += v * v;
  }
  return total;
}

function solve(matrix: number[][], rhs: number[]): number[] | null {
  const n = rhs.length;
  const a = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) {
        pivot = row;
      }
    }
    if (Math.abs(a[pivot][col]) < 1e-18) {
      return null;
    }
    [a[col], a[pivot]] = [a[pivot], a[col]];
    for (let row = 0; row < n; row += 1) {
      if (row === col) {
        continue;
      }
      const factor = a[row][col] / a[col][col];
      for (let k = col; k <= n; k += 1) {
        a[row][k] -= factor * a[col][k];
      }
    }
  }
  return a.map((row, i) => row[n] / row[i]);
}

interface Solution {
  pose: Pose;
  cost: number;
}

// Levenberg-Marquardt over (rotation, translation) with the focal held fixed
function fitAtFocal(
  points: Point[],
  coarse: Point[],
  focal: number,
  cx: number,
  cy: number,
): Solution {
  let pose = initialPose(coarse, focal, cx, cy);
  if (!inFront(pose)) {
    return { pose, cost: Infinity };
  }
  // assignment is redone once per solve, not continuously: every step would let the
  // rectangle slide onto a wrong, lower-cost fit, while never redoing it locks in a bad start
  let which = assignEdges(points, assignmentBox(points, coarse));
  let cost = Infinity;
  for (let round = 0; round < ASSIGNMENT_ROUNDS; round += 1) {
    const solved = solveAssigned(points, which, pose, focal, cx, cy);
    pose = solved.pose;
    cost = solved.cost;
    const reassigned = assignEdges(points, project(pose, focal, cx, cy));
    if (reassigned.every((edge, i) => edge === which[i])) {
      break;
    }
    which = reassigned;
  }
  return { pose, cost };
}

function solveAssigned(
  points: Point[],
  which: Int32Array,
  initial: Pose,
  focal: number,
  cx: number,
  cy: number,
): Solution {
  const weights = edgeWeights(which, points.length);
  let pose = initial;
  const evaluate = (p: Pose): Float64Array =>
    residuals(points, project(p, focal, cx, cy), which, weights);
  let current = evaluate(pose);
  let cost = sumSquares(current);
  let damping = 1e-2;
  for (let iteration = 0; iteration < ITERATIONS; iteration += 1) {
    const jacobian: Float64Array[] = [];
    for (let k = 0; k < 6; k += 1) {
      const bumped = { base: pose.base, params: [...pose.params] };
      bumped.params[k] += STEP;
      const column = evaluate(bumped);
      for (let i = 0; i < column.length; i += 1) {
        column[i] = (column[i] - current[i]) / STEP;
      }
      jacobian.push(column);
    }
    const gram = jacobian.map((a) => jacobian.map((b) => dotArrays(a, b)));
    const gradient = jacobian.map((a) => -dotArrays(a, current));
    const damped = gram.map((row, i) =>
      row.map((v, j) => (i === j ? v + damping * (v + 1e-9) : v)),
    );
    const step = solve(damped, gradient);
    if (!step) {
      break;
    }
    const candidate = {
      base: pose.base,
      params: pose.params.map((v, i) => v + step[i]),
    };
    // a step that swings the plane through the camera is a fold of the cost, never a fit
    let trialCost = Infinity;
    let trial = current;
    if (inFront(candidate)) {
      trial = evaluate(candidate);
      trialCost = sumSquares(trial);
    }
    if (trialCost < cost) {
      pose = candidate;
      current = trial;
      cost = trialCost;
      damping = Math.max(damping / 3, 1e-6);
      if (Math.hypot(...step) < 1e-5) {
        break;
      }
    } else {
      damping = Math.min(damping * 5, 1e6);
    }
  }
  return { pose, cost };
}

function dotArrays(a: Float64Array, b: Float64Array): number {
  let total = 0;
  for (let i = 0; i < a.length; i += 1) {
    total += a[i] * b[i];
  }
  return total;
}

interface FocalSolution extends Solution {
  focal: number;
}

// the boundary decides the focal: the orthonormality residual of a coarse quad is flat in
// focal on frontal and end-on views and returns whatever it likes
function scanFocal(
  points: Point[],
  coarse: Point[],
  width: number,
  cx: number,
  cy: number,
): FocalSolution {
  const tried = new Map<number, Solution>();
  const at = (logFocal: number): number => {
    let solution = tried.get(logFocal);
    if (!solution) {
      solution = fitAtFocal(points, coarse, Math.exp(logFocal), cx, cy);
      tried.set(logFocal, solution);
    }
    return solution.cost;
  };
  const scan = FOCAL_SCAN.map((f) => Math.log(f * width));
  const costs = scan.map(at);
  let best = 0;
  for (let i = 1; i < costs.length; i += 1) {
    if (costs[i] < costs[best]) {
      best = i;
    }
  }
  let a = scan[Math.max(best - 1, 0)];
  let b = scan[Math.min(best + 1, scan.length - 1)];
  let c = b - GOLDEN * (b - a);
  let d = a + GOLDEN * (b - a);
  for (let i = 0; i < GOLDEN_ITERATIONS; i += 1) {
    if (at(c) < at(d)) {
      b = d;
      d = c;
      c = b - GOLDEN * (b - a);
    } else {
      a = c;
      c = d;
      d = a + GOLDEN * (b - a);
    }
  }
  let bestLog = scan[best];
  let bestSolution = tried.get(bestLog) ?? {
    pose: initialPose(coarse, 1, cx, cy),
    cost: Infinity,
  };
  for (const [logFocal, solution] of tried) {
    if (solution.cost < bestSolution.cost) {
      bestLog = logFocal;
      bestSolution = solution;
    }
  }
  return { ...bestSolution, focal: Math.exp(bestLog) };
}

function edgeGate(quad: Point[]): number {
  return GATE * Math.hypot(quad[1].x - quad[0].x, quad[1].y - quad[0].y);
}

// Which cells belong to the largest blob of the mask above half probability.
function largestComponent(probability: Float32Array, size: number): Uint8Array {
  const label = new Int32Array(size * size);
  const stack: number[] = [];
  let best = 0;
  let bestSize = 0;
  let next = 0;
  for (let seed = 0; seed < label.length; seed += 1) {
    if (label[seed] !== 0 || probability[seed] <= 0.5) {
      continue;
    }
    next += 1;
    let count = 0;
    label[seed] = next;
    stack.push(seed);
    while (stack.length) {
      const cell = stack.pop() ?? 0;
      count += 1;
      const row = Math.floor(cell / size);
      const col = cell - row * size;
      for (const [dr, dc] of [
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]) {
        const r = row + dr;
        const c = col + dc;
        if (r < 0 || r >= size || c < 0 || c >= size) {
          continue;
        }
        const n = r * size + c;
        if (label[n] === 0 && probability[n] > 0.5) {
          label[n] = next;
          stack.push(n);
        }
      }
    }
    if (count > bestSize) {
      bestSize = count;
      best = next;
    }
  }
  const inside = new Uint8Array(size * size);
  for (let i = 0; i < label.length; i += 1) {
    inside[i] = label[i] === best ? 1 : 0;
  }
  return inside;
}

// the boundary in frame pixels, interpolated where the mask crosses half probability so
// it never sits on a cell; border crossings and ones far from the coarse quad are dropped
export function boundaryPoints(
  probability: Float32Array,
  size: number,
  width: number,
  height: number,
  near: Point[] | null,
): Point[] {
  const gate = near ? edgeGate(near) : Infinity;
  const crossings: Point[] = [];
  const push = (x: number, y: number): void => {
    if (x <= BORDER_PX || x >= width - 1 - BORDER_PX) {
      return;
    }
    if (y <= BORDER_PX || y >= height - 1 - BORDER_PX) {
      return;
    }
    crossings.push({ x, y });
  };
  const sx = width / (size - 1);
  const sy = height / (size - 1);
  // only the largest blob is the keybed: the mask also bleeds onto the case below the keys
  // and a chair beside them, and their crossings would be fitted as the keybed's ends
  const inside = largestComponent(probability, size);
  for (let row = 0; row < size; row += 1) {
    for (let col = 0; col < size; col += 1) {
      const here = probability[row * size + col] - 0.5;
      const hereIn = inside[row * size + col];
      if (col + 1 < size) {
        const right = probability[row * size + col + 1] - 0.5;
        if (here * right < 0 && (hereIn || inside[row * size + col + 1])) {
          push((col + here / (here - right)) * sx, row * sy);
        }
      }
      if (row + 1 < size) {
        const below = probability[(row + 1) * size + col] - 0.5;
        if (here * below < 0 && (hereIn || inside[(row + 1) * size + col])) {
          push(col * sx, (row + here / (here - below)) * sy);
        }
      }
    }
  }
  const distance = near
    ? segmentDistances(crossings, near).distance
    : new Float64Array(crossings.length);
  const kept = crossings.filter((_, i) => distance[i] <= gate);
  const stride = Math.max(1, Math.ceil(kept.length / MAX_POINTS));
  return kept.filter((_, i) => i % stride === 0);
}

// each boundary point moves onto the nearest brightness step along its edge's normal and
// is dropped when there is none, so a mask sliver under the hands never enters as a false end
export function snapToGradient(
  gray: Float32Array,
  width: number,
  height: number,
  points: Point[],
  quad: Point[],
): Point[] {
  const smooth = blur(gray, width, height);
  const { which } = segmentDistances(points, quad);
  const centre = {
    x: quad.reduce((s, p) => s + p.x, 0) / 4,
    y: quad.reduce((s, p) => s + p.y, 0) / 4,
  };
  const outwards = [0, 1, 2, 3].map((i) => {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const length = Math.max(Math.hypot(b.x - a.x, b.y - a.y), 1e-9);
    const n = { x: -(b.y - a.y) / length, y: (b.x - a.x) / length };
    const mid = {
      x: (a.x + b.x) / 2 - centre.x,
      y: (a.y + b.y) / 2 - centre.y,
    };
    return n.x * mid.x + n.y * mid.y < 0 ? { x: -n.x, y: -n.y } : n;
  });
  const steps = Math.round((2 * SEARCH_PX) / SEARCH_STEP) + 1;
  const reach = Math.round(END_CONTRAST_PX / SEARCH_STEP);
  const values = new Float64Array(steps);
  const snapped: Point[] = [];
  for (let k = 0; k < points.length; k += 1) {
    const p = points[k];
    const n = outwards[which[k]];
    for (let s = 0; s < steps; s += 1) {
      const offset = -SEARCH_PX + s * SEARCH_STEP;
      values[s] = bilinear(
        smooth,
        width,
        height,
        p.x + offset * n.x,
        p.y + offset * n.y,
      );
    }
    // the edge is a step from key-bright to dark, not the strongest gradient, which instead
    // follows the case's lip beyond it; with no key-bright reach the gradient decides anyway
    let bestContrast = -Infinity;
    let contrastOffset = 0;
    for (let s = reach; s < steps - reach; s += 1) {
      let inside = 0;
      let outside = 0;
      for (let r = 1; r <= reach; r += 1) {
        inside += values[s - r] / reach;
        outside += values[s + r] / reach;
      }
      if (inside >= MIN_KEY_BRIGHTNESS && inside - outside > bestContrast) {
        bestContrast = inside - outside;
        contrastOffset = -SEARCH_PX + s * SEARCH_STEP;
      }
    }
    // otherwise the nearest real step to the boundary, not the strongest: the strongest in
    // reach was a blinking panel light beside the black keys, and the edge followed it
    let nearest = Infinity;
    let gradientOffset = 0;
    let previousGradient = 0;
    for (let s = 1; s < steps; s += 1) {
      const gradient = Math.abs(values[s] - values[s - 1]) / SEARCH_STEP;
      const next =
        s + 1 < steps ? Math.abs(values[s + 1] - values[s]) / SEARCH_STEP : 0;
      const offset = -SEARCH_PX + s * SEARCH_STEP - SEARCH_STEP / 2;
      if (
        gradient > MIN_RESPONSE &&
        gradient >= previousGradient &&
        gradient >= next &&
        Math.abs(offset) < nearest
      ) {
        nearest = Math.abs(offset);
        gradientOffset = offset;
      }
      previousGradient = gradient;
    }
    const keyed = bestContrast > MIN_END_CONTRAST;
    if (keyed || Number.isFinite(nearest)) {
      const offset = keyed ? contrastOffset : gradientOffset;
      snapped.push({ x: p.x + offset * n.x, y: p.y + offset * n.y });
    }
  }
  return snapped;
}

// The boundary's own box: centred on the points, along their principal axis, clockwise.
export function principalBox(points: Point[]): Point[] {
  const n = points.length;
  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x / n;
    my += p.y / n;
  }
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of points) {
    sxx += (p.x - mx) * (p.x - mx);
    sxy += (p.x - mx) * (p.y - my);
    syy += (p.y - my) * (p.y - my);
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  const along = { x: Math.cos(angle), y: Math.sin(angle) };
  const across = { x: -along.y, y: along.x };
  let u0 = Infinity;
  let u1 = -Infinity;
  let v0 = Infinity;
  let v1 = -Infinity;
  for (const p of points) {
    const u = (p.x - mx) * along.x + (p.y - my) * along.y;
    const v = (p.x - mx) * across.x + (p.y - my) * across.y;
    u0 = Math.min(u0, u);
    u1 = Math.max(u1, u);
    v0 = Math.min(v0, v);
    v1 = Math.max(v1, v);
  }
  const corner = (u: number, v: number): Point => ({
    x: mx + u * along.x + v * across.x,
    y: my + u * along.y + v * across.y,
  });
  return clockwise(
    canonicalQuad([
      corner(u0, v0),
      corner(u1, v0),
      corner(u1, v1),
      corner(u0, v1),
    ]),
  );
}

// the boundary's box, in the coarse quad's corner order: its ends cut the keys square,
// where the coarse quad's own ends can be a diagonal spike dragging long-edge points onto an end
function assignmentBox(points: Point[], coarse: Point[]): Point[] {
  const box = principalBox(points);
  const near = Math.hypot(box[0].x - coarse[0].x, box[0].y - coarse[0].y);
  const far = Math.hypot(box[2].x - coarse[0].x, box[2].y - coarse[0].y);
  return near > far ? [box[2], box[3], box[0], box[1]] : box;
}

// Which edge of the quad each point is evidence for: the nearest one.
function assignEdges(points: Point[], quad: Point[]): Int32Array {
  return segmentDistances(points, quad).which;
}

// How many times wider one end of the keybed is drawn than the other.
function endRatio(quad: Point[]): number {
  const near = Math.hypot(quad[3].x - quad[0].x, quad[3].y - quad[0].y);
  const far = Math.hypot(quad[1].x - quad[2].x, quad[1].y - quad[2].y);
  return Math.max(near, far) / Math.max(Math.min(near, far), 1e-6);
}

function lineThrough(points: Point[]): { c: Point; d: Point } {
  const n = points.length;
  const c = {
    x: points.reduce((s, p) => s + p.x, 0) / n,
    y: points.reduce((s, p) => s + p.y, 0) / n,
  };
  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (const p of points) {
    sxx += (p.x - c.x) * (p.x - c.x);
    sxy += (p.x - c.x) * (p.y - c.y);
    syy += (p.y - c.y) * (p.y - c.y);
  }
  const angle = 0.5 * Math.atan2(2 * sxy, sxx - syy);
  return { c, d: { x: Math.cos(angle), y: Math.sin(angle) } };
}

function intersection(
  c1: Point,
  d1: Point,
  c2: Point,
  d2: Point,
): Point | null {
  const det = d1.x * -d2.y - d1.y * -d2.x;
  if (Math.abs(det) < 1e-9) {
    return null;
  }
  const t = ((c2.x - c1.x) * -d2.y - (c2.y - c1.y) * -d2.x) / det;
  return { x: c1.x + t * d1.x, y: c1.y + t * d1.y };
}

export interface Frame {
  gray: Float32Array;
  width: number;
  height: number;
}

// the keys' edge at this end, read from the picture: the mask's own end evidence covers
// only part of the end, so a line through it would extrapolate wrongly to the far corner
function endFromPixels(
  smooth: Float32Array,
  frame: Frame,
  quad: Point[],
  end: number,
): { c: Point; d: Point } | null {
  const a = quad[end];
  const b = quad[(end + 1) % 4];
  const along = { x: b.x - a.x, y: b.y - a.y };
  const length = Math.hypot(along.x, along.y);
  if (length < 4) {
    return null;
  }
  let outward = { x: -along.y / length, y: along.x / length };
  const centre = {
    x: quad.reduce((s, p) => s + p.x, 0) / 4,
    y: quad.reduce((s, p) => s + p.y, 0) / 4,
  };
  const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
  if (outward.x * (mid.x - centre.x) + outward.y * (mid.y - centre.y) < 0) {
    outward = { x: -outward.x, y: -outward.y };
  }
  const steps = Math.round((2 * END_SEARCH_PX) / SEARCH_STEP) + 1;
  const reach = Math.round(END_CONTRAST_PX / SEARCH_STEP);
  const values = new Float64Array(steps);
  const found: Point[] = [];
  const foundT: number[] = [];
  const foundOffset: number[] = [];
  for (let s = 0; s < END_SAMPLES; s += 1) {
    const t = END_TRIM + ((1 - 2 * END_TRIM) * s) / (END_SAMPLES - 1);
    const base = { x: a.x + t * along.x, y: a.y + t * along.y };
    for (let k = 0; k < steps; k += 1) {
      const offset = -END_SEARCH_PX + k * SEARCH_STEP;
      values[k] = bilinear(
        smooth,
        frame.width,
        frame.height,
        base.x + offset * outward.x,
        base.y + offset * outward.y,
      );
    }
    let best = -Infinity;
    let bestOffset = 0;
    for (let k = reach; k < steps - reach; k += 1) {
      let inside = 0;
      let outside = 0;
      for (let r = 1; r <= reach; r += 1) {
        inside += values[k - r] / reach;
        outside += values[k + r] / reach;
      }
      const contrast = inside - outside;
      if (inside >= MIN_KEY_BRIGHTNESS && contrast > best) {
        best = contrast;
        bestOffset = -END_SEARCH_PX + k * SEARCH_STEP;
      }
    }
    if (best > MIN_END_CONTRAST) {
      found.push({
        x: base.x + bestOffset * outward.x,
        y: base.y + bestOffset * outward.y,
      });
      foundT.push(t);
      foundOffset.push(bestOffset);
    }
  }
  if (found.length < MIN_END_SAMPLES) {
    return null;
  }
  // the picture shifts the end to the keys' edge always, but sets its tilt only when samples
  // span most of it; the far end's short bright span otherwise swung the line frame to frame
  const meanT = foundT.reduce((s, t) => s + t, 0) / foundT.length;
  const shift = median(Float64Array.from(foundOffset));
  const parallel = {
    c: {
      x: a.x + meanT * along.x + shift * outward.x,
      y: a.y + meanT * along.y + shift * outward.y,
    },
    d: { x: along.x / length, y: along.y / length },
  };
  if (Math.max(...foundT) - Math.min(...foundT) < MIN_END_SPAN) {
    return parallel;
  }
  const first = lineThrough(found);
  const residual = found.map((p) =>
    Math.abs(-(p.x - first.c.x) * first.d.y + (p.y - first.c.y) * first.d.x),
  );
  const cut = Math.max(2 * median(Float64Array.from(residual)), 1);
  const kept = found.filter((_, i) => residual[i] <= cut);
  if (kept.length < MIN_END_SAMPLES) {
    return parallel;
  }
  const line = lineThrough(kept);
  const alignment = (line.d.x * along.x + line.d.y * along.y) / length;
  return Math.abs(alignment) < MIN_END_ALIGNMENT ? parallel : line;
}

// the ends are redrawn where the picture shows the keys stopping, cut by the rectangle's
// long edges, since those stay the rectangle's strength while its own end tilt does not
export function refineEnds(frame: Frame, quad: Point[]): Point[] {
  const smooth = blur(frame.gray, frame.width, frame.height);
  const out = quad.map((p) => ({ ...p }));
  const longEdges: Record<number, { c: Point; d: Point }> = {
    0: {
      c: quad[0],
      d: { x: quad[1].x - quad[0].x, y: quad[1].y - quad[0].y },
    },
    2: {
      c: quad[3],
      d: { x: quad[2].x - quad[3].x, y: quad[2].y - quad[3].y },
    },
  };
  const ends: [number, Record<number, number>][] = [
    [1, { 1: 0, 2: 2 }],
    [3, { 0: 0, 3: 2 }],
  ];
  for (const [end, corners] of ends) {
    const line = endFromPixels(smooth, frame, quad, end);
    if (!line) {
      continue;
    }
    for (const [corner, longEdge] of Object.entries(corners)) {
      const edge = longEdges[longEdge];
      const moved = intersection(line.c, line.d, edge.c, edge.d);
      const k = Number(corner);
      if (
        moved &&
        Math.hypot(moved.x - quad[k].x, moved.y - quad[k].y) <= MAX_END_SHIFT_PX
      ) {
        out[k] = moved;
      }
    }
  }
  return out;
}

// the rectangle pose that best explains the boundary points, or null when the evidence
// can't pin one down; hold a focal across frames, since one image trades it against the far end
export function fitRectangle(
  points: Point[],
  coarseQuad: Point[],
  width: number,
  height: number,
  focal?: number,
  frame?: Frame,
): RectFit | null {
  if (points.length < 8) {
    lastDecline = `only ${points.length} boundary points`;
    return null;
  }
  const cx = width / 2;
  const cy = height / 2;
  // every start is solved and the closest-explaining outline wins: a spiky-mask start can
  // land 2 px from the boundary but 270 px off at the far end; the box lands within 7 px
  let coarse = clockwise(canonicalQuad(coarseQuad));
  let solution: FocalSolution | null = null;
  let fitted: Point[] = coarse;
  let best = Infinity;
  for (const start of [coarse, principalBox(points)]) {
    const attempt =
      focal === undefined
        ? scanFocal(points, start, width, cx, cy)
        : { ...fitAtFocal(points, start, focal, cx, cy), focal };
    if (!Number.isFinite(attempt.cost)) {
      lastDecline = "initial pose not in front of the camera";
      continue;
    }
    const quad = project(attempt.pose, attempt.focal, cx, cy);
    const residual = median(segmentDistances(points, quad).distance);
    if (residual < best) {
      best = residual;
      coarse = start;
      solution = attempt;
      fitted = quad;
    }
  }
  // the fit answers to the boundary, not the coarse quad: a coarse quad from a sliver of mask
  // is garbage exactly when the fit matters most, while the boundary is real evidence
  if (!solution || best > MAX_RESIDUAL_PX) {
    if (solution) {
      lastDecline = `boundary ${best.toFixed(1)} px off the fit`;
    }
    return null;
  }
  // both long edges plus one end are enough once the focal is held, since the known length
  // places the missing end; unheld, every edge must be seen or focal and end trade off freely
  const counts = [0, 0, 0, 0];
  for (const edge of assignEdges(points, assignmentBox(points, coarse))) {
    counts[edge] += 1;
  }
  const seen = counts.map((n) => n >= MIN_EDGE_POINTS);
  const endsSeen = (seen[1] ? 1 : 0) + (seen[3] ? 1 : 0);
  if (!(seen[0] && seen[2]) || endsSeen < (focal === undefined ? 2 : 1)) {
    lastDecline = `edge points ${counts.join("/")}`;
    return null;
  }
  const ratio = endRatio(fitted);
  if (ratio > MAX_END_RATIO) {
    lastDecline = `one end drawn ${ratio.toFixed(1)}x the other`;
    return null;
  }
  return {
    quad: frame ? refineEnds(frame, fitted) : fitted,
    rectangle: fitted,
    cost: solution.cost,
    focal: solution.focal,
  };
}
