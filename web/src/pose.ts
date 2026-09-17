import { findHomography, type Point } from "./homography";
import { WHITE_COUNT } from "./keys";

export const WHITE_KEY_MM = 23.5;
// visible key depth of the instrument in front of the camera, measured from its labelled
// frames: the pose residual over 16 snaps bottoms out at a span-to-depth ratio of 7.2
export const KEYBED_DEPTH_MM = 118;
export const WHITE_KEY_COUNT = WHITE_COUNT;
export const DEPTH_UNITS = KEYBED_DEPTH_MM / WHITE_KEY_MM;
const SCAN_SAMPLES = 200;
const GOLDEN_ITERATIONS = 100;
const GOLDEN_RATIO = (Math.sqrt(5) - 1) / 2;

/** A point of the keybed's own space, in white-key widths: u along the keys, v
 * across their depth from the player's edge, w standing off the plane. */
export interface Vector3 {
  u: number;
  v: number;
  w: number;
}

export interface PlanePose {
  focal: number;
  rotation: number[][];
  translation: number[];
  worldWidthMm: number;
  residual: number;
}

// the keybed's depth in white-key widths, the default until the user measures it from a
// top view, the one view where aspect isn't tangled with focal and pose
let depthUnits = DEPTH_UNITS;

export function setKeybedDepth(units: number): void {
  depthUnits = units;
}

export function keybedDepth(): number {
  return depthUnits;
}

// the camera's focal as a fraction of the frame width, a webcam's 65 degree lens until the
// user measures it from an oblique view, where the two vanishing points fix it
let cameraFocal = 0.75;

export function setCameraFocal(fraction: number): void {
  cameraFocal = fraction;
}

export function cameraFocalFraction(): number {
  return cameraFocal;
}

/** The keybed as the fit and the pose both see it, so a depth the user measures
 * moves them together. Corner 0 to 1 spans the keys and 1 to 2 the depth, with
 * the player at the near edge. */
export function worldCorners(): Point[] {
  return [
    { x: 0, y: 0 },
    { x: WHITE_KEY_COUNT, y: 0 },
    { x: WHITE_KEY_COUNT, y: depthUnits },
    { x: 0, y: depthUnits },
  ];
}

function edgeLength(quad: Point[], index: number): number {
  return Math.hypot(quad[index].x - quad[0].x, quad[index].y - quad[0].y);
}

function roll(quad: Point[], by: number): Point[] {
  return quad.map((_, i) => quad[(i + by) % quad.length]);
}

// The world frame puts the white keys on edge 0->1, so an image quad has to agree before it solves.
// The remaining 180 degree choice is the dragger's: handle 1 to 2 runs along the black keys.
export function canonicalQuad(quad: Point[]): Point[] {
  return edgeLength(quad, 1) >= edgeLength(quad, 3) ? quad : roll(quad, 1);
}

function inverseCalibrationHomography(
  focal: number,
  width: number,
  height: number,
  h: number[],
): number[][] {
  const cx = width / 2;
  const cy = height / 2;
  const b: number[][] = [[], [], []];
  for (let col = 0; col < 3; col += 1) {
    b[0][col] = (h[col] - cx * h[col + 6]) / focal;
    b[1][col] = (h[col + 3] - cy * h[col + 6]) / focal;
    b[2][col] = h[col + 6];
  }
  return b;
}

function column(m: number[][], j: number): number[] {
  return [m[0][j], m[1][j], m[2][j]];
}

function norm(v: number[]): number {
  return Math.hypot(v[0], v[1], v[2]);
}

function unit(v: number[]): number[] {
  const n = norm(v);
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a: number[], b: number[]): number[] {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function orthogonalityResidual(r1: number[], r2: number[]): number {
  const dot = r1[0] * r2[0] + r1[1] * r2[1] + r1[2] * r2[2];
  return Math.abs(dot) + Math.abs(1 - norm(cross(r1, r2)));
}

// a rotation's columns are orthogonal AND the same length. The length test is what refuses a
// frontal view of the wrong aspect, where orthogonality alone is satisfied by any rectangle.
function poseResidual(b: number[][]): number {
  const c0 = column(b, 0);
  const c1 = column(b, 1);
  const n0 = norm(c0);
  const n1 = norm(c1);
  return (
    orthogonalityResidual(unit(c0), unit(c1)) +
    Math.abs(1 - Math.min(n0, n1) / Math.max(n0, n1))
  );
}

function goldenSection(
  fn: (f: number) => number,
  lo: number,
  hi: number,
): number {
  let a = lo;
  let b = hi;
  let c = b - GOLDEN_RATIO * (b - a);
  let d = a + GOLDEN_RATIO * (b - a);
  let fc = fn(c);
  let fd = fn(d);
  for (let i = 0; i < GOLDEN_ITERATIONS; i += 1) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - GOLDEN_RATIO * (b - a);
      fc = fn(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + GOLDEN_RATIO * (b - a);
      fd = fn(d);
    }
  }
  return (a + b) / 2;
}

export function estimateFocal(
  imageCorners: Point[],
  width: number,
  height: number,
): number {
  const h = findHomography(worldCorners(), canonicalQuad(imageCorners));
  const lo = 0.3 * width;
  const hi = 3 * width;
  const residual = (f: number): number => {
    return poseResidual(inverseCalibrationHomography(f, width, height, h));
  };
  const step = (hi - lo) / SCAN_SAMPLES;
  let bestFocal = lo;
  let bestResidual = residual(lo);
  for (let i = 1; i <= SCAN_SAMPLES; i += 1) {
    const f = lo + i * step;
    const value = residual(f);
    if (value < bestResidual) {
      bestFocal = f;
      bestResidual = value;
    }
  }
  const goldenFocal = goldenSection(residual, lo, hi);
  const goldenResidual = residual(goldenFocal);
  return goldenResidual < bestResidual ? goldenFocal : bestFocal;
}

export function solvePose(
  imageCorners: Point[],
  width: number,
  height: number,
): PlanePose {
  const h = findHomography(worldCorners(), canonicalQuad(imageCorners));
  const focal = estimateFocal(imageCorners, width, height);
  const b = inverseCalibrationHomography(focal, width, height, h);
  const scale = norm(column(b, 0));
  const r1 = [b[0][0] / scale, b[1][0] / scale, b[2][0] / scale];
  const r2 = [b[0][1] / scale, b[1][1] / scale, b[2][1] / scale];
  const r3 = cross(r1, r2);
  const metricScale = WHITE_KEY_MM / scale;
  return {
    focal,
    rotation: [
      [r1[0], r2[0], r3[0]],
      [r1[1], r2[1], r3[1]],
      [r1[2], r2[2], r3[2]],
    ],
    translation: [
      b[0][2] * metricScale,
      b[1][2] * metricScale,
      b[2][2] * metricScale,
    ],
    worldWidthMm: WHITE_KEY_COUNT * WHITE_KEY_MM,
    residual: poseResidual(b),
  };
}

/** Where a point of the keybed's own space lands in the picture. u runs along
 * the keys and v across their depth, both in white-key widths, and w stands off
 * the plane, which is where anything drawn over the instrument lives. */
export function projectSpace(
  pose: PlanePose,
  u: number,
  v: number,
  w: number,
  width: number,
  height: number,
): Point {
  const xmm = u * WHITE_KEY_MM;
  const ymm = v * WHITE_KEY_MM;
  const zmm = w * WHITE_KEY_MM;
  const r = pose.rotation;
  const t = pose.translation;
  const at = (row: number): number =>
    r[row][0] * xmm + r[row][1] * ymm + r[row][2] * zmm + t[row];
  const zc = at(2);
  return {
    x: width / 2 + (pose.focal * at(0)) / zc,
    y: height / 2 + (pose.focal * at(1)) / zc,
  };
}

export function projectPoint(
  pose: PlanePose,
  u: number,
  v: number,
  width: number,
  height: number,
): Point {
  return projectSpace(pose, u, v, 0, width, height);
}

/** How far in front of the camera a point of the keybed's space sits, in
 * white-key widths. Anything at or behind zero has no picture to be drawn in. */
export function spaceDepth(
  pose: PlanePose,
  u: number,
  v: number,
  w: number,
): number {
  const r = pose.rotation;
  return (
    (r[2][0] * u * WHITE_KEY_MM +
      r[2][1] * v * WHITE_KEY_MM +
      r[2][2] * w * WHITE_KEY_MM +
      pose.translation[2]) /
    WHITE_KEY_MM
  );
}

/** Where the camera itself stands in the keybed's space, in white-key widths.
 * What is drawn above the keys is aimed at this. */
export function cameraPosition(pose: PlanePose): Vector3 {
  const r = pose.rotation;
  const t = pose.translation;
  const at = (col: number): number =>
    -(r[0][col] * t[0] + r[1][col] * t[1] + r[2][col] * t[2]) / WHITE_KEY_MM;
  return { u: at(0), v: at(1), w: at(2) };
}
