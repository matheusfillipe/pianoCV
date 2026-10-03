import { split } from "./board";
import {
  applyHomography,
  findHomography,
  type Point,
  solve,
} from "./homography";
import type { KeyMatcher } from "./keymatch";
import { isBlack, keyUnits, whiteIndex } from "./keys";
import type { KeySegmenter } from "./keyseg";
import type { Size } from "./keyspace";
import { type Bar, blackKeyDepth } from "./keyspace";
import { keybedDepth, projectSpace, solvePose, WHITE_KEY_MM } from "./pose";

declare global {
  interface Window {
    // the most recent rectify and what it found, for the lab to read a live page off rather
    // than guess at the pipeline's internal state
    pianocvKeyStrip?: {
      width: number;
      height: number;
      data: Uint8ClampedArray;
    };
    pianocvKeyRead?: KeyRead;
    pianocvKeyEvidence?: KeyEvidence | null;
    // every read of the held keyboard so far with how well it fit its picture
    pianocvReadFits?: { confidence: number; fit: number; chosen: boolean }[];
  }
}

export const STRIP_WIDTH = 1200;
export const STRIP_HEIGHT = 150;

/** The far band is where black keys live; the near band is the exposed strip of every white
 * key in front of the black keys, where the gap between two white keys always shows. */
const BLACK_BAND: readonly [number, number] = [0.12, 0.42];
const SEPARATOR_BAND: readonly [number, number] = [0.8, 0.96];

const MIN_RUN_WIDTH_PX = 4;
const MIN_RUNS = 3;
const MIN_WHITE_KEYS = 3;
/** No real keyboard, concert grand included, carries more white keys than this; a walk past it
 * has been fooled by noise into taking steps far smaller than any real key. */
const MOST_WHITE_KEYS = 60;
const MIN_BLACK_SEPARATION = 0.3;
const MIN_SEPARATOR_SEPARATION = 0.15;

/** How far the walk may snap to real evidence, as a fraction of the current local key width. */
const WALK_TOLERANCE_FRACTION = 0.35;
/** How far a boundary may move once the walk is done, small enough it can only refine a
 * boundary already close to right, never reach past a neighbouring key. */
const MICRO_TOLERANCE_FRACTION = 0.18;
const WIDTH_SMOOTH_ALPHA = 0.5;

/** The mean of the three within-group semitone gaps (C#-D#, F#-G#, G#-A#) a chromatic keyboard
 * carries, used only to turn a measured pixel gap into a first guess at the local key width. */
const WITHIN_GROUP_UNIT_GAP = 1.07;

const LETTERS = ["C", "D", "E", "F", "G", "A", "B"];
/** E and B have no black key between them and the next white key. */
const NO_BLACK_AFTER = new Set([2, 6]);

const FALLBACK_BLACK_OFFSET = 0.62;
const FALLBACK_BLACK_WIDTH = 0.56;

const WHITE_PITCH_CLASSES = [0, 2, 4, 5, 7, 9, 11];
/** Any octave serves: the template only ever uses positions relative to its first white key. */
const TEMPLATE_OCTAVE_BASE = 60;
const TEMPLATE_MIN_POINTS = 8;
const TEMPLATE_ROUNDS = 3;
/** How far evidence may sit from the fitted keyboard, in white-key widths, before we treat it as
 * a misread rather than let it pull the whole board. */
const TEMPLATE_OUTLIER_KEYS = 0.3;
/** How close a dip or black run must land to a template position to be taken as that key. */
const TEMPLATE_SNAP_KEYS = 0.3;
/** How close the strip's own edge must land to a key boundary to count as the board's end. A mask
 * that stopped on the end lands within a tenth of a key; one that ran most of a key past it can
 * still land within the looser separator tolerance and invent one more key. */
const STRIP_EDGE_SNAP_KEYS = 0.12;
/** How much of a black key's width we skip at each side when reading where it ends, so the gap
 * beside it never counts. */
const BLACK_END_INSET = 0.25;
/** The depth range, as strip fractions, where a black key can end on any real keyboard. */
const BLACK_END_SEARCH: readonly [number, number] = [0.4, 0.92];
const BLACK_END_MIN_KEYS = 5;
/** Heights a black key's top may stand above the white keys, either way round since which side
 * of the plane the camera sits on depends on how the corners wind; the fit picks the one that
 * lets white and black evidence agree. */
export const RAISE_CANDIDATES_MM = [0, 4, 8, 12, 16];
/** How many keys a board may carry on past the strip's edge when white keys run right up to it:
 * the mask can stop short of the board's end, and keys past the strip are out of sight rather
 * than absent. */
const UNSEEN_KEYS = 3;
/** How much of a white key's width we skip at each side when reading its front's brightness. */
const WHITE_FRONT_INSET = 0.2;
/** How much of a key must lie inside the strip before its front counts as evidence. */
const WHITE_FRONT_MIN_INSIDE = 0.4;
/** How bright, against its nearest seen white keys, a front must be to count as one more key. */
const WHITE_FRONT_SHARE = 0.6;
const MAX_EXTEND_KEYS = 6;

/** The sizes keyboards are built in, told apart by the letter of their first white key, with the
 * MIDI note that first key plays. */
export const STANDARD_BOARDS = [
  { phase: "C", whiteKeys: 29, lowestPitch: 36 },
  { phase: "C", whiteKeys: 36, lowestPitch: 36 },
  { phase: "E", whiteKeys: 45, lowestPitch: 28 },
  { phase: "A", whiteKeys: 52, lowestPitch: 21 },
] as const;
const BLACK_BAND_MIDDLE = (BLACK_BAND[0] + BLACK_BAND[1]) / 2;

export type SourceImage = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
};

export type Strip = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
};

/** Where the keys begin, as a depth fraction at each end of the board, so the trimmed edge can
 * follow a case boundary that is deeper at one end. */
export type FarEdge = {
  readonly left: number;
  readonly right: number;
};

export type DetectedKey = {
  readonly bar: Bar;
  readonly black: boolean;
  /** Semitones above the board's first white key, which is what maps a MIDI note onto it. */
  readonly semitone: number;
};

export type KeyRead =
  | {
      readonly kind: "read";
      readonly whiteKeys: number;
      readonly totalKeys: number;
      /** The white key the strip's left edge starts on, C to B. */
      readonly phase: string;
      readonly confidence: number;
      /** How many white keys wide the strip was, which is the rectangle the pose is solved
       * against; wider than the board when the mask ran past its ends. */
      readonly stripKeys: number;
      /** How far the black keys' tops stand off the keybed, as the fit measured it, in the
       * pose's own sign. */
      readonly blackRaiseMm: number;
      /** How far, in strip pixels along the keys, the black keys the strip shows sit from where
       * the pose puts them, as a line over the strip; the pose's lens and tilt are estimated from
       * one rectangle, so we pin the drawn black keys to what the evidence shows. */
      readonly blackShift: Line;
      /** The corners, in strip pixels of the quad the read was asked about, of the outline the
       * keys were actually read on, once its ends were squared to the keys. */
      readonly outline: readonly Point[];
      readonly keyAt: (index: number) => DetectedKey | null;
    }
  | {
      readonly kind: "unsure";
      readonly reason: string;
      readonly confidence: number;
    };

function stripCorners(width: number, height: number): Point[] {
  return [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: width, y: height },
    { x: 0, y: height },
  ];
}

export function sampleBilinear(
  source: SourceImage,
  x: number,
  y: number,
): [number, number, number] | null {
  if (x < 0 || y < 0 || x > source.width - 1 || y > source.height - 1) {
    return null;
  }
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(source.width - 1, x0 + 1);
  const y1 = Math.min(source.height - 1, y0 + 1);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xx: number, yy: number, c: number): number =>
    source.data[(yy * source.width + xx) * 4 + c];
  const lerp = (c: number): number => {
    const top = at(x0, y0, c) * (1 - fx) + at(x1, y0, c) * fx;
    const bottom = at(x0, y1, c) * (1 - fx) + at(x1, y1, c) * fx;
    return top * (1 - fy) + bottom * fy;
  };
  return [lerp(0), lerp(1), lerp(2)];
}

/** Warps the held quad flat: the far edge, where the black keys end, lands at the top, and the
 * player's edge lands at the bottom, so the rest of this module can work in undistorted pixels. */
export function rectifyStrip(
  source: SourceImage,
  quad: readonly Point[],
  width = STRIP_WIDTH,
  height = STRIP_HEIGHT,
): Strip | null {
  if (quad.length < 4) {
    return null;
  }
  const quadPx = quad
    .slice(0, 4)
    .map((p) => ({ x: p.x * source.width, y: p.y * source.height }));
  const toSource = findHomography(stripCorners(width, height), quadPx);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = applyHomography(toSource, x + 0.5, y + 0.5);
      const rgb = sampleBilinear(source, p.x, p.y);
      const at = (y * width + x) * 4;
      if (rgb === null) {
        continue;
      }
      data[at] = rgb[0];
      data[at + 1] = rgb[1];
      data[at + 2] = rgb[2];
      data[at + 3] = 255;
    }
  }
  return { width, height, data };
}

/** Where a strip point at (x, y) appears in the same strip once raised `raiseMm` off the keybed:
 * the flat strip shows anything standing above the keys displaced by parallax. */
export type Lift = (x: number, y: number, raiseMm: number) => number;
/** A lift for a board of `whiteKeys`, whose width sets the aspect the pose is solved against. */
export type LiftFor = (whiteKeys: number) => Lift;

function boardWorld(whiteKeys: number, depth: number): Point[] {
  return [
    { x: 0, y: 0 },
    { x: whiteKeys, y: 0 },
    { x: whiteKeys, y: depth },
    { x: 0, y: depth },
  ];
}

export type LinePoint = { readonly x: number; readonly y: number };
export type Line = {
  readonly meanX: number;
  readonly meanY: number;
  readonly slope: number;
};

export function fitLine(points: readonly LinePoint[]): Line {
  const meanX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const meanY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  let numerator = 0;
  let denominator = 0;
  for (const point of points) {
    numerator += (point.x - meanX) * (point.y - meanY);
    denominator += (point.x - meanX) ** 2;
  }
  return {
    meanX,
    meanY,
    slope: denominator === 0 ? 0 : numerator / denominator,
  };
}

export function lineAt(line: Line, x: number): number {
  return line.meanY + line.slope * (x - line.meanX);
}

export function lineResidual(line: Line, point: LinePoint): number {
  return Math.abs(point.y - lineAt(line, point.x));
}

/** A strip's own corners, in strip pixels: the outline of a read made on the quad as given. */
export const PLAIN_OUTLINE: readonly Point[] = stripCorners(
  STRIP_WIDTH,
  STRIP_HEIGHT,
);

/** The quad whose strip corners are `outline` in the strip of `quad`. */
export function outlineQuad(
  quad: readonly Point[],
  outline: readonly Point[],
): Point[] {
  const toQuad = findHomography(
    stripCorners(STRIP_WIDTH, STRIP_HEIGHT),
    quad.slice(0, 4).map((p) => ({ x: p.x, y: p.y })),
  );
  return outline.map((p) => applyHomography(toQuad, p.x, p.y));
}

function framePixels(quad: readonly Point[], frame: Size): Point[] {
  return quad
    .slice(0, 4)
    .map((p) => ({ x: p.x * frame.width, y: p.y * frame.height }));
}

export function liftFor(quad: readonly Point[], frame: Size): LiftFor {
  const corners = framePixels(quad, frame);
  const toStrip = findHomography(
    corners,
    stripCorners(STRIP_WIDTH, STRIP_HEIGHT),
  );
  return (whiteKeys) => {
    const depth = keybedDepth();
    const pose = solvePose(
      corners,
      frame.width,
      frame.height,
      boardWorld(whiteKeys, depth),
    );
    return (x, y, raiseMm) => {
      const image = projectSpace(
        pose,
        (x / STRIP_WIDTH) * whiteKeys,
        (y / STRIP_HEIGHT) * depth,
        raiseMm / WHITE_KEY_MM,
        frame.width,
        frame.height,
      );
      return applyHomography(toStrip, image.x, image.y).x;
    };
  };
}

/** The visible faces of every key in the current frame, as frame fractions: white keys flat on
 * the keybed, and each black key as its raised top plus the front face that drops to the white
 * keys, since no outline flat on the keybed can cover a key standing above it. */
export function projectKeyFaces(
  read: KeyRead,
  quad: readonly Point[],
  frame: Size,
): DetectedKey[] {
  if (
    read.kind !== "read" ||
    quad.length < 4 ||
    frame.width === 0 ||
    frame.height === 0
  ) {
    return [];
  }
  const outlined = outlineQuad(quad, read.outline);
  const depth = keybedDepth();
  const world = boardWorld(read.stripKeys, depth);
  const pose = solvePose(
    framePixels(outlined, frame),
    frame.width,
    frame.height,
    world,
  );
  const raise = read.blackRaiseMm / WHITE_KEY_MM;
  const pinned = (p: Point): Point => ({
    x: p.x + lineAt(read.blackShift, p.x),
    y: p.y,
  });
  const lift = (p: Point, w: number): Point => {
    const image = projectSpace(
      pose,
      (p.x / STRIP_WIDTH) * read.stripKeys,
      (p.y / STRIP_HEIGHT) * depth,
      w,
      frame.width,
      frame.height,
    );
    return { x: image.x / frame.width, y: image.y / frame.height };
  };
  const faces: DetectedKey[] = [];
  for (let index = 0; index < read.totalKeys; index += 1) {
    const key = read.keyAt(index);
    if (key === null) {
      continue;
    }
    const { black, semitone } = key;
    const [a, b, c, d] = black ? key.bar.map(pinned) : key.bar;
    if (!black) {
      faces.push({
        black,
        semitone,
        bar: [lift(a, 0), lift(b, 0), lift(c, 0), lift(d, 0)],
      });
      continue;
    }
    faces.push({
      black,
      semitone,
      bar: [lift(a, raise), lift(b, raise), lift(c, raise), lift(d, raise)],
    });
    faces.push({
      black,
      semitone,
      bar: [lift(d, raise), lift(c, raise), lift(c, 0), lift(d, 0)],
    });
  }
  return faces;
}

function luminance(data: Uint8ClampedArray, at: number): number {
  return 0.299 * data[at] + 0.587 * data[at + 1] + 0.114 * data[at + 2];
}

const FAR_EDGE_CONTRAST_MIN = 60;
const FAR_EDGE_BRIGHTNESS_MIN = 60;
const MIN_TRIM_FRACTION = 0.02;
const MAX_TRIM_FRACTION = 0.35;
const MIN_KEY_ROWS = 20;
const NEAR_EDGE_CHECK_ROWS = 6;
const NEAR_EDGE_MIN_HITS = 4;
const FAR_EDGE_BANDS = 12;
const FAR_EDGE_MIN_BANDS = 8;
/** How many rows either side of a candidate edge we average, wide enough to ride out a blur. */
const FAR_EDGE_STEP_ROWS = 3;
/** The smallest case-to-keys brightness step we accept as an edge rather than sensor noise. */
const FAR_EDGE_STEP_MIN = 10;
/** The step as a share of the brightness past it: the end of the board furthest from the light,
 * where black keys crowd the back, climbs only a few levels, so an absolute step alone rejects it. */
const FAR_EDGE_STEP_SHARE = 0.2;
/** How many rows either side of the edge we average to judge that step, past most blurs. */
const FAR_EDGE_PLATEAU_ROWS = 10;
/** How close to the steepest rise a row must be to count as part of the same blurred edge. */
const FAR_EDGE_PEAK_SHARE = 0.9;
/** How far a column band may sit from the fitted key boundary before we treat it as a misread. */
const FAR_EDGE_OUTLIER_SPAN = 0.06;

function rowIsKeys(strip: Strip, y: number): boolean {
  let min = 255;
  let max = 0;
  let sum = 0;
  let count = 0;
  for (let x = 0; x < strip.width; x += 1) {
    const at = (y * strip.width + x) * 4;
    if (strip.data[at + 3] === 0) {
      continue;
    }
    const value = luminance(strip.data, at);
    min = Math.min(min, value);
    max = Math.max(max, value);
    sum += value;
    count += 1;
  }
  if (count === 0) {
    return false;
  }
  return (
    max - min > FAR_EDGE_CONTRAST_MIN && sum / count > FAR_EDGE_BRIGHTNESS_MIN
  );
}

function bandBrightness(
  strip: Strip,
  y: number,
  from: number,
  to: number,
): number | null {
  let sum = 0;
  let count = 0;
  for (let x = from; x < to; x += 1) {
    const at = (y * strip.width + x) * 4;
    if (strip.data[at + 3] === 0) {
      continue;
    }
    sum += luminance(strip.data, at);
    count += 1;
  }
  return count === 0 ? null : sum / count;
}

function bandColumn(
  strip: Strip,
  from: number,
  to: number,
  rows: number,
): number[] | null {
  const brightness: number[] = [];
  for (let y = 0; y < rows; y += 1) {
    const value = bandBrightness(strip, y, from, to);
    if (value === null) {
      return null;
    }
    brightness.push(value);
  }
  return brightness;
}

/** The row where a column band steps from the dark case up onto the keys. */
function bandDepth(strip: Strip, from: number, to: number): number | null {
  const reach = Math.min(
    strip.height - FAR_EDGE_PLATEAU_ROWS,
    Math.ceil(MAX_TRIM_FRACTION * strip.height) + FAR_EDGE_STEP_ROWS,
  );
  const brightness = bandColumn(strip, from, to, reach + FAR_EDGE_PLATEAU_ROWS);
  return brightness === null
    ? null
    : steepestRise(brightness, FAR_EDGE_STEP_ROWS, reach);
}

/** The row between `from` and `to` where brightness steps up most steeply, or null when no step
 * is big enough to be an edge rather than texture. A camera blurs an edge over several rows, and
 * for a symmetric blur the steepest point is the true edge, where any fixed brightness level fires
 * early on a dark surface and late on a bright one. */
function steepestRise(
  brightness: readonly number[],
  from: number,
  to: number,
): number | null {
  const mean = (start: number, end: number): number => {
    const first = Math.max(0, start);
    const last = Math.min(brightness.length, end);
    return (
      brightness.slice(first, last).reduce((sum, v) => sum + v, 0) /
      (last - first)
    );
  };
  const rises: number[] = [];
  for (let y = from; y < to; y += 1) {
    rises.push(
      mean(y, y + FAR_EDGE_STEP_ROWS) - mean(y - FAR_EDGE_STEP_ROWS, y),
    );
  }
  if (rises.length === 0) {
    return null;
  }
  const peak = Math.max(...rises);
  if (peak <= 0) {
    return null;
  }
  // a blur wider than our window leaves a run of equally steep rows, so we take the middle of
  // that run; taking its first row would put the edge back at the start of the fade
  const top = rises.indexOf(peak);
  let first = top;
  while (first > 0 && rises[first - 1] >= peak * FAR_EDGE_PEAK_SHARE) {
    first -= 1;
  }
  let last = top;
  while (
    last < rises.length - 1 &&
    rises[last + 1] >= peak * FAR_EDGE_PEAK_SHARE
  ) {
    last += 1;
  }
  const edge = from + (first + last) / 2;
  const at = Math.round(edge);
  const after = mean(at, at + FAR_EDGE_PLATEAU_ROWS);
  const step = after - mean(at - FAR_EDGE_PLATEAU_ROWS, at);
  return step >= FAR_EDGE_STEP_MIN && step >= after * FAR_EDGE_STEP_SHARE
    ? edge
    : null;
}

/** Where the keys begin, measured per column band and fitted as a line, so a case boundary that
 * sits deeper at one end of the board is described by a tilt rather than collapsed to one depth
 * that can only ever be right in the middle. */
export function measureFarEdge(strip: Strip): FarEdge | null {
  const keyRows: boolean[] = [];
  for (let y = 0; y < strip.height; y += 1) {
    keyRows.push(rowIsKeys(strip, y));
  }
  if (keyRows.filter(Boolean).length < MIN_KEY_ROWS) {
    return null;
  }
  const nearHits = keyRows.slice(-NEAR_EDGE_CHECK_ROWS).filter(Boolean).length;
  if (nearHits < NEAR_EDGE_MIN_HITS) {
    return null;
  }
  const width = strip.width / FAR_EDGE_BANDS;
  const samples: { x: number; depth: number }[] = [];
  for (let band = 0; band < FAR_EDGE_BANDS; band += 1) {
    const from = Math.round(band * width);
    const to = Math.round((band + 1) * width);
    const first = bandDepth(strip, from, to);
    if (first !== null) {
      samples.push({
        x: (from + to) / 2 / strip.width,
        depth: first / strip.height,
      });
    }
  }
  if (samples.length < FAR_EDGE_MIN_BANDS) {
    return null;
  }
  // a real tilt puts the end bands furthest from the middle, so we judge a band by how far it
  // sits from the fitted line, never by how far it sits from the median depth
  const asPoint = (sample: { x: number; depth: number }): LinePoint => ({
    x: sample.x,
    y: sample.depth,
  });
  const first = fitLine(samples.map(asPoint));
  const kept = samples.filter(
    (sample) => lineResidual(first, asPoint(sample)) <= FAR_EDGE_OUTLIER_SPAN,
  );
  if (kept.length < FAR_EDGE_MIN_BANDS) {
    return null;
  }
  const { meanX, meanY: meanDepth, slope } = fitLine(kept.map(asPoint));
  // the fit only sees band centres, so reading it out at the very ends extrapolates, and a small
  // slope error lands entirely on the two corners; we never claim a depth past what we measured
  const depths = kept.map((sample) => sample.depth);
  const shallowest = Math.min(...depths);
  const deepest = Math.max(...depths);
  const at = (x: number): number =>
    Math.min(deepest, Math.max(shallowest, meanDepth + slope * (x - meanX)));
  const left = at(0);
  const right = at(1);
  const inRange = (value: number): boolean =>
    value >= MIN_TRIM_FRACTION && value <= MAX_TRIM_FRACTION;
  if (!inRange(left) || !inRange(right)) {
    return null;
  }
  return { left, right };
}

/** Moves the far edge (corners 0 and 1) toward the near edge (corners 2 and 3) by a fraction of
 * the depth, in the plane the quad is already drawn in. The near edge never moves: only the far
 * edge was ever measured onto the case instead of the keys. */
export function trimFarEdge(
  quad: readonly Point[],
  edge: FarEdge | null,
): Point[] {
  if (quad.length < 4 || edge === null) {
    return [...quad];
  }
  // the depths are measured on the keybed plane, and depth maps into the image projectively, so we
  // move the edge through the same homography the strip uses; lerping the image-space edges by an
  // equal fraction tilts the far edge, because the two ends foreshorten differently
  const toFrame = findHomography(
    stripCorners(STRIP_WIDTH, STRIP_HEIGHT),
    quad.slice(0, 4),
  );
  return [
    applyHomography(toFrame, 0, STRIP_HEIGHT * edge.left),
    applyHomography(toFrame, STRIP_WIDTH, STRIP_HEIGHT * edge.right),
    quad[2],
    quad[3],
  ];
}

function bandProfile(strip: Strip, fromFrac: number, toFrac: number): number[] {
  const fromY = Math.max(0, Math.floor(fromFrac * strip.height));
  const toY = Math.min(strip.height - 1, Math.ceil(toFrac * strip.height));
  const profile = new Array<number>(strip.width).fill(255);
  for (let x = 0; x < strip.width; x += 1) {
    let sum = 0;
    let count = 0;
    for (let y = fromY; y <= toY; y += 1) {
      const at = (y * strip.width + x) * 4;
      if (strip.data[at + 3] === 0) {
        continue;
      }
      sum += luminance(strip.data, at);
      count += 1;
    }
    if (count > 0) {
      profile[x] = sum / count;
    }
  }
  return profile;
}

export type Run = {
  readonly start: number;
  readonly end: number;
  readonly center: number;
};

function darkRuns(
  profile: readonly number[],
  at: number,
  minWidth: number,
): Run[] {
  const runs: Run[] = [];
  let start: number | null = null;
  for (let x = 0; x <= profile.length; x += 1) {
    const dark = x < profile.length && profile[x] < at;
    if (dark && start === null) {
      start = x;
    } else if (!dark && start !== null) {
      if (x - start >= minWidth) {
        runs.push({ start, end: x - 1, center: (start + x - 1) / 2 });
      }
      start = null;
    }
  }
  return runs;
}

function findDips(profile: readonly number[]): number[] {
  const { at, separation } = split(profile);
  if (separation < MIN_SEPARATOR_SEPARATION) {
    return [];
  }
  const dips: number[] = [];
  for (let x = 1; x < profile.length - 1; x += 1) {
    if (
      profile[x] < at &&
      profile[x] <= profile[x - 1] &&
      profile[x] <= profile[x + 1]
    ) {
      dips.push(x);
    }
  }
  return dips;
}

/** A between-group gap (say, D#-F#) is a run's real piano geometry: it is always flanked by
 * within-group gaps on both sides, and wider than either by a comfortable margin. Comparing
 * each gap only to its own neighbours, rather than to the whole strip's gaps at once, is what
 * lets this still work when the warp makes a gap on one side of the strip many pixels wider
 * than the same kind of gap on the other side. */
const LARGE_GAP_RATIO = 1.3;

function isLargeGap(gaps: readonly number[], i: number): boolean {
  const gap = gaps[i];
  const prevOk = i === 0 || gap > gaps[i - 1] * LARGE_GAP_RATIO;
  const nextOk = i === gaps.length - 1 || gap > gaps[i + 1] * LARGE_GAP_RATIO;
  return prevOk && nextOk;
}

function groupRuns(runs: readonly Run[], gaps: readonly number[]): Run[][] {
  const groups: Run[][] = [];
  let current: Run[] = runs.length > 0 ? [runs[0]] : [];
  for (const [i, run] of runs.slice(1).entries()) {
    if (isLargeGap(gaps, i)) {
      groups.push(current);
      current = [];
    }
    current.push(run);
  }
  if (current.length > 0) {
    groups.push(current);
  }
  return groups;
}

/** The first within-group gap, closest to the strip's own left edge, so the walk starts from a
 * width the picture actually measures there rather than a strip-wide average a warp would make
 * unrepresentative of either end. */
function medianSpacing(xs: readonly number[]): number {
  const spacings = xs
    .slice(1)
    .map((x, i) => x - xs[i])
    .sort((a, b) => a - b);
  return spacings[Math.floor(spacings.length / 2)] ?? 0;
}

function firstSmallGap(gaps: readonly number[]): number | null {
  for (const [i, gap] of gaps.entries()) {
    if (!isLargeGap(gaps, i)) {
      return gap;
    }
  }
  return null;
}

/** The longest run of groups that alternate size two and three, since a stray split or merge at
 * either end of the strip is far more likely than one in the middle of a clean sequence. */
function longestAlternating(groups: readonly Run[][]): Run[][] {
  let bestStart = 0;
  let bestLen = 0;
  let start = 0;
  for (let i = 0; i <= groups.length; i += 1) {
    const size = i < groups.length ? groups[i].length : -1;
    const ok =
      (size === 2 || size === 3) &&
      (i === start || size !== groups[i - 1].length);
    if (!ok) {
      if (i - start > bestLen) {
        bestLen = i - start;
        bestStart = start;
      }
      start = i + 1;
    }
  }
  return groups.slice(bestStart, bestStart + bestLen);
}

function nearestWithin(
  list: readonly number[],
  target: number,
  tol: number,
): number | null {
  let best: number | null = null;
  let bestDist = Number.POSITIVE_INFINITY;
  for (const x of list) {
    const dist = Math.abs(x - target);
    if (dist <= tol && dist < bestDist) {
      bestDist = dist;
      best = x;
    }
  }
  return best;
}

type Boundary = { readonly x: number; readonly confirmed: boolean };

/** Walks the strip left to right, predicting the next white-key boundary from the current local
 * width and snapping to the nearest separator dip within a fraction of that width. The local
 * width only updates from evidence, so it drifts with the lens warp instead of assuming a fixed
 * spacing, and the walk simply stops when there is no room left for another key: the number of
 * steps is the count. Black-key edges sit well clear of the white-key boundaries they neighbour
 * (a sharp's own offset into its white key is a real fraction of a key wide), so they are never
 * close enough to a boundary to be safe walk evidence; they earn their keep locking the phase
 * instead, where the question is which key a run sits inside rather than where a boundary is. */
function walkBoundaries(
  seedWidth: number,
  stripWidth: number,
  dips: readonly number[],
): Boundary[] {
  const boundaries: Boundary[] = [{ x: 0, confirmed: true }];
  let x = 0;
  let width = seedWidth;
  let steps = 0;
  const maxSteps = Math.ceil(stripWidth / Math.max(1, seedWidth * 0.4)) + 4;
  while (steps < maxSteps) {
    steps += 1;
    if (width < 1) {
      break;
    }
    const predicted = x + width;
    const remaining = stripWidth - predicted;
    if (remaining < -width * 0.5) {
      break;
    }
    if (remaining < width * 0.5) {
      boundaries.push({ x: stripWidth, confirmed: true });
      break;
    }
    const tol = width * WALK_TOLERANCE_FRACTION;
    const evidence = nearestWithin(dips, predicted, tol);
    if (evidence !== null) {
      const measured = evidence - x;
      width = WIDTH_SMOOTH_ALPHA * measured + (1 - WIDTH_SMOOTH_ALPHA) * width;
      x = evidence;
      boundaries.push({ x, confirmed: true });
    } else {
      x = predicted;
      boundaries.push({ x, confirmed: false });
    }
  }
  return boundaries;
}

type Locked = {
  readonly offset: number;
  readonly agreement: number;
  readonly intervalRuns: ReadonlyMap<number, Run>;
};

/** How close, in white keys, a black key must sit to where a letter puts one to count for it. */
const LOCK_MATCH_KEYS = 0.3;
/** Black keys follow C, D, F, G and A. */
const LETTERS_WITH_BLACK = [0, 1, 3, 4, 5];

/** Where the black key after each white letter sits, in white keys from that white key's left
 * edge, from the key layout itself. */
const BLACK_AFTER = LETTERS.map((_, letter) => {
  if (!LETTERS_WITH_BLACK.includes(letter)) {
    return null;
  }
  const white = firstWhitePitch(letter);
  const { from, to } = keyUnits(white + 1);
  return (from + to) / 2 - whiteIndex(white);
});

/** The letter of the strip's first white key: for each of the seven, we predict where every
 * black key sits between the walked white keys and score how many seen black keys land near a
 * prediction, so one black key filed under the wrong white key can never swing the vote. */
function lockPhase(
  runs: readonly Run[],
  boundaries: readonly Boundary[],
  keyWidth: number,
): Locked | null {
  const at = (u: number): number | null => {
    const i = Math.floor(u);
    const next = boundaries[i + 1];
    const here = boundaries[i];
    return here === undefined || next === undefined
      ? null
      : here.x + (u - i) * (next.x - here.x);
  };
  const reach = keyWidth * LOCK_MATCH_KEYS;
  let best: (Locked & { readonly score: number }) | null = null;
  for (let offset = 0; offset < 7; offset += 1) {
    const predicted: { readonly interval: number; readonly x: number }[] = [];
    for (let i = 0; i < boundaries.length - 1; i += 1) {
      const after = BLACK_AFTER[letterOf(offset, i)];
      const x = after === null ? null : at(i + after);
      if (x !== null) {
        predicted.push({ interval: i, x });
      }
    }
    let score = 0;
    let agreeing = 0;
    const intervalRuns = new Map<number, Run>();
    for (const run of runs) {
      let nearest: { readonly interval: number; readonly x: number } | null =
        null;
      for (const p of predicted) {
        if (
          nearest === null ||
          Math.abs(p.x - run.center) < Math.abs(nearest.x - run.center)
        ) {
          nearest = p;
        }
      }
      const miss = nearest === null ? reach : Math.abs(nearest.x - run.center);
      score += Math.max(0, 1 - miss / reach);
      if (nearest !== null && miss < reach) {
        agreeing += 1;
        intervalRuns.set(nearest.interval, run);
      }
    }
    if (best === null || score > best.score) {
      best = {
        offset,
        agreement: runs.length === 0 ? 0 : agreeing / runs.length,
        intervalRuns,
        score,
      };
    }
  }
  return best === null || best.intervalRuns.size === 0
    ? null
    : {
        offset: best.offset,
        agreement: best.agreement,
        intervalRuns: best.intervalRuns,
      };
}

function letterOf(offset: number, index: number): number {
  return (offset + index) % 7;
}

type Adjusted = {
  readonly x: number;
  readonly confirmed: boolean;
  readonly moved: number;
};

/** Lets every boundary move independently to its own nearest separator dip, within a window
 * small enough it can never cross into a neighbouring key. A boundary the walk had to predict
 * gets the same chance to confirm itself here. */
function neighbourWidth(boundaries: readonly Boundary[], i: number): number {
  const leftWidth = i > 0 ? boundaries[i].x - boundaries[i - 1].x : null;
  const rightWidth =
    i < boundaries.length - 1 ? boundaries[i + 1].x - boundaries[i].x : null;
  if (leftWidth === null) {
    return rightWidth ?? 1;
  }
  if (rightWidth === null) {
    return leftWidth;
  }
  return (leftWidth + rightWidth) / 2;
}

function microAdjust(
  boundaries: readonly Boundary[],
  dips: readonly number[],
): Adjusted[] {
  return boundaries.map((boundary, i) => {
    const localWidth = neighbourWidth(boundaries, i);
    const tol = localWidth * MICRO_TOLERANCE_FRACTION;
    const found = nearestWithin(dips, boundary.x, tol);
    if (found === null) {
      return { x: boundary.x, confirmed: boundary.confirmed, moved: 0 };
    }
    return {
      x: found,
      confirmed: true,
      moved: localWidth === 0 ? 0 : Math.abs(found - boundary.x) / localWidth,
    };
  });
}

function buildKeys(
  boundaries: readonly Adjusted[],
  offset: number,
  intervalRuns: ReadonlyMap<number, Run>,
  height: number,
): DetectedKey[] {
  const keys: DetectedKey[] = [];
  const whiteKeys = boundaries.length - 1;
  for (let i = 0; i < whiteKeys; i += 1) {
    const from = boundaries[i].x;
    const to = boundaries[i + 1].x;
    const semitone =
      12 * Math.floor((offset + i) / 7) +
      WHITE_PITCH_CLASSES[(offset + i) % 7] -
      WHITE_PITCH_CLASSES[offset];
    keys.push({
      black: false,
      semitone,
      bar: [
        { x: from, y: 0 },
        { x: to, y: 0 },
        { x: to, y: height },
        { x: from, y: height },
      ],
    });
    const letter = letterOf(offset, i);
    if (i < whiteKeys - 1 && !NO_BLACK_AFTER.has(letter)) {
      const width = to - from;
      const run = intervalRuns.get(i);
      const x0 = run ? run.start : from + width * FALLBACK_BLACK_OFFSET;
      const x1 = run ? run.end : x0 + width * FALLBACK_BLACK_WIDTH;
      const near = height * blackKeyDepth;
      keys.push({
        black: true,
        semitone: semitone + 1,
        bar: [
          { x: x0, y: 0 },
          { x: x1, y: 0 },
          { x: x1, y: near },
          { x: x0, y: near },
        ],
      });
    }
  }
  return keys;
}

type Correspondence = { readonly u: number; readonly x: number };

/** Where a position along the board, in white-key widths, lands in the strip: a flat keyboard
 * seen in perspective maps its length onto a line by exactly this one-dimensional projective law. */
type Projective = {
  readonly a: number;
  readonly b: number;
  readonly c: number;
};

const projectAlong = (map: Projective, u: number): number =>
  (map.a * u + map.b) / (map.c * u + 1);
const unprojectAlong = (map: Projective, x: number): number =>
  (x - map.b) / (map.a - map.c * x);

function fitProjective(points: readonly Correspondence[]): Projective | null {
  const matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhs = [0, 0, 0];
  for (const { u, x } of points) {
    const row = [u, 1, -u * x];
    for (let i = 0; i < 3; i += 1) {
      rhs[i] += row[i] * x;
      for (let j = 0; j < 3; j += 1) {
        matrix[i][j] += row[i] * row[j];
      }
    }
  }
  const [a, b, c] = solve(matrix, rhs);
  return [a, b, c].every(Number.isFinite) ? { a, b, c } : null;
}

function residual(map: Projective, point: Correspondence): number {
  return Math.abs(projectAlong(map, point.u) - point.x);
}

function robustFit(
  points: readonly Correspondence[],
  tolerance: number,
): Projective | null {
  if (points.length < TEMPLATE_MIN_POINTS) {
    return null;
  }
  const first = fitProjective(points);
  if (first === null) {
    return null;
  }
  const kept = points.filter((point) => residual(first, point) <= tolerance);
  return kept.length < TEMPLATE_MIN_POINTS ? null : fitProjective(kept);
}

type TemplateKey = { readonly from: number; readonly to: number };

function firstWhitePitch(letter: number): number {
  return TEMPLATE_OCTAVE_BASE + WHITE_PITCH_CLASSES[letter];
}

/** Every black key of a keyboard whose first white key is `letter`, in white-key widths from that
 * key's left edge, reaching well past both ends so any run can find its template key. */
function blackTemplate(letter: number): TemplateKey[] {
  const first = firstWhitePitch(letter);
  const base = whiteIndex(first);
  const keys: TemplateKey[] = [];
  for (let pitch = first - 24; pitch < first + 12 * 12; pitch += 1) {
    if (isBlack(pitch)) {
      const units = keyUnits(pitch);
      keys.push({ from: units.from - base, to: units.to - base });
    }
  }
  return keys;
}

function nearestKey(
  keys: readonly TemplateKey[],
  u: number,
): TemplateKey | null {
  let best: TemplateKey | null = null;
  for (const key of keys) {
    if (best === null || Math.abs(key.from - u) < Math.abs(best.from - u)) {
      best = key;
    }
  }
  return best;
}

type FittedKeyboard = {
  readonly map: Projective;
  /** The board's first white key, in the fit's own key positions. */
  readonly firstKey: number;
  readonly whiteKeys: number;
  readonly raiseMm: number;
  /** How many white keys wide the strip is, which the pose behind the raise was solved for. */
  readonly stripKeys: number;
  /** The share of what the strip showed that the fitted keyboard lands on. */
  readonly explainedShare: number;
};

/** Fits the standard keyboard to the strip as one rigid, perspective-foreshortened template.
 * Every separator and every black key is a correspondence between a known template position
 * and a pixel, so the sharp near keys and the rigid pattern carry the far keys, where blur
 * leaves too little evidence for any key to be placed on its own. */
/** What one strip showed, in strip pixels, for the fit to explain. */
type StripEvidence = {
  readonly width: number;
  readonly height: number;
  readonly dips: readonly number[];
  readonly runs: readonly Run[];
  readonly walked: readonly Adjusted[];
  readonly locked: Locked;
};

function fitKeyboard(
  evidence: StripEvidence,
  liftFor: LiftFor | null,
): FittedKeyboard | null {
  const walkedSpan = evidence.walked.length - 1;
  const first = fitOverRaises(
    evidence,
    liftFor?.(walkedSpan) ?? null,
    walkedSpan,
  );
  if (first === null || liftFor === null) {
    return first;
  }
  // the walk miscounts the strip by a key or more where blur hides separators, and the pose's
  // focal and tilt, which set the parallax, follow from that width; the rigid fit knows it better
  const span =
    unprojectAlong(first.map, evidence.width) - unprojectAlong(first.map, 0);
  return fitOverRaises(evidence, liftFor(span), span) ?? first;
}

function fitOverRaises(
  evidence: StripEvidence,
  lift: Lift | null,
  stripKeys: number,
): FittedKeyboard | null {
  let best: (FittedKeyboard & { readonly cost: number }) | null = null;
  for (const raiseMm of lift === null ? [0] : RAISE_CANDIDATES_MM) {
    // a black key seen in the flat strip sits where its raised top appears, so we move each
    // black observation back by its parallax before it meets white-key evidence on the plane
    const flatten = (x: number): number =>
      lift === null
        ? x
        : 2 * x - lift(x, evidence.height * BLACK_BAND_MIDDLE, raiseMm);
    const fitted = fitAtRaise(evidence, flatten);
    if (fitted !== null && (best === null || fitted.cost < best.cost)) {
      best = { ...fitted, raiseMm, stripKeys };
    }
  }
  return best;
}

function fitAtRaise(
  { width, dips, runs, walked, locked }: StripEvidence,
  flatten: (x: number) => number,
): {
  readonly map: Projective;
  readonly firstKey: number;
  readonly whiteKeys: number;
  readonly cost: number;
  readonly explainedShare: number;
} | null {
  const blacks = blackTemplate(locked.offset);
  const tolerance = (width / (walked.length - 1)) * TEMPLATE_OUTLIER_KEYS;
  const seed: Correspondence[] = [];
  walked.forEach((boundary, i) => {
    if (boundary.confirmed && i > 0 && i < walked.length - 1) {
      seed.push({ u: i, x: boundary.x });
    }
  });
  for (const [interval, run] of locked.intervalRuns) {
    // past the last confirmed boundary the walk's intervals are guesses, and a black key filed
    // under the wrong interval would pull the whole fit toward a consistent but wrong board
    if (!walked[interval]?.confirmed || !walked[interval + 1]?.confirmed) {
      continue;
    }
    const black = blacks.find(
      (key) => key.from > interval && key.from < interval + 1,
    );
    if (black) {
      seed.push(
        { u: black.from, x: flatten(run.start) },
        { u: black.to, x: flatten(run.end) },
      );
    }
  }
  const associate = (fitted: Projective): Correspondence[] => {
    const evidence: Correspondence[] = [];
    for (const dip of dips) {
      const u = unprojectAlong(fitted, dip);
      if (Math.abs(u - Math.round(u)) < TEMPLATE_SNAP_KEYS) {
        evidence.push({ u: Math.round(u), x: dip });
      }
    }
    for (const run of runs) {
      const start = flatten(run.start);
      const u = unprojectAlong(fitted, start);
      const black = nearestKey(blacks, u);
      if (black && Math.abs(black.from - u) < TEMPLATE_SNAP_KEYS) {
        evidence.push(
          { u: black.from, x: start },
          { u: black.to, x: flatten(run.end) },
        );
      }
    }
    return evidence;
  };
  let map = robustFit(seed, tolerance);
  for (let round = 0; round < TEMPLATE_ROUNDS && map !== null; round += 1) {
    map = robustFit(associate(map), tolerance) ?? map;
  }
  if (map === null) {
    return null;
  }
  const fittedMap = map;
  const explained = associate(fittedMap);
  // the board spans the keys we actually saw, never simply the strip, since the strip's ends are
  // the mask's guess and routinely reach past the last key or stop short of it
  // where the mask did stop on the board's end, the strip's own edge is the end of the last key,
  // with no dark line of its own left in view to show it
  const stripEdges = [0, width]
    .map((x) => unprojectAlong(fittedMap, x))
    .filter((u) => Math.abs(u - Math.round(u)) < STRIP_EDGE_SNAP_KEYS)
    .map((u) => ({
      u: Math.round(u),
      x: projectAlong(fittedMap, Math.round(u)),
    }));
  const extent = keyExtent([
    ...explained.filter((point) => residual(fittedMap, point) <= tolerance),
    ...stripEdges,
  ]);
  if (extent === null) {
    return null;
  }
  const whiteKeys = extent.last - extent.first + 1;
  const monotonic =
    fittedMap.c * extent.first + 1 > 0 &&
    fittedMap.c * (extent.last + 1) + 1 > 0 &&
    projectAlong(fittedMap, extent.last + 1) >
      projectAlong(fittedMap, extent.first);
  if (!monotonic || whiteKeys < MIN_WHITE_KEYS || whiteKeys > MOST_WHITE_KEYS) {
    return null;
  }
  // evidence the fit could not explain counts as a full miss, so a wrong raise cannot score well
  // by leaving the black keys out of its own evidence
  const possible = dips.length + 2 * runs.length;
  const cost =
    (explained.reduce(
      (sum, point) =>
        sum + Math.min(residual(fittedMap, point) ** 2, tolerance ** 2),
      0,
    ) +
      Math.max(0, possible - explained.length) * tolerance ** 2) /
    Math.max(1, possible);
  const inliers = explained.filter(
    (point) => residual(fittedMap, point) <= tolerance,
  ).length;
  return {
    map,
    firstKey: extent.first,
    whiteKeys,
    cost,
    explainedShare: inliers / Math.max(1, possible),
  };
}

/** The first and last white key the evidence shows. A key counts as seen when separators mark
 * both of its edges or a black key's edge lies over it; one separator alone is not enough, since
 * the dark line where the last key meets the case is a separator with no key beyond it. */
function keyExtent(
  points: readonly Correspondence[],
): { readonly first: number; readonly last: number } | null {
  const separators = new Set<number>();
  const underBlack = new Set<number>();
  for (const { u } of points) {
    if (Number.isInteger(u)) {
      separators.add(u);
    } else {
      underBlack.add(Math.floor(u));
    }
  }
  const candidates = [...separators, ...underBlack];
  if (candidates.length === 0) {
    return null;
  }
  const seen = (key: number): boolean =>
    underBlack.has(key) || (separators.has(key) && separators.has(key + 1));
  let first: number | null = null;
  let last: number | null = null;
  for (
    let key = Math.min(...candidates) - 1;
    key <= Math.max(...candidates);
    key += 1
  ) {
    if (seen(key)) {
      first ??= key;
      last = key;
    }
  }
  return first === null || last === null ? null : { first, last };
}

/** Grows the seen keys outward while the next key's front is as bright as the seen white keys.
 * Brightness survives blur, where the separators that prove a key vanish first, and the case
 * past a board's end is dark, so this tells a far end too blurred to prove from a mask that ran
 * past the board. */
function extendOverWhiteKeys(
  strip: Strip,
  map: Projective,
  seen: KeySpan,
): BrightSpan {
  const top = Math.round(strip.height * SEPARATOR_BAND[0]);
  const bottom = Math.round(strip.height * SEPARATOR_BAND[1]);
  const front = (key: number): number | null => {
    const from = projectAlong(map, key);
    const to = projectAlong(map, key + 1);
    const inset = (to - from) * WHITE_FRONT_INSET;
    const left = Math.max(0, Math.round(from + inset));
    const right = Math.min(strip.width, Math.round(to - inset));
    if (right - left < (to - from) * WHITE_FRONT_MIN_INSIDE) {
      return null;
    }
    let sum = 0;
    let count = 0;
    for (let y = top; y < bottom; y += 1) {
      const value = bandBrightness(strip, y, left, right);
      if (value !== null) {
        sum += value;
        count += 1;
      }
    }
    return count === 0 ? null : sum / count;
  };
  // light falls off along a board, so a key is compared with its nearest neighbours rather than
  // with the whole board, whose far end can read half as bright as its near end
  const reference = (keys: readonly number[]): number | null => {
    const values = keys
      .map(front)
      .filter((value): value is number => value !== null)
      .sort((a, b) => a - b);
    return values.length === 0 ? null : values[Math.floor(values.length / 2)];
  };
  const isKeyBeside = (key: number, neighbours: readonly number[]): boolean => {
    const near = reference(neighbours);
    const value = front(key);
    return near !== null && value !== null && value >= near * WHITE_FRONT_SHARE;
  };
  let { first, last } = seen;
  while (
    first > seen.first - MAX_EXTEND_KEYS &&
    isKeyBeside(first - 1, [first, first + 1, first + 2])
  ) {
    first -= 1;
  }
  while (
    last < seen.last + MAX_EXTEND_KEYS &&
    isKeyBeside(last + 1, [last, last - 1, last - 2])
  ) {
    last += 1;
  }
  return {
    first,
    last,
    openBefore: front(first - 1) === null,
    openAfter: front(last + 1) === null,
  };
}

function letterAt(offset: number, key: number): number {
  return (((offset + key) % 7) + 7) % 7;
}

type KeySpan = { readonly first: number; readonly last: number };
/** Keys with bright fronts, and whether each end ran into the strip's edge rather than into
 * dark case, in which case the board may carry on out of sight. */
type BrightSpan = KeySpan & {
  readonly openBefore: boolean;
  readonly openAfter: boolean;
};

/** The board a keyboard really is. Keys proven by separators or black keys must all be on it,
 * and it cannot reach past keys whose fronts are bright like white keys; among the sizes
 * keyboards are built in, we take the one that fits between those bounds and whose first key
 * has the right letter. A board of a size nobody builds is far less likely than a key lost to
 * blur or a patch of bright floor read as one more key. */
function standardBoard(
  offset: number,
  proven: KeySpan,
  bright: BrightSpan,
): { readonly firstKey: number; readonly whiteKeys: number } {
  const before = bright.openBefore ? UNSEEN_KEYS : 0;
  const after = bright.openAfter ? UNSEEN_KEYS : 0;
  let best: {
    readonly firstKey: number;
    readonly whiteKeys: number;
    readonly slack: number;
  } | null = null;
  for (const board of STANDARD_BOARDS) {
    for (let start = bright.first - before; start <= proven.first; start += 1) {
      const end = start + board.whiteKeys - 1;
      const fits =
        LETTERS[letterAt(offset, start)] === board.phase &&
        end >= proven.last &&
        end <= bright.last + after;
      const slack =
        Math.abs(start - bright.first) + Math.abs(end - bright.last);
      if (fits && (best === null || slack < best.slack)) {
        best = { firstKey: start, whiteKeys: board.whiteKeys, slack };
      }
    }
  }
  return (
    best ?? {
      firstKey: bright.first,
      whiteKeys: bright.last - bright.first + 1,
    }
  );
}

/** Where the black keys end, as a share of the strip's depth. A black key's front face meets the
 * white keys at a point on the keybed plane, so the flat strip shows it at its true length
 * whatever the keys' height and the camera's angle. */
function measureBlackDepth(
  strip: Strip,
  spans: readonly (readonly [number, number])[],
): number | null {
  const ends: number[] = [];
  for (const [x0, x1] of spans) {
    const inset = (x1 - x0) * BLACK_END_INSET;
    const brightness = bandColumn(
      strip,
      Math.round(x0 + inset),
      Math.round(x1 - inset),
      strip.height,
    );
    if (brightness === null) {
      continue;
    }
    const end = steepestRise(
      brightness,
      Math.round(strip.height * BLACK_END_SEARCH[0]),
      Math.round(strip.height * BLACK_END_SEARCH[1]),
    );
    if (end !== null) {
      ends.push(end / strip.height);
    }
  }
  if (ends.length < BLACK_END_MIN_KEYS) {
    return null;
  }
  ends.sort((a, b) => a - b);
  return ends[Math.floor(ends.length / 2)];
}

function keysFromTemplate(
  map: Projective,
  letter: number,
  whiteKeys: number,
  height: number,
  blackDepth: number,
  firstKey: number,
): DetectedKey[] {
  const first = firstWhitePitch(letter);
  const base = whiteIndex(first);
  const bar = (from: number, to: number, depth: number): Bar => [
    { x: from, y: 0 },
    { x: to, y: 0 },
    { x: to, y: depth },
    { x: from, y: depth },
  ];
  const keys: DetectedKey[] = [];
  let pitch = first;
  for (let i = 0; i < whiteKeys; i += 1) {
    while (isBlack(pitch)) {
      pitch += 1;
    }
    keys.push({
      black: false,
      semitone: pitch - first,
      bar: bar(
        projectAlong(map, firstKey + i),
        projectAlong(map, firstKey + i + 1),
        height,
      ),
    });
    if (i < whiteKeys - 1 && isBlack(pitch + 1)) {
      const units = keyUnits(pitch + 1);
      keys.push({
        black: true,
        semitone: pitch + 1 - first,
        bar: bar(
          projectAlong(map, firstKey + units.from - base),
          projectAlong(map, firstKey + units.to - base),
          height * blackDepth,
        ),
      });
    }
    pitch += 1;
  }
  return keys;
}

function fittedKeys(
  strip: Strip,
  map: Projective,
  letter: number,
  whiteKeys: number,
  firstKey: number,
): DetectedKey[] {
  const build = (blackDepth: number): DetectedKey[] =>
    keysFromTemplate(
      map,
      letter,
      whiteKeys,
      strip.height,
      blackDepth,
      firstKey,
    );
  const assumed = build(blackKeyDepth);
  const spans = assumed
    .filter((key) => key.black)
    .map((key) => [key.bar[0].x, key.bar[1].x] as const);
  const measured = measureBlackDepth(strip, spans);
  return measured === null ? assumed : build(measured);
}

const NO_SHIFT: Line = { meanX: 0, meanY: 0, slope: 0 };
/** A seen black key further than this from every drawn one, in white keys, belongs to no key. */
const SHIFT_MATCH_KEYS = 0.5;
/** A black key whose offset strays this far from the fitted line, in white keys, is a misread. */
const SHIFT_OUTLIER_KEYS = 0.15;
const SHIFT_MIN_KEYS = 4;

/** A black key's centre sits within this of the white-key boundary it straddles, in white keys. */
const LEAN_OUTLIER_KEYS = 0.2;
const LEAN_MIN_GROUPS = 2;

/** How far, in white keys, a group of two or three black keys sits off its boundaries on average
 * on a flat keyboard, from the key layout itself. */
function groupOffset(pitches: readonly number[]): number {
  const offsets = pitches.map((pitch) => {
    const { from, to } = keyUnits(pitch);
    const center = (from + to) / 2;
    return center - Math.round(center);
  });
  return offsets.reduce((sum, offset) => sum + offset, 0) / offsets.length;
}

const PAIR_OFFSET = groupOffset([61, 63]);
const TRIPLE_OFFSET = groupOffset([66, 68, 70]);

/** The line along the strip by which the black keys sit off the white-key boundaries they
 * straddle, past the offset the key layout itself gives each group of two or three, which is
 * the lean the view gives every raised top there. */
function measureLean(
  groups: readonly (readonly Run[])[],
  boundaries: readonly Boundary[],
  keyWidth: number,
): (x: number) => number {
  // a boundary the walk only predicted drifts with the walk, so we never level black keys to one
  const offsetOf = (run: Run): number | null => {
    let nearest: Boundary | null = null;
    for (const boundary of boundaries) {
      if (
        nearest === null ||
        Math.abs(boundary.x - run.center) < Math.abs(nearest.x - run.center)
      ) {
        nearest = boundary;
      }
    }
    return nearest?.confirmed ? run.center - nearest.x : null;
  };
  const points: LinePoint[] = [];
  for (const group of groups) {
    const offsets = group.map(offsetOf);
    if (offsets.every((offset): offset is number => offset !== null)) {
      points.push({
        x: group.reduce((sum, run) => sum + run.center, 0) / group.length,
        y:
          offsets.reduce((sum, offset) => sum + offset, 0) / group.length -
          (group.length === 2 ? PAIR_OFFSET : TRIPLE_OFFSET) * keyWidth,
      });
    }
  }
  if (points.length < LEAN_MIN_GROUPS) {
    return () => 0;
  }
  const first = fitLine(points);
  const kept = points.filter(
    (point) => lineResidual(first, point) <= keyWidth * LEAN_OUTLIER_KEYS,
  );
  const used = kept.length < LEAN_MIN_GROUPS ? points : kept;
  const line = fitLine(used);
  // past the groups we measured the line only extrapolates, so we never claim more lean than we saw
  const least = Math.min(...used.map((point) => point.y));
  const most = Math.max(...used.map((point) => point.y));
  return (x) => Math.min(most, Math.max(least, lineAt(line, x)));
}

/** The height to draw the black keys at: the one whose tops, seen from the pose, land nearest
 * the black keys the strip shows. The fit places the keys from leveled evidence and never needs
 * the pose, so the pose only has to make the drawing look raised. */
function drawnRaise(
  keys: readonly DetectedKey[],
  runs: readonly Run[],
  candidates: readonly number[],
  appearsAt: (raiseMm: number) => (x: number) => number,
  keyWidth: number,
): number {
  const centers = keys
    .filter((key) => key.black)
    .map((key) => key.bar.reduce((sum, p) => sum + p.x, 0) / 4);
  let best = candidates[0] ?? 0;
  let bestCost = Number.POSITIVE_INFINITY;
  for (const raiseMm of candidates) {
    const appears = appearsAt(raiseMm);
    const drawn = centers.map(appears);
    const cost = runs.reduce((sum, run) => {
      const miss = Math.min(
        ...drawn.map((x) => Math.abs(x - run.center)),
        keyWidth * SHIFT_MATCH_KEYS,
      );
      return sum + miss;
    }, 0);
    if (cost < bestCost) {
      bestCost = cost;
      best = raiseMm;
    }
  }
  return best;
}

/** The line along the strip that carries each drawn black key onto the black key the evidence
 * shows nearest it, both where they appear at the black band. */
function measureBlackShift(
  keys: readonly DetectedKey[],
  runs: readonly Run[],
  appears: (x: number) => number,
  keyWidth: number,
): Line {
  const drawn = keys
    .filter((key) => key.black)
    .map((key) => appears(key.bar.reduce((sum, p) => sum + p.x, 0) / 4));
  const offsets: LinePoint[] = [];
  for (const run of runs) {
    let nearest: number | null = null;
    for (const x of drawn) {
      if (
        nearest === null ||
        Math.abs(x - run.center) < Math.abs(nearest - run.center)
      ) {
        nearest = x;
      }
    }
    if (
      nearest !== null &&
      Math.abs(nearest - run.center) < keyWidth * SHIFT_MATCH_KEYS
    ) {
      offsets.push({ x: nearest, y: run.center - nearest });
    }
  }
  if (offsets.length < SHIFT_MIN_KEYS) {
    return NO_SHIFT;
  }
  const first = fitLine(offsets);
  const kept = offsets.filter(
    (point) => lineResidual(first, point) <= keyWidth * SHIFT_OUTLIER_KEYS,
  );
  return kept.length < SHIFT_MIN_KEYS ? first : fitLine(kept);
}

/** Where a strip shows its keys, in strip pixels: the gaps between white keys and the black keys'
 * extents. The brightness rules produce it, and a learned key matcher can stand in for them. */
export type KeyEvidence = {
  readonly dips: readonly number[];
  readonly runs: readonly Run[];
};

type Unsure = Extract<KeyRead, { kind: "unsure" }>;

function brightnessEvidence(strip: Strip): KeyEvidence | Unsure {
  const blackProfile = bandProfile(strip, BLACK_BAND[0], BLACK_BAND[1]);
  const blackSplit = split(blackProfile);
  if (blackSplit.separation < MIN_BLACK_SEPARATION) {
    return {
      kind: "unsure",
      reason: "no readable black-key pattern",
      confidence: 0,
    };
  }
  return {
    runs: darkRuns(blackProfile, blackSplit.at, MIN_RUN_WIDTH_PX),
    dips: findDips(bandProfile(strip, SEPARATOR_BAND[0], SEPARATOR_BAND[1])),
  };
}

/** Finds the keys in a rectified strip: the white-key separator lines near the player's edge and
 * the black-key runs near the far edge, walked and refined rather than fit to one assumed grid,
 * so a leftover lens warp does not have to be modelled to be tolerated. */
export function detectKeys(
  strip: Strip,
  lift: LiftFor | null = null,
  evidence: KeyEvidence | null = null,
): KeyRead {
  const seen = evidence ?? brightnessEvidence(strip);
  if ("kind" in seen) {
    return seen;
  }
  const { runs, dips } = seen;
  if (runs.length < MIN_RUNS) {
    return {
      kind: "unsure",
      reason: `only ${runs.length} black keys visible`,
      confidence: runs.length / (2 * MIN_RUNS),
    };
  }
  const gaps = runs.slice(1).map((run, i) => run.center - runs[i].center);
  const groups = longestAlternating(groupRuns(runs, gaps));
  if (groups.length === 0) {
    return {
      kind: "unsure",
      reason: "black keys did not form a clean two-and-three pattern",
      confidence: 0.1,
    };
  }

  const seed = firstSmallGap(gaps);
  // the matcher finds most white-key gaps, so their median spacing is the key width; one split
  // black key at the start would otherwise shrink every step of the walk
  const seedWidth =
    evidence === null
      ? (seed ?? strip.width / 30) / WITHIN_GROUP_UNIT_GAP
      : medianSpacing(dips);
  const walked = walkBoundaries(seedWidth, strip.width, dips);
  const whiteKeys = walked.length - 1;
  if (whiteKeys < MIN_WHITE_KEYS || whiteKeys > MOST_WHITE_KEYS) {
    return {
      kind: "unsure",
      reason: `walked an unlikely ${whiteKeys} white keys`,
      confidence: 0.1,
    };
  }

  // raised black tops lean off the white-key boundaries they straddle, far enough on steep views
  // to file a black key under the wrong white key, so we measure the lean and take it out first
  const lean = measureLean(groups, walked, seedWidth);
  const leveled = runs.map((run) => {
    const by = lean(run.center);
    return {
      start: run.start - by,
      end: run.end - by,
      center: run.center - by,
    };
  });
  const locked = lockPhase(leveled, walked, seedWidth);
  if (locked === null) {
    return {
      kind: "unsure",
      reason: "no black key landed inside a walked white key",
      confidence: 0.1,
    };
  }

  const adjusted = microAdjust(walked, dips);
  const confirmed = adjusted.filter((b) => b.confirmed).length;
  const evidenceRate = confirmed / adjusted.length;
  const avgMove =
    adjusted.reduce((sum, b) => sum + Math.min(1, b.moved), 0) /
    adjusted.length;
  const positionQuality = Math.max(0, 1 - avgMove);
  const groupEvidence = Math.min(1, groups.length / 4);
  // Evidence, fit quality and phase agreement each catch a different failure (too few keys
  // seen, a warp the walk fought rather than followed, black keys that do not read as one
  // consistent keyboard), so we require all three to be good rather than average them away.
  const confidence =
    evidenceRate *
    positionQuality *
    (0.4 + 0.6 * locked.agreement) *
    groupEvidence;

  const fitted = fitKeyboard(
    {
      width: strip.width,
      height: strip.height,
      dips,
      runs: leveled,
      walked: adjusted,
      locked,
    },
    lift,
  );
  if (fitted === null) {
    const keys = buildKeys(
      adjusted,
      locked.offset,
      locked.intervalRuns,
      strip.height,
    );
    return {
      kind: "read",
      whiteKeys,
      stripKeys: whiteKeys,
      blackRaiseMm: 0,
      blackShift: NO_SHIFT,
      outline: PLAIN_OUTLINE,
      totalKeys: keys.length,
      phase: LETTERS[letterOf(locked.offset, 0)],
      confidence,
      keyAt: (index) => keys[index] ?? null,
    };
  }
  const proven = {
    first: fitted.firstKey,
    last: fitted.firstKey + fitted.whiteKeys - 1,
  };
  const board = standardBoard(
    locked.offset,
    proven,
    extendOverWhiteKeys(strip, fitted.map, proven),
  );
  const letter = letterAt(locked.offset, board.firstKey);
  const keys = fittedKeys(
    strip,
    fitted.map,
    letter,
    board.whiteKeys,
    board.firstKey,
  );
  const liftAt = lift?.(fitted.stripKeys) ?? null;
  const keyWidth = strip.width / fitted.stripKeys;
  const appearsAt =
    (raiseMm: number) =>
    (x: number): number =>
      liftAt === null
        ? x
        : liftAt(x, strip.height * BLACK_BAND_MIDDLE, raiseMm);
  const raiseMm = drawnRaise(
    keys,
    runs,
    liftAt === null ? [0] : RAISE_CANDIDATES_MM,
    appearsAt,
    keyWidth,
  );
  return {
    kind: "read",
    whiteKeys: board.whiteKeys,
    stripKeys: fitted.stripKeys,
    blackRaiseMm: raiseMm,
    blackShift: measureBlackShift(keys, runs, appearsAt(raiseMm), keyWidth),
    outline: PLAIN_OUTLINE,
    totalKeys: keys.length,
    phase: LETTERS[letter],
    confidence: confidence * fitted.explainedShare,
    keyAt: (index) => keys[index] ?? null,
  };
}

let captureCanvas: HTMLCanvasElement | null = null;
const CAPTURE_WIDTH = 1600;

export function captureSource(
  frame: CanvasImageSource,
  size: Size,
): SourceImage | null {
  captureCanvas ??= document.createElement("canvas");
  const scale = Math.min(1, CAPTURE_WIDTH / size.width);
  const width = Math.max(1, Math.round(size.width * scale));
  const height = Math.max(1, Math.round(size.height * scale));
  captureCanvas.width = width;
  captureCanvas.height = height;
  const ctx = captureCanvas.getContext("2d", { willReadFrequently: true });
  if (ctx === null) {
    return null;
  }
  ctx.drawImage(frame, 0, 0, width, height);
  const pixels = ctx.getImageData(0, 0, width, height);
  return { width, height, data: pixels.data };
}

/** How the keys are read: every try a second and a half apart, keeping the one that fits its
 * picture best, and then never again until the keybed moves. Reads that agree on the board
 * still place its keys up to half a key apart, and a confident read can be one of those. */
const everyMs = 1500;
/** How soon we look again while the far edge is still unmeasured, since every look without it
 * leaves the drawn keybed reaching into the case. */
const untrimmedEveryMs = 250;
const tries = 16;
/** A board read this unsure has been wrong about the key count on our recordings, and keys
 * drawn off by one would light the wrong note, so nothing draws a read below it. */
export const TRUSTED_READ = 0.5;
const settleAfterMs = 250;
/** Like the tracker's hold, we draw a far edge only once this many looks in a row agree, since
 * the first look after a hold often catches the camera still settling. */
const TRIM_READS_TO_TRUST = 3;
const TRIM_AGREE_WITHIN = 0.03;

/** The median of the latest far-edge measurements, once enough of them agree at both ends of the
 * board; null while they are too few or still disagree. */
export function trustedEdge(recent: readonly FarEdge[]): FarEdge | null {
  if (recent.length < TRIM_READS_TO_TRUST) {
    return null;
  }
  const lefts = recent.map((edge) => edge.left).sort((a, b) => a - b);
  const rights = recent.map((edge) => edge.right).sort((a, b) => a - b);
  const spread = (values: number[]): number =>
    (values.at(-1) ?? 0) - (values[0] ?? 0);
  if (spread(lefts) > TRIM_AGREE_WITHIN || spread(rights) > TRIM_AGREE_WITHIN) {
    return null;
  }
  const middle = Math.floor(recent.length / 2);
  return { left: lefts[middle], right: rights[middle] };
}

export type KeyReader = {
  readonly last: () => KeyRead | null;
  /** The far edge this reader currently trusts, or null before any try has found evidence. */
  readonly fraction: () => FarEdge | null;
  readonly look: (
    frame: CanvasImageSource,
    quad: readonly Point[],
    size: Size,
    now: number,
  ) => void;
  readonly moved: (now: number) => void;
};

type BoardRead = Extract<KeyRead, { kind: "read" }>;

/** The grey level that best splits the values into a dark and a bright class. */
export function otsu(values: readonly number[]): number {
  const histogram = new Array<number>(256).fill(0);
  for (const v of values) {
    histogram[Math.min(255, Math.max(0, Math.round(v)))] += 1;
  }
  const total = values.length;
  const sum = histogram.reduce((s, count, level) => s + count * level, 0);
  let below = 0;
  let belowSum = 0;
  let best = 0;
  let threshold = 128;
  for (let level = 0; level < 256; level += 1) {
    below += histogram[level];
    if (below === 0 || below === total) {
      continue;
    }
    belowSum += histogram[level] * level;
    const above = total - below;
    const between =
      below * above * (belowSum / below - (sum - belowSum) / above) ** 2;
    if (between > best) {
      best = between;
      threshold = level;
    }
  }
  return threshold;
}

/** Points spread over a four-corner face, from its first edge (u) towards its third (v). */
function facePoints(face: readonly Point[], from: number, to: number): Point[] {
  const [a, b, c, d] = face;
  const points: Point[] = [];
  for (let i = 0; i < FIT_SAMPLES; i += 1) {
    const u = (i + 0.5) / FIT_SAMPLES;
    for (let j = 0; j < FIT_SAMPLES; j += 1) {
      const v = from + ((j + 0.5) / FIT_SAMPLES) * (to - from);
      const top = { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u };
      const bottom = { x: d.x + (c.x - d.x) * u, y: d.y + (c.y - d.y) * u };
      points.push({
        x: top.x + (bottom.x - top.x) * v,
        y: top.y + (bottom.y - top.y) * v,
      });
    }
  }
  return points;
}

const FIT_SAMPLES = 4;
/** The front of a white key, as a share of its depth, where no black key covers it. */
const WHITE_FRONT: readonly [number, number] = [0.75, 0.95];

/** How well a read's keys sit on the picture it was read from: the share of points on its black
 * keys that are dark and on the front of its white keys that are bright, 1 when every point
 * agrees. Two reads of one board can agree on its keys and still place them half a key apart. */
export function readFit(
  read: BoardRead,
  quad: readonly Point[],
  source: SourceImage,
): number {
  const faces = projectKeyFaces(read, quad, source);
  const grey = (p: Point): number | null => {
    const rgb = sampleBilinear(source, p.x * source.width, p.y * source.height);
    return rgb === null
      ? null
      : 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
  };
  const values = (points: readonly Point[]): number[] =>
    points.map(grey).filter((v): v is number => v !== null);
  const dark = values(
    faces
      .filter((face) => face.black)
      .flatMap((face) => facePoints(face.bar, 0, 1)),
  );
  const bright = values(
    faces
      .filter((face) => !face.black)
      .flatMap((face) => facePoints(face.bar, WHITE_FRONT[0], WHITE_FRONT[1])),
  );
  if (dark.length === 0 || bright.length === 0) {
    return 0;
  }
  const split = otsu([...dark, ...bright]);
  const darkShare = dark.filter((v) => v < split).length / dark.length;
  const brightShare = bright.filter((v) => v >= split).length / bright.length;
  return (darkShare + brightShare) / 2;
}

/** The read to trust across several looks at the same held keyboard: the board most looks
 * agree on, then the look among them that scores best, the most confident unless told how else
 * to score. A still camera shows the same keyboard every time, so a look that disagrees with the
 * rest caught a hand or a blur, and one lucky confident look should not outvote the others. */
export function boardConsensus(
  reads: readonly BoardRead[],
  score: (read: BoardRead) => number = (read) => read.confidence,
): BoardRead | null {
  const groups = new Map<string, BoardRead[]>();
  for (const read of reads) {
    const board = `${read.whiteKeys}${read.phase}`;
    groups.set(board, [...(groups.get(board) ?? []), read]);
  }
  let winner: BoardRead[] | null = null;
  const weight = (group: readonly BoardRead[]): number =>
    group.reduce((sum, read) => sum + read.confidence, 0);
  for (const group of groups.values()) {
    if (
      winner === null ||
      group.length > winner.length ||
      (group.length === winner.length && weight(group) > weight(winner))
    ) {
      winner = group;
    }
  }
  if (winner === null) {
    return null;
  }
  return winner.reduce((a, b) => (score(b) > score(a) ? b : a));
}

/** Reads the keys with the learned matcher when one is given, and with the brightness rules
 * whenever the matcher is missing or sees too little. A key segmenter, when given, first turns
 * the held quad into the outline the key pixels show, so the template is laid on the keys the
 * camera sees rather than on the mask's guess. */
export function createKeyReader(
  matcher: KeyMatcher | null = null,
  segmenter: KeySegmenter | null = null,
): KeyReader {
  let at = 0;
  let generation = 0;
  let matching = false;
  let taken = 0;
  let best: KeyRead | null = null;
  let reads: BoardRead[] = [];
  let fits = new Map<BoardRead, number>();
  let dueAt: number | null = null;
  let trimEdge: FarEdge | null = null;
  let recentEdges: FarEdge[] = [];

  return {
    last: () => best,
    fraction: () => trimEdge,
    look: (frame, quad, size, now) => {
      if (dueAt !== null && now < dueAt) {
        return;
      }
      dueAt = null;
      const settled = taken >= tries;
      const interval = trimEdge === null ? untrimmedEveryMs : everyMs;
      if (settled || now - at < interval) {
        return;
      }
      at = now;
      taken += 1;
      const source = captureSource(frame, size);
      if (source === null) {
        return;
      }
      const rawStrip = rectifyStrip(source, quad);
      if (rawStrip === null) {
        return;
      }
      const measured = measureFarEdge(rawStrip);
      if (measured !== null) {
        recentEdges = [...recentEdges, measured].slice(-TRIM_READS_TO_TRUST);
        trimEdge = trustedEdge(recentEdges) ?? trimEdge;
      }
      const keyQuad = trimEdge === null ? quad : trimFarEdge(quad, trimEdge);
      const strip = rectifyStrip(source, keyQuad);
      if (strip === null) {
        return;
      }
      const record = (
        readStrip: Strip,
        readQuad: readonly Point[],
        outline: readonly Point[],
        evidence: KeyEvidence | null,
      ): void => {
        const detected = detectKeys(
          readStrip,
          liftFor(readQuad, source),
          evidence,
        );
        const found =
          detected.kind === "read" ? { ...detected, outline } : detected;
        window.pianocvKeyStrip = readStrip;
        window.pianocvKeyRead = found;
        window.pianocvKeyEvidence = evidence;
        if (found.kind === "read") {
          reads = [...reads, found];
          fits.set(found, readFit(found, keyQuad, source));
        }
        // any trusted read beats every untrusted one, and among trusted reads the best fit wins
        best =
          boardConsensus(reads, (read) =>
            read.confidence >= TRUSTED_READ
              ? 1 + (fits.get(read) ?? 0)
              : read.confidence,
          ) ?? found;
        window.pianocvReadFits = reads.map((read) => ({
          confidence: read.confidence,
          fit: fits.get(read) ?? 0,
          chosen: read === best,
        }));
      };
      if (matcher === null) {
        record(strip, keyQuad, PLAIN_OUTLINE, null);
        return;
      }
      if (matching) {
        return;
      }
      matching = true;
      const started = generation;
      const toKeyStrip = findHomography(
        keyQuad.slice(0, 4).map((p) => ({ x: p.x, y: p.y })),
        [...PLAIN_OUTLINE],
      );
      void (
        segmenter
          ?.segment(frame, size, keyQuad)
          .then((found) => found.outline) ?? Promise.resolve(null)
      )
        .catch(() => null)
        .then((outlined) => matcher.match(source, outlined ?? keyQuad))
        .catch(() => null)
        .then((matched) => {
          matching = false;
          if (started !== generation) {
            return;
          }
          const squared =
            matched === null ? null : rectifyStrip(source, matched.quad);
          if (matched === null || squared === null) {
            record(strip, keyQuad, PLAIN_OUTLINE, null);
            return;
          }
          record(
            squared,
            matched.quad,
            matched.quad.map((p) => applyHomography(toKeyStrip, p.x, p.y)),
            matched.evidence,
          );
        });
    },
    moved: (now) => {
      generation += 1;
      taken = 0;
      best = null;
      reads = [];
      fits = new Map();
      dueAt = now + settleAfterMs;
      trimEdge = null;
      recentEdges = [];
    },
  };
}
