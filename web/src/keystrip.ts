import { split } from "./board";
import { applyHomography, findHomography, type Point } from "./homography";
import type { Size } from "./keyspace";
import { type Bar, blackKeyDepth } from "./keyspace";

declare global {
  interface Window {
    // the most recent rectify and what it found, for the lab to read a live page off rather
    // than guess at the pipeline's internal state
    kvtKeyStrip?: { width: number; height: number; data: Uint8ClampedArray };
    kvtKeyRead?: KeyRead;
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
/** Which two runs a group of two is, and which three a group of three is, left to right. */
const GROUP2_LETTERS = [0, 1];
const GROUP3_LETTERS = [3, 4, 5];
/** E and B have no black key between them and the next white key. */
const NO_BLACK_AFTER = new Set([2, 6]);

const FALLBACK_BLACK_OFFSET = 0.62;
const FALLBACK_BLACK_WIDTH = 0.56;

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

export type DetectedKey = {
  readonly bar: Bar;
  readonly black: boolean;
};

export type KeyRead =
  | {
      readonly kind: "read";
      readonly whiteKeys: number;
      readonly totalKeys: number;
      /** The white key the strip's left edge starts on, C to B. */
      readonly phase: string;
      readonly confidence: number;
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

function sampleBilinear(
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

/** Maps a detection's keys, given in strip pixels, into the current frame through the same
 * homography a fresh rectify would use, so drawing tracks the followed quad every frame without
 * rebuilding the strip. */
export function projectKeys(
  read: KeyRead,
  quad: readonly Point[],
): DetectedKey[] {
  if (read.kind !== "read" || quad.length < 4) {
    return [];
  }
  const toFrame = findHomography(
    stripCorners(STRIP_WIDTH, STRIP_HEIGHT),
    quad.slice(0, 4),
  );
  const project = (p: Point): Point => applyHomography(toFrame, p.x, p.y);
  const keys: DetectedKey[] = [];
  for (let index = 0; index < read.totalKeys; index += 1) {
    const key = read.keyAt(index);
    if (key === null) {
      continue;
    }
    keys.push({
      black: key.black,
      bar: [
        project(key.bar[0]),
        project(key.bar[1]),
        project(key.bar[2]),
        project(key.bar[3]),
      ],
    });
  }
  return keys;
}

function luminance(data: Uint8ClampedArray, at: number): number {
  return 0.299 * data[at] + 0.587 * data[at + 1] + 0.114 * data[at + 2];
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

type Run = {
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

type Labeled = { readonly run: Run; readonly letter: number };

function labelGroups(groups: readonly Run[][]): Labeled[] {
  const labeled: Labeled[] = [];
  for (const group of groups) {
    const letters = group.length === 2 ? GROUP2_LETTERS : GROUP3_LETTERS;
    for (const [i, run] of group.entries()) {
      labeled.push({ run, letter: letters[i] });
    }
  }
  return labeled;
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

function intervalIndexOf(
  x: number,
  boundaries: readonly Boundary[],
): number | null {
  for (let i = 0; i < boundaries.length - 1; i += 1) {
    if (x >= boundaries[i].x && x < boundaries[i + 1].x) {
      return i;
    }
  }
  return null;
}

type Locked = {
  readonly offset: number;
  readonly agreement: number;
  readonly intervalRuns: ReadonlyMap<number, Run>;
};

/** Finds which walked white-key interval each black run's own left edge falls into (a run's
 * centre sits too close to a white-key boundary for some notes to tell the interval reliably,
 * but the left edge always lands well inside its own key), then takes the most common letter
 * offset those runs agree on. */
function lockPhase(
  labeled: readonly Labeled[],
  boundaries: readonly Boundary[],
): Locked | null {
  const intervalRuns = new Map<number, Run>();
  const intervalLetters = new Map<number, number>();
  for (const { run, letter } of labeled) {
    const i = intervalIndexOf(run.start, boundaries);
    if (i === null) {
      continue;
    }
    intervalRuns.set(i, run);
    intervalLetters.set(i, letter);
  }
  if (intervalLetters.size === 0) {
    return null;
  }
  const counts = new Map<number, number>();
  for (const [i, letter] of intervalLetters) {
    const offset = (((letter - i) % 7) + 7) % 7;
    counts.set(offset, (counts.get(offset) ?? 0) + 1);
  }
  let offset = 0;
  let best = 0;
  for (const [candidate, count] of counts) {
    if (count > best) {
      best = count;
      offset = candidate;
    }
  }
  return { offset, agreement: best / intervalLetters.size, intervalRuns };
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
    keys.push({
      black: false,
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

/** Finds the keys in a rectified strip: the white-key separator lines near the player's edge and
 * the black-key runs near the far edge, walked and refined rather than fit to one assumed grid,
 * so a leftover lens warp does not have to be modelled to be tolerated. */
export function detectKeys(strip: Strip): KeyRead {
  const blackProfile = bandProfile(strip, BLACK_BAND[0], BLACK_BAND[1]);
  const blackSplit = split(blackProfile);
  if (blackSplit.separation < MIN_BLACK_SEPARATION) {
    return {
      kind: "unsure",
      reason: "no readable black-key pattern",
      confidence: 0,
    };
  }
  const runs = darkRuns(blackProfile, blackSplit.at, MIN_RUN_WIDTH_PX);
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
  const seedWidth = (seed ?? strip.width / 30) / WITHIN_GROUP_UNIT_GAP;
  const separatorProfile = bandProfile(
    strip,
    SEPARATOR_BAND[0],
    SEPARATOR_BAND[1],
  );
  const dips = findDips(separatorProfile);
  const walked = walkBoundaries(seedWidth, strip.width, dips);
  const whiteKeys = walked.length - 1;
  if (whiteKeys < MIN_WHITE_KEYS || whiteKeys > MOST_WHITE_KEYS) {
    return {
      kind: "unsure",
      reason: `walked an unlikely ${whiteKeys} white keys`,
      confidence: 0.1,
    };
  }

  const labeled = labelGroups(groups);
  const locked = lockPhase(labeled, walked);
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

  const keys = buildKeys(
    adjusted,
    locked.offset,
    locked.intervalRuns,
    strip.height,
  );
  return {
    kind: "read",
    whiteKeys,
    totalKeys: keys.length,
    phase: LETTERS[letterOf(locked.offset, 0)],
    confidence,
    keyAt: (index) => keys[index] ?? null,
  };
}

let captureCanvas: HTMLCanvasElement | null = null;
const CAPTURE_WIDTH = 1600;

function captureSource(
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

/** How the keys are read: a few tries a second and a half apart, keeping the best of them, and
 * then never again until the keybed moves. Mirrors the board reader's own cadence, since both
 * read a picture that does not change once the keybed is held. */
const everyMs = 1500;
const tries = 6;
const enough = 0.9;
const settleAfterMs = 500;

export type KeyReader = {
  readonly last: () => KeyRead | null;
  readonly look: (
    frame: CanvasImageSource,
    quad: readonly Point[],
    size: Size,
    now: number,
  ) => void;
  readonly moved: (now: number) => void;
};

export function createKeyReader(): KeyReader {
  let at = 0;
  let taken = 0;
  let best: KeyRead | null = null;
  let dueAt: number | null = null;

  return {
    last: () => best,
    look: (frame, quad, size, now) => {
      if (dueAt !== null && now < dueAt) {
        return;
      }
      dueAt = null;
      const settled =
        taken >= tries || (best?.kind === "read" && best.confidence >= enough);
      if (settled || now - at < everyMs) {
        return;
      }
      at = now;
      taken += 1;
      const source = captureSource(frame, size);
      if (source === null) {
        return;
      }
      const strip = rectifyStrip(source, quad);
      if (strip === null) {
        return;
      }
      const found = detectKeys(strip);
      window.kvtKeyStrip = strip;
      window.kvtKeyRead = found;
      if (
        best === null ||
        best.kind === "unsure" ||
        (found.kind === "read" && found.confidence > best.confidence)
      ) {
        best = found;
      }
    },
    moved: (now) => {
      taken = 0;
      best = null;
      dueAt = now + settleAfterMs;
    },
  };
}
