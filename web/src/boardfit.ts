import { applyHomography, findHomography, type Point } from "./homography";
import type { Board } from "./keypolygons";
import { isBlack, keyUnits } from "./keys";

const SAMPLES = 360;
const DEPTHS = [0.18, 0.26, 0.34, 0.42, 0.5];
const STRETCHES = 6;
const BLEEDS = [0, 0.1, 0.2, 0.3, 0.4];
const MIN_SEPARATION = 0.4;
const MIN_AGREEMENT = 0.62;
const MIN_WHITES = 12;
const MAX_WHITES = 56;
const DEFAULT_BLACK_DEPTH = 0.62;
const MIN_BLACK_DEPTH = 0.35;
const MAX_BLACK_DEPTH = 0.82;
const DEPTH_STEP = 0.02;
const DEPTH_PROBE = 0.035;
const OCTAVE_STEPS = 100;
const OCTAVE_WIDE = 7 * OCTAVE_STEPS;
const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const WHITE_PITCHES = Array.from(
  { length: 88 },
  (_, index) => index + 21,
).filter((pitch) => !isBlack(pitch));

type Stripe = {
  readonly seen: readonly (boolean | null)[];
};

type Shape = {
  readonly whites: number;
  readonly phase: number;
  readonly bleed: number;
  readonly agreement: number;
};

function brightness(pixels: ImageData, point: Point): number | null {
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 1 || y < 1 || x >= pixels.width - 1 || y >= pixels.height - 1) {
    return null;
  }
  let total = 0;
  for (const [dx, dy] of [
    [0, 0],
    [-1, 0],
    [1, 0],
    [0, -1],
    [0, 1],
  ]) {
    const at = ((y + dy) * pixels.width + x + dx) * 4;
    total +=
      0.299 * (pixels.data[at] ?? 0) +
      0.587 * (pixels.data[at + 1] ?? 0) +
      0.114 * (pixels.data[at + 2] ?? 0);
  }
  return total / 5;
}

function split(levels: readonly number[]): {
  readonly at: number;
  readonly separation: number;
} {
  const bins = 64;
  const counts = new Array<number>(bins).fill(0);
  for (const level of levels) {
    const bin = Math.min(
      bins - 1,
      Math.max(0, Math.floor((level / 256) * bins)),
    );
    counts[bin] = (counts[bin] ?? 0) + 1;
  }
  let sum = 0;
  for (const [bin, count] of counts.entries()) sum += bin * count;
  let behind = 0;
  let behindSum = 0;
  let spread = 0;
  let at = 0;
  for (const [bin, count] of counts.entries()) {
    behind += count;
    if (behind === 0 || behind === levels.length) continue;
    behindSum += bin * count;
    const ahead = levels.length - behind;
    const between =
      (behind / levels.length) *
      (ahead / levels.length) *
      (behindSum / behind - (sum - behindSum) / ahead) ** 2;
    if (between > spread) {
      spread = between;
      at = ((bin + 1) / bins) * 256;
    }
  }
  const mean = sum / levels.length;
  let variance = 0;
  for (const [bin, count] of counts.entries()) {
    variance += ((bin - mean) ** 2 * count) / levels.length;
  }
  return { at, separation: variance === 0 ? 0 : spread / variance };
}

function readStripe(
  pixels: ImageData,
  homography: number[],
  depth: number,
): Stripe {
  const levels = Array.from({ length: SAMPLES }, (_, index) =>
    brightness(
      pixels,
      applyHomography(homography, (index + 0.5) / SAMPLES, depth),
    ),
  );
  const seen: (boolean | null)[] = new Array(SAMPLES).fill(null);
  const wide = Math.ceil(SAMPLES / STRETCHES);
  for (let from = 0; from < SAMPLES; from += wide) {
    const stretch = levels.slice(from, from + wide);
    const readable = stretch.flatMap((level) =>
      level === null ? [] : [level],
    );
    if (readable.length < wide / 2) continue;
    const cut = split(readable);
    if (cut.separation < MIN_SEPARATION) continue;
    for (const [index, level] of stretch.entries()) {
      if (level !== null) seen[from + index] = level < cut.at;
    }
  }
  return { seen };
}

const octaveCache = new Map<number, Uint8Array>();

function darkOctave(bleed: number): Uint8Array {
  const cached = octaveCache.get(bleed);
  if (cached !== undefined) return cached;
  const dark = new Uint8Array(OCTAVE_WIDE);
  for (let pitch = 60; pitch < 72; pitch += 1) {
    if (!isBlack(pitch)) continue;
    const key = keyUnits(pitch);
    const from = Math.round((key.from - bleed) * OCTAVE_STEPS);
    const to = Math.round((key.to + bleed) * OCTAVE_STEPS);
    for (let unit = from; unit < to; unit += 1) {
      dark[((unit % OCTAVE_WIDE) + OCTAVE_WIDE) % OCTAVE_WIDE] = 1;
    }
  }
  octaveCache.set(bleed, dark);
  return dark;
}

function agreementOf(
  seen: readonly (boolean | null)[],
  phase: number,
  whites: number,
  dark: Uint8Array,
): number {
  let matched = 0;
  let known = 0;
  for (const [index, black] of seen.entries()) {
    if (black === null) continue;
    const unit = phase + ((index + 0.5) / seen.length) * whites;
    const at = Math.round(unit * OCTAVE_STEPS);
    const expected =
      dark[((at % OCTAVE_WIDE) + OCTAVE_WIDE) % OCTAVE_WIDE] === 1;
    known += 1;
    if (black === expected) matched += 1;
  }
  return known === 0 ? 0 : matched / known;
}

function rangeFor(whites: number, phase: number): Board | null {
  let best: Board | null = null;
  for (const [index, lowest] of WHITE_PITCHES.entries()) {
    const highest = WHITE_PITCHES[index + whites - 1];
    if (highest === undefined || keyUnits(lowest).from % 7 !== phase) continue;
    const candidate = {
      lowest,
      highest,
      origin: keyUnits(lowest).from,
      span: whites,
      blackDepth: DEFAULT_BLACK_DEPTH,
    };
    if (
      best === null ||
      Math.abs((lowest + highest) / 2 - 64.5) <
        Math.abs((best.lowest + best.highest) / 2 - 64.5)
    ) {
      best = candidate;
    }
  }
  return best;
}

function bestShape(stripe: Stripe): Shape | null {
  let best: Shape | null = null;
  for (let whites = MIN_WHITES; whites <= MAX_WHITES; whites += 1) {
    for (let phase = 0; phase < 7; phase += 1) {
      if (rangeFor(whites, phase) === null) continue;
      for (const bleed of BLEEDS) {
        const agreement = agreementOf(
          stripe.seen,
          phase,
          whites,
          darkOctave(bleed),
        );
        if (best === null || agreement > best.agreement) {
          best = { whites, phase, bleed, agreement };
        }
      }
    }
  }
  return best;
}

function refine(stripe: Stripe, coarse: Shape): Shape {
  let best = coarse;
  const dark = darkOctave(coarse.bleed);
  for (
    let phase = coarse.phase - 0.5;
    phase <= coarse.phase + 0.5;
    phase += 0.04
  ) {
    for (
      let whites = coarse.whites - 0.5;
      whites <= coarse.whites + 0.5;
      whites += 0.04
    ) {
      if (whites < MIN_WHITES) continue;
      const agreement = agreementOf(stripe.seen, phase, whites, dark);
      if (agreement > best.agreement) {
        best = { whites, phase, bleed: coarse.bleed, agreement };
      }
    }
  }
  return best;
}

function fitBlackDepth(
  pixels: ImageData,
  homography: number[],
  board: Board,
): number {
  const blackCenters = Array.from(
    { length: board.highest - board.lowest + 1 },
    (_, index) => board.lowest + index,
  )
    .filter(isBlack)
    .map((pitch) => {
      const key = keyUnits(pitch);
      return (key.from + key.to - 2 * board.origin) / (2 * board.span);
    })
    .filter((u) => u > 0.04 && u < 0.96);
  let best = { depth: DEFAULT_BLACK_DEPTH, contrast: -Infinity };
  for (
    let depth = MIN_BLACK_DEPTH;
    depth <= MAX_BLACK_DEPTH;
    depth += DEPTH_STEP
  ) {
    let contrast = 0;
    let known = 0;
    for (const u of blackCenters) {
      const before = brightness(
        pixels,
        applyHomography(homography, u, depth - DEPTH_PROBE),
      );
      const after = brightness(
        pixels,
        applyHomography(homography, u, depth + DEPTH_PROBE),
      );
      if (before === null || after === null) continue;
      contrast += after - before;
      known += 1;
    }
    if (known >= 6 && contrast / known > best.contrast) {
      best = { depth, contrast: contrast / known };
    }
  }
  return best.contrast >= 18 ? best.depth : DEFAULT_BLACK_DEPTH;
}

export function fitBoard(
  pixels: ImageData,
  quad: readonly Point[],
): Board | null {
  const homography = findHomography(UNIT_SQUARE, quad);
  let best: { readonly coarse: Shape; readonly shape: Shape } | null = null;
  for (const depth of DEPTHS) {
    const stripe = readStripe(pixels, homography, depth);
    if (stripe.seen.filter((seen) => seen !== null).length < SAMPLES / 6) {
      continue;
    }
    const coarse = bestShape(stripe);
    if (coarse === null) continue;
    const shape = refine(stripe, coarse);
    if (best === null || shape.agreement > best.shape.agreement) {
      best = { coarse, shape };
    }
  }
  if (best === null || best.shape.agreement < MIN_AGREEMENT) return null;
  const board = rangeFor(
    Math.round(best.coarse.whites),
    Math.round(best.coarse.phase),
  );
  if (board === null) return null;
  return {
    ...board,
    origin:
      keyUnits(board.lowest).from +
      (best.shape.phase - Math.round(best.shape.phase)),
    span: best.shape.whites,
    blackDepth: fitBlackDepth(pixels, homography, {
      ...board,
      origin:
        keyUnits(board.lowest).from +
        (best.shape.phase - Math.round(best.shape.phase)),
      span: best.shape.whites,
      blackDepth: DEFAULT_BLACK_DEPTH,
    }),
  };
}
