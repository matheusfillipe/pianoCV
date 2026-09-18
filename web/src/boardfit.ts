import { applyHomography, findHomography, type Point } from "./homography";
import type { BlackKeyGeometry, Board } from "./keypolygons";
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
const MIN_BLACK_DEPTH = 0.32;
const MAX_BLACK_DEPTH = 0.82;
const BLACK_DEPTH_STEP = 0.01;
const BLACK_EDGE_PROBE = 0.025;
const LOCAL_BLACK_EDGE = 0.035;
const LOCAL_BLACK_MIN_SCORE = 18;
const LOCAL_BLACK_OFFSETS = [-0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06];
const LOCAL_BLACK_WIDTHS = [0.82, 0.91, 1, 1.09, 1.18];
const OCTAVE_STEPS = 100;
const OCTAVE_WIDE = 7 * OCTAVE_STEPS;
const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const WHITE_PITCHES = Array.from({ length: 128 }, (_, index) => index).filter(
  (pitch) => !isBlack(pitch),
);

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
    // Low pitches can have negative coordinates. Normalize the remainder
    // before comparing it with the stripe phase; JavaScript's `%` keeps the
    // sign of the dividend.
    const keyPhase = ((keyUnits(lowest).from % 7) + 7) % 7;
    if (highest === undefined || keyPhase !== phase) continue;
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

function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
}

function fitBlackDepth(
  pixels: ImageData,
  homography: number[],
  board: Board,
): number {
  const centres = Array.from(
    { length: board.highest - board.lowest + 1 },
    (_, index) => board.lowest + index,
  )
    .filter(isBlack)
    .flatMap((pitch) => {
      const key = keyUnits(pitch);
      const middle = (key.from + key.to) / 2;
      const half = (key.to - key.from) * 0.18;
      return [middle - half, middle, middle + half];
    })
    .map((unit) => (unit - board.origin) / board.span)
    .filter((u) => u > 0.04 && u < 0.96);
  let bestDepth = DEFAULT_BLACK_DEPTH;
  let bestContrast = -Infinity;
  for (
    let depth = MIN_BLACK_DEPTH;
    depth <= MAX_BLACK_DEPTH;
    depth += BLACK_DEPTH_STEP
  ) {
    const contrasts: number[] = [];
    for (const u of centres) {
      const before = brightness(
        pixels,
        applyHomography(homography, u, depth - BLACK_EDGE_PROBE),
      );
      const after = brightness(
        pixels,
        applyHomography(homography, u, depth + BLACK_EDGE_PROBE),
      );
      if (before !== null && after !== null) contrasts.push(after - before);
    }
    const contrast = median(contrasts);
    if (contrast !== null && contrast > bestContrast) {
      bestContrast = contrast;
      bestDepth = depth;
    }
  }
  return bestContrast >= 12 ? bestDepth : DEFAULT_BLACK_DEPTH;
}

function localBlackScore(
  pixels: ImageData,
  homography: number[],
  u0: number,
  u1: number,
  depth: number,
): number | null {
  const centre = (u0 + u1) / 2;
  const inside = brightness(
    pixels,
    applyHomography(homography, centre, depth * 0.65),
  );
  const left = brightness(
    pixels,
    applyHomography(homography, u0 - LOCAL_BLACK_EDGE, depth * 0.65),
  );
  const right = brightness(
    pixels,
    applyHomography(homography, u1 + LOCAL_BLACK_EDGE, depth * 0.65),
  );
  const before = brightness(
    pixels,
    applyHomography(homography, centre, depth - BLACK_EDGE_PROBE),
  );
  const after = brightness(
    pixels,
    applyHomography(homography, centre, depth + BLACK_EDGE_PROBE),
  );
  if (inside === null || left === null || right === null) return null;
  const sides = (left + right) / 2 - inside;
  const front = before !== null && after !== null ? after - before : 0;
  return sides + front;
}

function fitBlackKeys(
  pixels: ImageData,
  homography: number[],
  board: Board,
  fallbackDepth: number,
): BlackKeyGeometry[] {
  const fitted: BlackKeyGeometry[] = [];
  let previousU1 = -Infinity;
  for (let pitch = board.lowest; pitch <= board.highest; pitch += 1) {
    if (!isBlack(pitch)) continue;
    const units = keyUnits(pitch);
    const expectedU0 = (units.from - board.origin) / board.span;
    const expectedU1 = (units.to - board.origin) / board.span;
    const expectedWidth = expectedU1 - expectedU0;
    let best: {
      readonly u0: number;
      readonly u1: number;
      readonly depth: number;
      readonly score: number;
    } | null = null;
    for (const offset of LOCAL_BLACK_OFFSETS) {
      for (const widthScale of LOCAL_BLACK_WIDTHS) {
        const width = expectedWidth * widthScale;
        const centre = (expectedU0 + expectedU1) / 2 + offset / board.span;
        const u0 = Math.max(0.01, centre - width / 2);
        const u1 = Math.min(0.99, centre + width / 2);
        if (u1 <= u0) continue;
        for (
          let depth = MIN_BLACK_DEPTH;
          depth <= MAX_BLACK_DEPTH;
          depth += BLACK_DEPTH_STEP * 4
        ) {
          const evidence = localBlackScore(pixels, homography, u0, u1, depth);
          if (evidence === null) continue;
          const prior = Math.abs(widthScale - 1) * 8;
          const score = evidence - prior;
          if (best === null || score > best.score) {
            best = { u0, u1, depth, score };
          }
        }
      }
    }
    const candidate = best;
    const usable =
      candidate !== null && candidate.score >= LOCAL_BLACK_MIN_SCORE;
    let u0 = usable && candidate !== null ? candidate.u0 : expectedU0;
    let u1 = usable && candidate !== null ? candidate.u1 : expectedU1;
    const depth =
      usable && candidate !== null ? candidate.depth : fallbackDepth;
    const minimumGap = expectedWidth * 0.18;
    if (u0 <= previousU1 + minimumGap) {
      const shift = previousU1 + minimumGap - u0;
      u0 += shift;
      u1 += shift;
    }
    if (u1 > 1 || u1 <= u0) {
      u0 = expectedU0;
      u1 = expectedU1;
    }
    fitted.push({
      pitch,
      u0,
      u1,
      depth,
      confidence:
        usable && candidate !== null ? Math.min(1, candidate.score / 180) : 0,
    });
    previousU1 = u1;
  }
  return fitted;
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
  const origin =
    keyUnits(board.lowest).from +
    (best.shape.phase - Math.round(best.shape.phase));
  const fitted = {
    ...board,
    origin,
    span: best.shape.whites,
    blackDepth: DEFAULT_BLACK_DEPTH,
  };
  const blackDepth = fitBlackDepth(pixels, homography, fitted);
  return {
    ...fitted,
    blackDepth,
    blackKeys: fitBlackKeys(pixels, homography, fitted, blackDepth),
  };
}
