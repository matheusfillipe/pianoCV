import type { Point } from "./homography";
import { isBlack, keyUnits } from "./keys";
import {
  type Board,
  type KeybedSpace,
  type PitchRange,
  spanInKeys,
} from "./keyspace";

/** A frame to read; it can be smaller than the display frame since colour
 * needs far less detail than the picture on screen. */
export type Picture = {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;
  /** How much of the frame's size this picture is. */
  readonly scale: number;
};

export type BoardRead =
  | {
      readonly kind: "read";
      readonly board: Board;
      /** Share of the readable samples whose colour matched the board found. */
      readonly agreement: number;
      /** Share of the samples that came back dark, and how deep into the
       * keybed they were taken. */
      readonly darkShare: number;
      readonly depth: number;
    }
  | { readonly kind: "unsure"; readonly reason: string };

/** How many points are read across the keys; even a sixty-key board still
 * gets several samples per key. */
const samples = 360;

/** Depths tried for the black-key read, since the far edge the detector finds
 * varies with the instrument's own panel. */
const blackDepths = [0.18, 0.26, 0.34, 0.42, 0.5];

/** Pixels a key must cover before its colour means anything: a keybed's far
 * end seen at an angle compresses to a smear that would otherwise read as
 * solid black. */
const leastPixelsPerKey = 3;

/** Splits the keys into stretches read independently, since light falls
 * unevenly across a room; a stretch whose colours don't separate says nothing
 * about where the keys are. */
const stretches = 6;
const leastSeparation = 0.4;

/** Share of the board that must agree before it is worth drawing; hands and a
 * smeared far end mean a keyboard rarely reads perfectly. */
const leastAgreement = 0.62;

/** Board sizes worth considering: an octave and a half up to a full concert
 * board. */
const fewestWhites = 12;
const mostWhites = 56;

const lowestPitch = 21;
const highestPitch = 108;

/** Colour can't tell which octave a board sits in, only its shape, so ties
 * are broken by centring on a full piano. */
const pianoCentre = (lowestPitch + highestPitch) / 2;

const whiteKeys: number[] = [];
for (let pitch = lowestPitch; pitch <= highestPitch; pitch += 1) {
  if (!isBlack(pitch)) {
    whiteKeys.push(pitch);
  }
}

/** How far a black key bleeds over the white beside it in the picture, fitted
 * rather than assumed since it depends on where the camera stands. */
const bleeds = [0, 0.1, 0.2, 0.3, 0.4];

const octaveSteps = 100;
const octaveWide = 7 * octaveSteps;
const octaves = new Map<number, Uint8Array>();

/** The black-key pattern for one octave, spread by the bleed the camera can't
 * see past; it repeats every seven white keys, so one octave answers for the
 * whole board. */
function darkOctave(bleed: number): Uint8Array {
  const held = octaves.get(bleed);
  if (held !== undefined) {
    return held;
  }
  const dark = new Uint8Array(octaveWide);
  for (let pitch = 60; pitch < 72; pitch += 1) {
    const units = keyUnits(pitch);
    if (units.to - units.from === 1) {
      continue;
    }
    const from = Math.round((units.from - bleed) * octaveSteps);
    const to = Math.round((units.to + bleed) * octaveSteps);
    for (let at = from; at < to; at += 1) {
      dark[((at % octaveWide) + octaveWide) % octaveWide] = 1;
    }
  }
  octaves.set(bleed, dark);
  return dark;
}

/** Otsu's split: the brightness cut that separates a stretch's readings
 * furthest apart, which on a keybed is the cut between black and white keys. */
function split(level: readonly number[]): {
  readonly at: number;
  readonly separation: number;
} {
  const bins = 64;
  const counts = new Array<number>(bins).fill(0);
  for (const value of level) {
    const bin = Math.min(
      bins - 1,
      Math.max(0, Math.floor((value / 256) * bins)),
    );
    counts[bin] = (counts[bin] ?? 0) + 1;
  }
  const total = level.length;
  let sum = 0;
  for (const [bin, count] of counts.entries()) {
    sum += bin * count;
  }
  let behind = 0;
  let behindSum = 0;
  let best = { at: 0, separation: 0 };
  let spread = 0;
  for (const [bin, count] of counts.entries()) {
    behind += count;
    if (behind === 0 || behind === total) {
      continue;
    }
    behindSum += bin * count;
    const ahead = total - behind;
    const between =
      (behind / total) *
      (ahead / total) *
      (behindSum / behind - (sum - behindSum) / ahead) ** 2;
    if (between > spread) {
      spread = between;
      best = { at: ((bin + 1) / bins) * 256, separation: 0 };
    }
  }
  const mean = sum / total;
  let variance = 0;
  for (const [bin, count] of counts.entries()) {
    variance += ((bin - mean) ** 2 * count) / total;
  }
  return {
    at: best.at,
    separation: variance === 0 ? 0 : spread / variance,
  };
}

function brightnessAt(picture: Picture, point: Point | null): number | null {
  if (point === null) {
    return null;
  }
  const x = Math.round(point.x * picture.scale);
  const y = Math.round(point.y * picture.scale);
  if (x < 1 || y < 1 || x >= picture.width - 1 || y >= picture.height - 1) {
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
    const at = ((y + (dy ?? 0)) * picture.width + x + (dx ?? 0)) * 4;
    total +=
      0.299 * (picture.data[at] ?? 0) +
      0.587 * (picture.data[at + 1] ?? 0) +
      0.114 * (picture.data[at + 2] ?? 0);
  }
  return total / 5;
}

type Stripe = {
  readonly seen: readonly (boolean | null)[];
  readonly darkShare: number;
  readonly depth: number;
};

/** What the camera reads across the keys at one depth: true where a black key
 * covers the sample, null where the picture has nothing to say. */
function readStripe(
  space: KeybedSpace,
  picture: Picture,
  depth: number,
): Stripe {
  const line: (Point | null)[] = [];
  for (let index = 0; index < samples; index += 1) {
    line.push(space.onKeys((index + 0.5) / samples, depth));
  }
  const perSample = spanInKeys / samples;
  const level = line.map((point, index) => {
    const before = line[Math.max(0, index - 1)];
    const after = line[Math.min(line.length - 1, index + 1)];
    if (point === null || before == null || after == null) {
      return null;
    }
    const step =
      (Math.hypot(after.x - before.x, after.y - before.y) * picture.scale) /
      (2 * perSample);
    return step < leastPixelsPerKey ? null : brightnessAt(picture, point);
  });
  const seen: (boolean | null)[] = new Array<boolean | null>(samples).fill(
    null,
  );
  let dark = 0;
  let known = 0;
  const wide = Math.ceil(samples / stretches);
  for (let from = 0; from < samples; from += wide) {
    const stretch = level.slice(from, from + wide);
    const read = stretch.flatMap((value) => (value === null ? [] : [value]));
    if (read.length < wide / 2) {
      continue;
    }
    const cut = split(read);
    if (cut.separation < leastSeparation) {
      continue;
    }
    for (const [index, value] of stretch.entries()) {
      if (value === null) {
        continue;
      }
      const black = value < cut.at;
      seen[from + index] = black;
      known += 1;
      if (black) {
        dark += 1;
      }
    }
  }
  return { seen, darkShare: known === 0 ? 0 : dark / known, depth };
}

/** How much of a reading a keyboard of this many white keys, starting this
 * far into an octave, explains. */
function agreementOf(
  stripe: readonly (boolean | null)[],
  phase: number,
  whites: number,
  dark: Uint8Array,
): number {
  let matched = 0;
  let known = 0;
  for (const [index, seen] of stripe.entries()) {
    if (seen === null) {
      continue;
    }
    known += 1;
    const units = phase + ((index + 0.5) / stripe.length) * whites;
    const at = Math.round(units * octaveSteps);
    const expected = dark[((at % octaveWide) + octaveWide) % octaveWide] === 1;
    if (expected === seen) {
      matched += 1;
    }
  }
  return known === 0 ? 0 : matched / known;
}

type Shape = {
  /** White keys across the keybed; need not be a whole number since the
   * corners rarely land exactly on a key's edge. */
  readonly whites: number;
  /** Where the keybed's near corner falls inside an octave, in white keys. */
  readonly phase: number;
  readonly bleed: number;
  readonly agreement: number;
};

/** How finely the fit is refined once the whole-key answer is found, in white
 * keys. */
const refineBy = 0.04;
/** How far the fit may walk: half a key either way, since which keyboard it
 * is was already settled by the whole-key answer. */
const refineBy2 = 0.5;

/** Walks the fit around the best whole-key answer, in both width and start,
 * since neither is really a whole number. */
function refine(stripe: Stripe, coarse: Shape): Shape {
  let best = coarse;
  const dark = darkOctave(coarse.bleed);
  for (
    let phase = coarse.phase - refineBy2;
    phase <= coarse.phase + refineBy2;
    phase += refineBy
  ) {
    for (
      let whites = coarse.whites - refineBy2;
      whites <= coarse.whites + refineBy2;
      whites += refineBy
    ) {
      if (whites < fewestWhites) {
        continue;
      }
      const agreement = agreementOf(stripe.seen, phase, whites, dark);
      if (agreement > best.agreement) {
        best = { whites, phase, bleed: coarse.bleed, agreement };
      }
    }
  }
  return best;
}

/** The board's size and start, which is all colour can say: one C reads
 * exactly like the next. */
function bestShape(stripe: Stripe, played: PitchRange | null): Shape | null {
  // Shapes worth trying don't depend on bleed, so we find them once and score
  // colour per bleed.
  const shapes: { whites: number; phase: number }[] = [];
  for (let whites = fewestWhites; whites <= mostWhites; whites += 1) {
    for (let phase = 0; phase < 7; phase += 1) {
      if (boardOfShape({ whites, phase, bleed: 0, agreement: 0 }, played)) {
        shapes.push({ whites, phase });
      }
    }
  }
  let best: Shape | null = null;
  for (const bleed of bleeds) {
    const dark = darkOctave(bleed);
    for (const { whites, phase } of shapes) {
      const agreement = agreementOf(stripe.seen, phase, whites, dark);
      if (best === null || agreement > best.agreement) {
        best = { whites, phase, bleed, agreement };
      }
    }
  }
  return best;
}

/** Which keyboard of that shape it is: every octave fits the colour equally,
 * so notes already played must fit inside it, and otherwise we centre on a
 * full piano. */
function boardOfShape(
  shape: Shape,
  played: PitchRange | null,
): PitchRange | null {
  let best: PitchRange | null = null;
  const whites = Math.round(shape.whites);
  const phase = ((Math.round(shape.phase) % 7) + 7) % 7;
  for (const [index, lowest] of whiteKeys.entries()) {
    const highest = whiteKeys[index + whites - 1];
    if (highest === undefined) {
      break;
    }
    if (
      ((keyUnits(lowest).from % 7) + 7) % 7 !== phase ||
      (played !== null && (played.lowest < lowest || played.highest > highest))
    ) {
      continue;
    }
    const range = { lowest, highest };
    if (best === null || nearerMiddle(range, best)) {
      best = range;
    }
  }
  return best;
}

/** Which keyboard is in front of the camera, read off its black keys: their
 * spacing gives the key count and phase, and every depth into the keybed is
 * tried since a low camera compresses the far edge unpredictably. Colour
 * alone can't tell one octave from another, so notes already played narrow
 * it, and otherwise we centre on a full piano. */
export function readBoard(
  space: KeybedSpace,
  picture: Picture,
  played: PitchRange | null = null,
): BoardRead {
  let best: { coarse: Shape; shape: Shape; stripe: Stripe } | null = null;
  let readable = false;
  const tried: string[] = [];
  for (const depth of blackDepths) {
    const stripe = readStripe(space, picture, depth);
    if (stripe.seen.filter((seen) => seen !== null).length < samples / 6) {
      continue;
    }
    readable = true;
    const coarse = bestShape(stripe, played);
    const shape = coarse === null ? null : refine(stripe, coarse);
    tried.push(
      `${depth}: ${percent(shape?.agreement ?? 0)} over ${shape?.whites ?? 0} keys, ${percent(stripe.darkShare)} dark`,
    );
    if (
      coarse !== null &&
      shape !== null &&
      (best === null || shape.agreement > best.shape.agreement)
    ) {
      best = { coarse, shape, stripe };
    }
  }
  if (!readable) {
    return { kind: "unsure", reason: "the keys are not in the picture" };
  }
  // The whole-key answer says which keyboard it is; refinement only says
  // where its keys fall.
  const range = best === null ? null : boardOfShape(best.coarse, played);
  if (
    best === null ||
    range === null ||
    best.shape.agreement < leastAgreement
  ) {
    return {
      kind: "unsure",
      reason: `the black keys read as a keyboard only ${percent(best?.shape.agreement ?? 0)} of the way (${tried.join("; ")})`,
    };
  }
  return {
    kind: "read",
    board: {
      ...range,
      origin:
        keyUnits(range.lowest).from +
        (best.shape.phase - Math.round(best.shape.phase)),
      span: best.shape.whites,
    },
    agreement: best.shape.agreement,
    darkShare: best.stripe.darkShare,
    depth: best.stripe.depth,
  };
}

function percent(share: number): string {
  return `${(share * 100).toFixed(0)}%`;
}

function middleOf(range: PitchRange): number {
  return (range.lowest + range.highest) / 2;
}

function nearerMiddle(one: PitchRange, other: PitchRange): boolean {
  return (
    Math.abs(middleOf(one) - pianoCentre) <
    Math.abs(middleOf(other) - pianoCentre)
  );
}
