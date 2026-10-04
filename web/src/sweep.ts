import type { PitchRange } from "./keyspace";
import { whiteKeysOf } from "./keyspace";

export interface Range {
  readonly min: number;
  readonly max: number;
}

export function between(random: () => number, range: Range): number {
  return range.min + random() * (range.max - range.min);
}

// the pose, lens and framing ranges a person actually films a keyboard from, named once so the
// covered space is legible rather than buried in scattered random() calls
export const SWEEP_ELEVATION_DEG: Range = { min: 3, max: 85 };
// azimuth 0 faces the keys head on and +-90 looks straight down the keybed from one end; people
// film past the ends too, from a rear corner, so the sweep runs past +-90 both ways
export const SWEEP_AZIMUTH_DEG: Range = { min: -130, max: 130 };
// a factor of the board's own span, so a 61-key board is filmed from proportionally closer than
// an 88-key one rather than sharing one absolute distance
export const SWEEP_DISTANCE_FACTOR: Range = { min: 0.35, max: 1.8 };
export const SWEEP_ROLL_DEG: Range = { min: -30, max: 30 };
// the runtime holds its lens fixed, but phones and webcams span wide-angle to telephoto
export const SWEEP_FOV_DEG: Range = { min: 15, max: 95 };
export const SWEEP_OFFSET_X_FACTOR: Range = { min: -0.9, max: 0.9 };
export const SWEEP_OFFSET_Y_FACTOR: Range = { min: -0.4, max: 0.4 };

export interface SweepPose {
  readonly elevationDeg: number;
  readonly azimuthDeg: number;
  readonly distanceFactor: number;
  readonly rollDeg: number;
  readonly fovDeg: number;
  readonly offsetXFactor: number;
  readonly offsetYFactor: number;
}

export function samplePose(random: () => number): SweepPose {
  return {
    elevationDeg: between(random, SWEEP_ELEVATION_DEG),
    azimuthDeg: between(random, SWEEP_AZIMUTH_DEG),
    distanceFactor: between(random, SWEEP_DISTANCE_FACTOR),
    rollDeg: between(random, SWEEP_ROLL_DEG),
    fovDeg: between(random, SWEEP_FOV_DEG),
    offsetXFactor: between(random, SWEEP_OFFSET_X_FACTOR),
    offsetYFactor: between(random, SWEEP_OFFSET_Y_FACTOR),
  };
}

const NOTE_NAMES = [
  "C",
  "C#",
  "D",
  "D#",
  "E",
  "F",
  "F#",
  "G",
  "G#",
  "A",
  "A#",
  "B",
];

export function noteName(pitch: number): string {
  return `${NOTE_NAMES[((pitch % 12) + 12) % 12]}${Math.floor(pitch / 12) - 1}`;
}

export interface BoardSize {
  readonly keys: number;
  readonly whiteKeys: number;
  readonly lowestNote: string;
  readonly range: PitchRange;
}

function boardSize(keys: number, range: PitchRange): BoardSize {
  return {
    keys,
    whiteKeys: whiteKeysOf(range),
    lowestNote: noteName(range.lowest),
    range,
  };
}

// the keyboard sizes a piano video is actually filmed on: the runtime assumes one fixed size,
// and the owner's own instrument turned out to be a different one
export const BOARD_SIZES: readonly BoardSize[] = [
  boardSize(49, { lowest: 36, highest: 84 }),
  boardSize(61, { lowest: 36, highest: 96 }),
  boardSize(76, { lowest: 28, highest: 103 }),
  boardSize(88, { lowest: 21, highest: 108 }),
];

export function pickBoardSize(random: () => number): BoardSize {
  return (
    BOARD_SIZES[Math.floor(random() * BOARD_SIZES.length)] ?? BOARD_SIZES[0]
  );
}

// present in most frames so the model mostly sees a case, and sometimes just floating keys,
// the way the model looked before a case existed at all
export const CASE_PRESENCE_PROBABILITY = 0.9;
export const CASE_BACK_DEPTH_MM: Range = { min: 60, max: 250 };
export const CASE_TOP_STANDOFF_MM: Range = { min: 10, max: 150 };
export const CASE_CHEEK_WIDTH_MM: Range = { min: 20, max: 80 };

export type CaseColorFamily =
  | "black"
  | "darkGrey"
  | "silver"
  | "white"
  | "wood";

export interface CaseColorSpec {
  readonly family: CaseColorFamily;
  readonly weight: number;
  readonly hue: Range;
  readonly saturation: Range;
  readonly lightness: Range;
  readonly roughness: Range;
  readonly metalness: Range;
}

// real keyboard cases are almost always black or dark grey plastic; silver, white and wood
// (digital pianos styled as furniture) show up too, just less often
const CASE_COLOR_FAMILIES: readonly CaseColorSpec[] = [
  {
    family: "black",
    weight: 0.4,
    hue: { min: 0, max: 1 },
    saturation: { min: 0, max: 0.05 },
    lightness: { min: 0.02, max: 0.08 },
    roughness: { min: 0.2, max: 0.85 },
    metalness: { min: 0, max: 0.15 },
  },
  {
    family: "darkGrey",
    weight: 0.3,
    hue: { min: 0, max: 1 },
    saturation: { min: 0, max: 0.08 },
    lightness: { min: 0.12, max: 0.24 },
    roughness: { min: 0.2, max: 0.85 },
    metalness: { min: 0, max: 0.15 },
  },
  {
    family: "silver",
    weight: 0.1,
    hue: { min: 0, max: 1 },
    saturation: { min: 0, max: 0.05 },
    lightness: { min: 0.55, max: 0.75 },
    roughness: { min: 0.1, max: 0.4 },
    metalness: { min: 0.5, max: 0.9 },
  },
  {
    family: "white",
    weight: 0.1,
    hue: { min: 0, max: 1 },
    saturation: { min: 0, max: 0.03 },
    lightness: { min: 0.85, max: 0.96 },
    roughness: { min: 0.25, max: 0.7 },
    metalness: { min: 0, max: 0.05 },
  },
  {
    family: "wood",
    weight: 0.1,
    hue: { min: 0.05, max: 0.09 },
    saturation: { min: 0.35, max: 0.6 },
    lightness: { min: 0.25, max: 0.45 },
    roughness: { min: 0.35, max: 0.7 },
    metalness: { min: 0, max: 0.05 },
  },
];

export function pickCaseColorFamily(random: () => number): CaseColorSpec {
  const total = CASE_COLOR_FAMILIES.reduce((sum, spec) => sum + spec.weight, 0);
  let pick = random() * total;
  for (const spec of CASE_COLOR_FAMILIES) {
    if (pick < spec.weight) {
      return spec;
    }
    pick -= spec.weight;
  }
  return CASE_COLOR_FAMILIES[CASE_COLOR_FAMILIES.length - 1];
}
