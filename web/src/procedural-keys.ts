import { Vector3 } from "three";
import {
  BACK_X,
  DEPTH,
  FRONT_X,
  KEY_BOTTOM_Y,
  KEY_TOP_Y,
  mmToUnits,
  WHITE_KEY_WIDTH_MM,
} from "./keybed3d";
import { isBlack, whiteIndex } from "./keys";
import { between, type Range } from "./sweep";

export interface KeyboardBoard {
  readonly lowestPitch: number;
  readonly keys: number;
}

export interface KeyVariation {
  readonly whiteWidthScale: number;
  readonly blackWidthFrac: number;
  readonly blackOffsetJitter: readonly [number, number, number, number, number];
  readonly blackHeightMm: number;
  readonly blackLengthFrac: number;
  readonly whiteGapMm: number;
  readonly bevelMm: number;
}

export const WHITE_WIDTH_SCALE: Range = { min: 0.92, max: 1.08 };
export const BLACK_WIDTH_FRAC: Range = { min: 0.5, max: 0.66 };
export const BLACK_OFFSET_JITTER: Range = { min: -0.03, max: 0.03 };
export const BLACK_HEIGHT_MM: Range = { min: 5, max: 16 };
export const BLACK_LENGTH_FRAC: Range = { min: 0.55, max: 0.72 };
export const WHITE_GAP_MM: Range = { min: 0.3, max: 2.4 };
export const BEVEL_MM: Range = { min: 0.3, max: 2 };

export function sampleKeyVariation(random: () => number): KeyVariation {
  return {
    whiteWidthScale: between(random, WHITE_WIDTH_SCALE),
    blackWidthFrac: between(random, BLACK_WIDTH_FRAC),
    blackOffsetJitter: [
      between(random, BLACK_OFFSET_JITTER),
      between(random, BLACK_OFFSET_JITTER),
      between(random, BLACK_OFFSET_JITTER),
      between(random, BLACK_OFFSET_JITTER),
      between(random, BLACK_OFFSET_JITTER),
    ],
    blackHeightMm: between(random, BLACK_HEIGHT_MM),
    blackLengthFrac: between(random, BLACK_LENGTH_FRAC),
    whiteGapMm: between(random, WHITE_GAP_MM),
    bevelMm: between(random, BEVEL_MM),
  };
}

interface BlackTemplate {
  readonly pc: number;
  readonly offset: number;
}

// a black key's offset from the left edge of the white key it starts in, in white-key widths;
// real pianos place each black key differently rather than centring it over the white join
const BLACK_TEMPLATES: readonly BlackTemplate[] = [
  { pc: 1, offset: 0.6 }, // C#
  { pc: 3, offset: 0.75 }, // D#
  { pc: 6, offset: 0.6 }, // F#
  { pc: 8, offset: 0.63 }, // G#
  { pc: 10, offset: 0.66 }, // A#
];

export interface KeyMeshBox {
  readonly size: readonly [number, number, number];
  readonly center: readonly [number, number, number];
}

export interface KeyGeometry {
  readonly pitch: number;
  readonly black: boolean;
  readonly body: KeyMeshBox;
  readonly cap: KeyMeshBox;
  readonly topCorners: readonly Vector3[];
  readonly frontCorners: readonly Vector3[] | null;
}

export interface Keyboard {
  readonly board: KeyboardBoard;
  readonly whiteKeys: number;
  readonly variation: KeyVariation;
  readonly keys: readonly KeyGeometry[];
  readonly minZ: number;
  readonly maxZ: number;
}

function meshBox(
  minX: number,
  maxX: number,
  minY: number,
  maxY: number,
  minZ: number,
  maxZ: number,
): KeyMeshBox {
  return {
    size: [maxX - minX, maxY - minY, maxZ - minZ],
    center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
  };
}

// corner 0 to 1 runs along the back edge from the low side, 1 to 2 crosses the depth, matching
// keybed3d's own cornersFor convention
function topCornersOf(
  minX: number,
  maxX: number,
  topY: number,
  lowZ: number,
  highZ: number,
): Vector3[] {
  return [
    new Vector3(minX, topY, lowZ),
    new Vector3(minX, topY, highZ),
    new Vector3(maxX, topY, highZ),
    new Vector3(maxX, topY, lowZ),
  ];
}

function frontCornersOf(
  frontX: number,
  topY: number,
  lowZ: number,
  highZ: number,
): Vector3[] {
  return [
    new Vector3(frontX, topY, lowZ),
    new Vector3(frontX, topY, highZ),
    new Vector3(frontX, KEY_TOP_Y, highZ),
    new Vector3(frontX, KEY_TOP_Y, lowZ),
  ];
}

// white keys are drawn full width, front to back, under the black keys rather than notched
// around them; from above (the camera never looks from below the black keys) this reads the same
// as an interlocked white key and needs one box per key instead of two.
export function buildKeyboard(
  board: KeyboardBoard,
  variation: KeyVariation,
): Keyboard {
  const whiteUnit = mmToUnits(WHITE_KEY_WIDTH_MM) * variation.whiteWidthScale;
  const gapFrac =
    variation.whiteGapMm / (WHITE_KEY_WIDTH_MM * variation.whiteWidthScale);
  const bevelUnits = mmToUnits(variation.bevelMm);
  const blackHeightUnits = mmToUnits(variation.blackHeightMm);
  const blackFrontX = BACK_X + variation.blackLengthFrac * DEPTH;
  const originIndex = whiteIndex(board.lowestPitch);
  const highestPitch = board.lowestPitch + board.keys - 1;
  const whiteKeys = whiteIndex(highestPitch) - originIndex + 1;
  const centerFrac = whiteKeys / 2;
  // the camera faces the keys from +x, which puts -z on the player's right, so pitch rises
  // toward -z for the low keys to land on the left as on a real keyboard
  const zOf = (frac: number): number => (centerFrac - frac) * whiteUnit;

  const keys: KeyGeometry[] = [];
  for (let pitch = board.lowestPitch; pitch <= highestPitch; pitch += 1) {
    const black = isBlack(pitch);
    let leftFrac: number;
    let rightFrac: number;
    if (black) {
      const pc = ((pitch % 12) + 12) % 12;
      const slot = BLACK_TEMPLATES.findIndex((template) => template.pc === pc);
      const template = BLACK_TEMPLATES[slot];
      const baseIndex = whiteIndex(pitch - 1) - originIndex;
      leftFrac =
        baseIndex + template.offset + variation.blackOffsetJitter[slot];
      rightFrac = leftFrac + variation.blackWidthFrac;
    } else {
      const index = whiteIndex(pitch) - originIndex;
      leftFrac = index + gapFrac / 2;
      rightFrac = index + 1 - gapFrac / 2;
    }
    const minZ = zOf(rightFrac);
    const maxZ = zOf(leftFrac);
    const minX = BACK_X;
    const maxX = black ? blackFrontX : FRONT_X;
    const topY = black ? KEY_TOP_Y + blackHeightUnits : KEY_TOP_Y;
    const bottomY = black ? KEY_TOP_Y : KEY_BOTTOM_Y;
    const bevelY = topY - bevelUnits;

    keys.push({
      pitch,
      black,
      body: meshBox(minX, maxX, bottomY, bevelY, minZ, maxZ),
      cap: meshBox(
        minX + bevelUnits,
        maxX - bevelUnits,
        bevelY,
        topY,
        minZ + bevelUnits,
        maxZ - bevelUnits,
      ),
      topCorners: topCornersOf(minX, maxX, topY, maxZ, minZ),
      frontCorners: black ? frontCornersOf(maxX, topY, maxZ, minZ) : null,
    });
  }

  let minZ = Number.POSITIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;
  for (const key of keys) {
    minZ = Math.min(minZ, key.body.center[2] - key.body.size[2] / 2);
    maxZ = Math.max(maxZ, key.body.center[2] + key.body.size[2] / 2);
  }

  return { board, whiteKeys, variation, keys, minZ, maxZ };
}
