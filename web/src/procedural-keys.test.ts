import { describe, expect, it } from "vitest";
import { mmToUnits, WHITE_KEY_WIDTH_MM } from "./keybed3d";
import { buildKeyboard, sampleKeyVariation } from "./procedural-keys";

const STANDARD_BOARDS = [
  { lowestPitch: 36, keys: 49, whiteKeys: 29 },
  { lowestPitch: 36, keys: 61, whiteKeys: 36 },
  { lowestPitch: 28, keys: 76, whiteKeys: 45 },
  { lowestPitch: 21, keys: 88, whiteKeys: 52 },
];

const BLACK_TEMPLATE_OFFSETS: Record<number, number> = {
  1: 0.6,
  3: 0.75,
  6: 0.6,
  8: 0.63,
  10: 0.66,
};

describe("buildKeyboard", () => {
  const midpoint = sampleKeyVariation(() => 0.5);

  it("generates the right key count, first pitch and white count for every standard board", () => {
    for (const board of STANDARD_BOARDS) {
      const keyboard = buildKeyboard(board, midpoint);
      expect(keyboard.keys).toHaveLength(board.keys);
      expect(keyboard.keys[0]?.pitch).toBe(board.lowestPitch);
      expect(keyboard.whiteKeys).toBe(board.whiteKeys);
    }
  });

  it("keeps pitches contiguous", () => {
    const board = STANDARD_BOARDS[2];
    const keyboard = buildKeyboard(board, midpoint);
    for (const [i, key] of keyboard.keys.entries()) {
      expect(key.pitch).toBe(board.lowestPitch + i);
    }
  });

  it("never places a black key right after E or B", () => {
    const keyboard = buildKeyboard(STANDARD_BOARDS[3], midpoint);
    for (const key of keyboard.keys) {
      const pc = ((key.pitch % 12) + 12) % 12;
      if (pc === 4 || pc === 11) {
        const next = keyboard.keys.find((k) => k.pitch === key.pitch + 1);
        expect(next?.black).toBeFalsy();
      }
    }
  });

  it("spaces white keys uniformly", () => {
    const keyboard = buildKeyboard(STANDARD_BOARDS[1], midpoint);
    const whites = keyboard.keys.filter((key) => !key.black);
    const whiteUnit = mmToUnits(WHITE_KEY_WIDTH_MM);
    for (const key of whites) {
      expect(key.body.size[2]).toBeCloseTo(whites[0].body.size[2]);
    }
    for (let i = 1; i < whites.length; i += 1) {
      const spacing = whites[i].body.center[2] - whites[i - 1].body.center[2];
      expect(spacing).toBeCloseTo(whiteUnit);
    }
  });

  it("places every black key within its template's jittered offset, at the sampled width", () => {
    const whiteUnit = mmToUnits(WHITE_KEY_WIDTH_MM);
    for (let i = 0; i < 30; i += 1) {
      const seed = i / 30;
      const variation = sampleKeyVariation(() => seed);
      const keyboard = buildKeyboard(STANDARD_BOARDS[3], variation);
      for (const key of keyboard.keys) {
        if (!key.black) {
          continue;
        }
        const preceding = keyboard.keys.find((k) => k.pitch === key.pitch - 1);
        if (!preceding) {
          continue;
        }
        const blackLeft = key.body.center[2] - key.body.size[2] / 2;
        const precedingSlotLeft = preceding.body.center[2] - whiteUnit / 2;
        const offset = (blackLeft - precedingSlotLeft) / whiteUnit;
        const pc = ((key.pitch % 12) + 12) % 12;
        const template = BLACK_TEMPLATE_OFFSETS[pc];
        expect(offset).toBeGreaterThanOrEqual(template - 0.021);
        expect(offset).toBeLessThanOrEqual(template + 0.021);
        expect(key.body.size[2] / whiteUnit).toBeCloseTo(
          variation.blackWidthFrac,
        );
      }
    }
  });
});
