import { describe, expect, it } from "vitest";
import { type Picture, readBoard } from "./board";
import type { Point } from "./homography";
import { isBlack } from "./keys";
import {
  boardOf,
  type Keybed,
  type KeybedSpace,
  keyBand,
  keybedSpace,
  keysOf,
  type PitchRange,
  type Size,
} from "./keyspace";

const frame: Size = { width: 960, height: 540 };
const keys = { left: 0.08, right: 0.92, far: 0.3, near: 0.74 };

const overhead: Keybed = {
  quad: [
    { x: keys.left, y: keys.far },
    { x: keys.right, y: keys.far },
    { x: keys.right, y: keys.near },
    { x: keys.left, y: keys.near },
  ],
};

/** A keyboard painted flat on, which is the picture an overhead camera would
 * take of one from straight above. */
function pictureOf(range: PitchRange): Picture {
  const data = new Uint8ClampedArray(frame.width * frame.height * 4);
  const left = keys.left * frame.width;
  const wide = (keys.right - keys.left) * frame.width;
  const far = keys.far * frame.height;
  const deep = (keys.near - keys.far) * frame.height;
  const black = keysOf(range)
    .filter(isBlack)
    .map((pitch) => keyBand(pitch, boardOf(range)));
  for (let y = 0; y < frame.height; y += 1) {
    for (let x = 0; x < frame.width; x += 1) {
      const along = (x - left) / wide;
      const across = (y - far) / deep;
      const inside = along >= 0 && along <= 1 && across >= 0 && across <= 1;
      const onBlack =
        inside &&
        across < 0.62 &&
        black.some((band) => along >= band.from && along < band.to);
      const level = inside ? (onBlack ? 28 : 226) : 90;
      const at = (y * frame.width + x) * 4;
      data[at] = level;
      data[at + 1] = level;
      data[at + 2] = level;
      data[at + 3] = 255;
    }
  }
  return { width: frame.width, height: frame.height, data, scale: 1 };
}

function read(
  range: PitchRange,
  played: PitchRange | null = null,
): PitchRange | string {
  const space = keybedSpace(overhead, frame);
  if (space === null) {
    throw new Error("A rectangle of the right shape has a pose");
  }
  const found = readBoard(space, pictureOf(range), played);
  return found.kind === "read"
    ? { lowest: found.board.lowest, highest: found.board.highest }
    : found.reason;
}

describe("readBoard", () => {
  it("reads a 36 white key board starting on C", () => {
    const range = { lowest: 36, highest: 96 };
    const space = keybedSpace(overhead, frame);
    if (space === null) {
      throw new Error("A rectangle of the right shape has a pose");
    }
    const found = readBoard(space, pictureOf(range));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.board.lowest).toBe(range.lowest);
    expect(found.board.highest).toBe(range.highest);
    expect(found.agreement).toBeGreaterThan(0.9);
  });

  it("reads the same span starting on F, a different phase", () => {
    expect(read({ lowest: 29, highest: 89 })).toEqual({
      lowest: 29,
      highest: 89,
    });
  });

  it("counts the keys rather than assuming them", () => {
    expect(read({ lowest: 28, highest: 103 })).toEqual({
      lowest: 28,
      highest: 103,
    });
  });

  it("takes the octave from the notes the player has sounded", () => {
    const high = { lowest: 48, highest: 108 };
    expect(read(high)).not.toEqual(high);
    expect(read(high, { lowest: 108, highest: 108 })).toEqual(high);
  });

  it("says so when the picture holds no keyboard", () => {
    const space = keybedSpace(overhead, frame);
    if (space === null) {
      throw new Error("A rectangle of the right shape has a pose");
    }
    const blank: Picture = {
      width: frame.width,
      height: frame.height,
      data: new Uint8ClampedArray(frame.width * frame.height * 4).fill(200),
      scale: 1,
    };
    expect(readBoard(space, blank).kind).toBe("unsure");
  });
});

describe("readBoard, compressed far end", () => {
  const range = { lowest: 36, highest: 96 };
  const nearWidth = 900;
  const farWidth = 4;
  const width = nearWidth + farWidth + 4;
  const height = 8;

  const stripSpace: KeybedSpace = {
    onKeys: (along): Point => {
      const x =
        along < 0.5
          ? along * 2 * nearWidth
          : nearWidth + (along - 0.5) * 2 * farWidth;
      return { x, y: height / 2 };
    },
  };

  function stripPicture(farLevel: number): Picture {
    const board = boardOf(range);
    const black = keysOf(range)
      .filter(isBlack)
      .map((pitch) => keyBand(pitch, board));
    const isDark = (along: number): boolean =>
      black.some((band) => along >= band.from && along < band.to);
    const data = new Uint8ClampedArray(width * height * 4);
    for (let x = 0; x < width; x += 1) {
      let level: number;
      if (x < nearWidth) {
        level = isDark(x / (2 * nearWidth)) ? 28 : 226;
      } else if (x < nearWidth + farWidth) {
        level = farLevel;
      } else {
        level = 128;
      }
      for (let y = 0; y < height; y += 1) {
        const at = (y * width + x) * 4;
        data[at] = level;
        data[at + 1] = level;
        data[at + 2] = level;
        data[at + 3] = 255;
      }
    }
    return { width, height, data, scale: 1 };
  }

  it("ignores samples compressed below 3 px per key rather than reading them as black", () => {
    const solidBlack = readBoard(stripSpace, stripPicture(28));
    const solidWhite = readBoard(stripSpace, stripPicture(226));
    expect(solidBlack).toEqual(solidWhite);
    expect(solidBlack.kind).toBe("read");
  });
});
