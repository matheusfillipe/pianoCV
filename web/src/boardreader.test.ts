import { describe, expect, it } from "vitest";
import type { Picture } from "./board";
import { createBoardReader } from "./boardreader";
import { isBlack } from "./keys";
import {
  boardOf,
  type KeybedSpace,
  keyBand,
  keysOf,
  type PitchRange,
  type Size,
} from "./keyspace";

const range: PitchRange = { lowest: 36, highest: 96 };
const width = 900;
const height = 8;

const space: KeybedSpace = {
  onKeys: (along) => ({ x: along * width, y: height / 2 }),
};

function goodPicture(): Picture {
  const board = boardOf(range);
  const black = keysOf(range)
    .filter(isBlack)
    .map((pitch) => keyBand(pitch, board));
  const isDark = (along: number): boolean =>
    black.some((band) => along >= band.from && along < band.to);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let x = 0; x < width; x += 1) {
    const level = isDark(x / width) ? 28 : 226;
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

function unreadablePicture(): Picture {
  return {
    width,
    height,
    data: new Uint8ClampedArray(width * height * 4).fill(150),
    scale: 1,
  };
}

const size: Size = { width: 100, height: 100 };
const frame = {} as unknown as CanvasImageSource;

describe("createBoardReader", () => {
  it("stops trying once a read clears the agreement bar", () => {
    let calls = 0;
    const reader = createBoardReader(() => {
      calls += 1;
      return goodPicture();
    });
    reader.look(space, frame, size, 1500);
    reader.look(space, frame, size, 3000);
    reader.look(space, frame, size, 4500);
    expect(calls).toBe(1);
    expect(reader.board()?.lowest).toBe(range.lowest);
    expect(reader.board()?.highest).toBe(range.highest);
    expect(reader.last()?.kind).toBe("read");
  });

  it("gives up after its try budget when nothing clears the bar", () => {
    let calls = 0;
    const reader = createBoardReader(() => {
      calls += 1;
      return unreadablePicture();
    });
    for (let i = 1; i <= 10; i += 1) {
      reader.look(space, frame, size, i * 1500);
    }
    expect(calls).toBe(6);
    expect(reader.board()).toBeNull();
    expect(reader.last()?.kind).toBe("unsure");
  });

  it("takes a pinned range without touching the picture", () => {
    let calls = 0;
    const reader = createBoardReader(() => {
      calls += 1;
      return goodPicture();
    });
    const pinned: PitchRange = { lowest: 21, highest: 60 };
    reader.look(space, frame, size, 1500, pinned);
    expect(calls).toBe(0);
    expect(reader.board()).toEqual(boardOf(pinned));
  });
});
