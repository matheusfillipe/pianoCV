import { describe, expect, it } from "vitest";
import {
  CROP_HEIGHT,
  CROP_WIDTH,
  cropFor,
  cropToFrame,
  keyOutline,
  snapKeys,
} from "./keyseg";

describe("cropFor", () => {
  it("crops exactly as the training crop does", () => {
    const quad = [
      { x: 120, y: 300 },
      { x: 520, y: 180 },
      { x: 560, y: 260 },
      { x: 150, y: 400 },
    ];
    // where tools/src/pianocv/keyseg.py's crop_for puts these corners in the crop
    const inCrop = [
      [232.02915148360225, 42.96928682977614],
      [784.2748568453931, 53.541558216206795],
      [802.30990803401, 170.45844178379315],
      [229.38608363699458, 181.0307131702238],
    ];
    const crop = cropFor(quad);
    inCrop.forEach(([x, y], i) => {
      const back = cropToFrame(crop, x, y);
      expect(back.x).toBeCloseTo(quad[i].x, 6);
      expect(back.y).toBeCloseTo(quad[i].y, 6);
    });
  });
});

describe("keyOutline", () => {
  it("draws the keybed's edges through the key pixels and ignores a stray speck", () => {
    const classes = new Uint8Array(CROP_WIDTH * CROP_HEIGHT);
    // a keybed from x 40 to 460 whose far edge climbs from row 30 to row 20 along the keys
    for (let x = 40; x < 460; x += 1) {
      const far = Math.round(30 - ((x - 40) * 10) / 420);
      for (let y = far; y < 120; y += 1) {
        classes[y * CROP_WIDTH + x] = y < far + 40 && x % 12 < 6 ? 2 : 1;
      }
    }
    classes[5 * CROP_WIDTH + 200] = 1;
    const outline = keyOutline(classes);
    expect(outline).not.toBeNull();
    const [farLeft, farRight, nearRight, nearLeft] = outline ?? [];
    expect(farLeft.x).toBeCloseTo(39.5, 0);
    expect(farRight.x).toBeCloseTo(459.5, 0);
    expect(farLeft.y).toBeCloseTo(29.5, 0);
    expect(farRight.y).toBeCloseTo(19.5, 0);
    expect(nearLeft.y).toBeCloseTo(119.5, 0);
    expect(nearRight.y).toBeCloseTo(119.5, 0);
  });

  it("finds nothing when there are too few key pixels", () => {
    expect(keyOutline(new Uint8Array(CROP_WIDTH * CROP_HEIGHT))).toBeNull();
  });
});

describe("snapKeys", () => {
  const key = {
    black: false,
    bar: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 6 },
      { x: 0, y: 6 },
    ],
  };
  const region = (bar: { x: number; y: number }[]) => ({
    black: false,
    bar,
    centre: { x: 0.5, y: 3 },
    area: 6,
  });

  it("moves each corner onto the segmented outline and keeps the corner count", () => {
    const wobbly = region([
      { x: 0.1, y: 0.1 },
      { x: 0.5, y: 0.15 },
      { x: 1.1, y: 0 },
      { x: 1.05, y: 3 },
      { x: 1, y: 6.1 },
      { x: 0, y: 6 },
      { x: -0.05, y: 3 },
    ]);
    const [snapped] = snapKeys([key], [wobbly]);
    expect(snapped.bar).toHaveLength(4);
    expect(snapped.bar[0]).toEqual({ x: 0.1, y: 0.1 });
  });

  it("keeps a corner the outline does not reach", () => {
    const short = region([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 4 },
      { x: 0, y: 4 },
    ]);
    const [snapped] = snapKeys([key], [short]);
    expect(snapped.bar[2]).toEqual({ x: 1, y: 6 });
  });
});
