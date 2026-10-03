import { describe, expect, it } from "vitest";
import {
  CROP_HEIGHT,
  CROP_WIDTH,
  createKeySnap,
  cropFor,
  cropToFrame,
  keyOutline,
  outlinePoints,
  sampleOutline,
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

  it("defaults to keyseg's own crop size", () => {
    const quad = [
      { x: 120, y: 300 },
      { x: 520, y: 180 },
      { x: 560, y: 260 },
      { x: 150, y: 400 },
    ];
    expect(cropFor(quad)).toEqual(cropFor(quad, CROP_WIDTH, CROP_HEIGHT));
  });

  it("maps frame points back correctly at a crop size other than keyseg's own", () => {
    const quad = [
      { x: 120, y: 300 },
      { x: 520, y: 180 },
      { x: 560, y: 260 },
      { x: 150, y: 400 },
    ];
    const width = 768;
    const height = 160;
    const crop = cropFor(quad, width, height);
    for (const [cx, cy] of [
      [0, 0],
      [width, 0],
      [width, height],
      [0, height],
      [width / 2, height / 2],
    ]) {
      const frame = cropToFrame(crop, cx, cy, width, height);
      // inverts the crop's own similarity transform independently of cropFor/cropToFrame's own
      // maths, so this checks the two agree at a size keyseg itself never uses
      const dx = frame.x - crop.centre.x;
      const dy = frame.y - crop.centre.y;
      const u = dx * crop.along.x + dy * crop.along.y;
      const v = dx * crop.across.x + dy * crop.across.y;
      expect(u / crop.scale + width / 2).toBeCloseTo(cx, 6);
      expect(v / crop.scale + height / 2).toBeCloseTo(cy, 6);
    }
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
    bar: outlinePoints(
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 6 },
        { x: 0, y: 6 },
      ],
      40,
    ),
  };
  const region = (bar: { x: number; y: number }[]) => ({
    black: false,
    bar,
    centre: { x: 0.5, y: 4 },
    area: 5,
  });

  it("bends a white key's side into the notch a black key cuts", () => {
    const notched = region([
      { x: 0, y: 0 },
      { x: 0.7, y: 0 },
      { x: 0.7, y: 3 },
      { x: 1, y: 3 },
      { x: 1, y: 6 },
      { x: 0, y: 6 },
    ]);
    const [snapped] = snapKeys([key], [notched]);
    const back = snapped.bar.filter(
      (_, i) => key.bar[i].x === 1 && key.bar[i].y > 0.8 && key.bar[i].y < 2.2,
    );
    const front = snapped.bar.filter(
      (_, i) => key.bar[i].x === 1 && key.bar[i].y > 4 && key.bar[i].y < 5.2,
    );
    expect(back.length).toBeGreaterThan(0);
    for (const p of back) {
      expect(p.x).toBeLessThan(0.85);
    }
    for (const p of front) {
      expect(p.x).toBeGreaterThan(0.95);
    }
  });

  it("keeps the template where the segmented outline is out of reach", () => {
    const short = region([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 4 },
      { x: 0, y: 4 },
    ]);
    const [snapped] = snapKeys([key], [short]);
    const last = key.bar.findIndex((p) => p.y === 6);
    expect(snapped.bar[last]).toEqual(key.bar[last]);
  });
});

describe("createKeySnap", () => {
  const black = {
    black: true,
    semitone: 1,
    bar: [
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 4 },
      { x: 0, y: 4 },
    ],
  };
  const beside = {
    black: true,
    bar: [
      { x: 0.3, y: 0 },
      { x: 1.3, y: 0 },
      { x: 1.3, y: 4 },
      { x: 0.3, y: 4 },
    ],
    centre: { x: 0.8, y: 2 },
    area: 4,
  };

  it("keeps its offset when the next segmentation misses it", () => {
    const snap = createKeySnap();
    const [first] = snap.apply([black], [beside], 1);
    const [missed] = snap.apply([black], [], 1);
    expect(missed.bar).toEqual(first.bar);
  });
});

describe("snapping a black key", () => {
  it("keeps its edges straight past a shadow the segmenter took for key", () => {
    const { points, edges } = sampleOutline(
      [
        { x: 0, y: 0 },
        { x: 4, y: 0 },
        { x: 4, y: 1 },
        { x: 0, y: 1 },
      ],
      40,
    );
    const shadowed = {
      black: true,
      bar: [
        { x: 0, y: 0 },
        { x: 4, y: 0 },
        { x: 4, y: 1 },
        { x: 2.4, y: 1 },
        { x: 2, y: 1.3 },
        { x: 1.6, y: 1 },
        { x: 0, y: 1 },
      ],
      centre: { x: 2, y: 0.5 },
      area: 4.1,
    };
    const [snapped] = snapKeys(
      [{ black: true, bar: points, edges }],
      [shadowed],
    );
    const bottom = snapped.bar.filter((_, i) => edges[i] === 2);
    for (const p of bottom) {
      expect(p.y).toBeCloseTo(1, 1);
    }
  });
});
