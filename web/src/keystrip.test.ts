import { describe, expect, it } from "vitest";
import { isBlack, keyUnits } from "./keys";
import {
  detectKeys,
  measureFarEdge,
  type Strip,
  trimFarEdge,
} from "./keystrip";

const WIDTH = 1200;
const HEIGHT = 150;
const WHITE_LEVEL = 226;
const BLACK_LEVEL = 24;
const SEPARATOR_LEVEL = 30;

function fill(data: Uint8ClampedArray, level: number): void {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = level;
    data[i + 1] = level;
    data[i + 2] = level;
    data[i + 3] = 255;
  }
}

function paintRect(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  level: number,
): void {
  const fromX = Math.max(0, Math.min(width, Math.round(x0)));
  const toX = Math.max(0, Math.min(width, Math.round(x1)));
  const fromY = Math.max(0, Math.min(height, Math.round(y0)));
  const toY = Math.max(0, Math.min(height, Math.round(y1)));
  for (let y = fromY; y < toY; y += 1) {
    for (let x = fromX; x < toX; x += 1) {
      const at = (y * width + x) * 4;
      data[at] = level;
      data[at + 1] = level;
      data[at + 2] = level;
      data[at + 3] = 255;
    }
  }
}

/** A rectified strip of a real keyboard shape: black keys near the far edge and a separator
 * line at every white-key boundary near the player's edge. `warp` remaps unit position to strip
 * fraction before scaling to pixels, so a test can paint the same keyboard with a leftover lens
 * warp instead of the even spacing a perfect rectify would give. */
function paintKeyboard(
  startPitch: number,
  whiteKeys: number,
  warp: (t: number) => number = (t) => t,
): Strip {
  let pitch = startPitch;
  let count = 1;
  while (count < whiteKeys) {
    pitch += 1;
    if (!isBlack(pitch)) {
      count += 1;
    }
  }
  const endPitch = pitch;
  const u0 = keyUnits(startPitch).from;
  const totalUnits = keyUnits(endPitch).to - u0;
  const toX = (u: number): number => warp((u - u0) / totalUnits) * WIDTH;

  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  fill(data, WHITE_LEVEL);
  for (let p = startPitch; p <= endPitch; p += 1) {
    if (!isBlack(p)) {
      continue;
    }
    const units = keyUnits(p);
    paintRect(
      data,
      WIDTH,
      HEIGHT,
      toX(units.from),
      toX(units.to),
      0,
      HEIGHT * 0.62,
      BLACK_LEVEL,
    );
  }
  for (let w = 0; w <= whiteKeys; w += 1) {
    const x = toX(u0 + w);
    paintRect(
      data,
      WIDTH,
      HEIGHT,
      x - 2,
      x + 2,
      HEIGHT * 0.8,
      HEIGHT * 0.96,
      SEPARATOR_LEVEL,
    );
  }
  return { width: WIDTH, height: HEIGHT, data };
}

// mulberry32, so the noise strip is the same run to run rather than flaking on a lucky draw
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function paintNoise(seed: number): Strip {
  const random = seeded(seed);
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  for (let i = 0; i < data.length; i += 4) {
    const level = Math.floor(random() * 256);
    data[i] = level;
    data[i + 1] = level;
    data[i + 2] = level;
    data[i + 3] = 255;
  }
  return { width: WIDTH, height: HEIGHT, data };
}

// a smooth remap fixed at both ends but bent through the middle, so the derivative (the local
// key width) at one edge is exactly twice what it is at the other: the ratio the owner measured
const STRONG_WARP = (t: number): number => t + (t * (1 - t)) / 3;

describe("detectKeys", () => {
  it("recovers a 36 white key board starting on C", () => {
    const found = detectKeys(paintKeyboard(36, 36));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.whiteKeys).toBe(36);
    expect(found.phase).toBe("C");
    expect(found.confidence).toBeGreaterThan(0.5);
    expect(found.totalKeys).toBeGreaterThan(found.whiteKeys);
  });

  it("recovers a 45 white key board starting on F", () => {
    const found = detectKeys(paintKeyboard(41, 45));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.whiteKeys).toBe(45);
    expect(found.phase).toBe("F");
    expect(found.confidence).toBeGreaterThan(0.5);
  });

  it("recovers the count and phase through a warp that doubles the local key width end to end", () => {
    const found = detectKeys(paintKeyboard(36, 36, STRONG_WARP));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.whiteKeys).toBe(36);
    expect(found.phase).toBe("C");
    expect(found.confidence).toBeGreaterThan(0.3);
  });

  it("recovers a warped 45 white key board too", () => {
    const found = detectKeys(paintKeyboard(41, 45, STRONG_WARP));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.whiteKeys).toBe(45);
    expect(found.phase).toBe("F");
  });

  it("returns low confidence rather than a count for noise", () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      const found = detectKeys(paintNoise(seed));
      expect(found.confidence).toBeLessThan(0.3);
    }
  });
});

/** A strip with a flat, dark case for the first `caseFraction` of its depth, then a keyboard-like
 * pattern (alternating bright and dark vertical stripes, high contrast at every remaining row)
 * for the rest, so `measureFarEdge` has a known depth to recover. */
function paintCaseThenKeys(caseFraction: number): Strip {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  fill(data, 30);
  const caseRows = Math.round(caseFraction * HEIGHT);
  for (let y = caseRows; y < HEIGHT; y += 1) {
    for (let x = 0; x < WIDTH; x += 1) {
      const at = (y * WIDTH + x) * 4;
      const level = x % 40 < 20 ? 220 : 20;
      data[at] = level;
      data[at + 1] = level;
      data[at + 2] = level;
      data[at + 3] = 255;
    }
  }
  return { width: WIDTH, height: HEIGHT, data };
}

function paintAllCase(): Strip {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  fill(data, 30);
  return { width: WIDTH, height: HEIGHT, data };
}

describe("measureFarEdge", () => {
  it("recovers where the keys start inside the strip", () => {
    const fraction = measureFarEdge(paintCaseThenKeys(0.2));
    expect(fraction).toBeCloseTo(0.2, 2);
  });

  it("recovers a shallower case too", () => {
    const fraction = measureFarEdge(paintCaseThenKeys(0.08));
    expect(fraction).toBeCloseTo(0.08, 2);
  });

  it("finds nothing on a strip with no keys", () => {
    expect(measureFarEdge(paintAllCase())).toBeNull();
  });
});

describe("trimFarEdge", () => {
  const quad = [
    { x: 0.1, y: 0.2 },
    { x: 0.9, y: 0.22 },
    { x: 0.85, y: 0.6 },
    { x: 0.15, y: 0.58 },
  ];

  it("never moves the near edge", () => {
    const trimmed = trimFarEdge(quad, 0.25);
    expect(trimmed[2]).toEqual(quad[2]);
    expect(trimmed[3]).toEqual(quad[3]);
  });

  it("leaves the quad untouched at fraction zero", () => {
    expect(trimFarEdge(quad, 0)).toEqual(quad);
  });

  it("moves the far edge toward the near edge by the given fraction", () => {
    const trimmed = trimFarEdge(quad, 0.25);
    expect(trimmed[0].x).toBeCloseTo(
      quad[0].x + (quad[3].x - quad[0].x) * 0.25,
    );
    expect(trimmed[0].y).toBeCloseTo(
      quad[0].y + (quad[3].y - quad[0].y) * 0.25,
    );
    expect(trimmed[1].x).toBeCloseTo(
      quad[1].x + (quad[2].x - quad[1].x) * 0.25,
    );
    expect(trimmed[1].y).toBeCloseTo(
      quad[1].y + (quad[2].y - quad[1].y) * 0.25,
    );
  });
});
