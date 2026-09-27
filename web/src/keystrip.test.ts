import { describe, expect, it } from "vitest";
import { applyHomography, findHomography, type Point } from "./homography";
import { isBlack, keyUnits } from "./keys";
import {
  boardConsensus,
  type DetectedKey,
  detectKeys,
  type FarEdge,
  measureFarEdge,
  PLAIN_OUTLINE,
  type Strip,
  trimFarEdge,
  trustedEdge,
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
  separatorsUpTo = 1,
  blackShift = 0,
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
      toX(units.from) + blackShift,
      toX(units.to) + blackShift,
      0,
      HEIGHT * 0.62,
      BLACK_LEVEL,
    );
  }
  for (let w = 0; w <= whiteKeys * separatorsUpTo; w += 1) {
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

// a flat board seen in perspective: keys at the near end come out twice as wide as the far ones
const PERSPECTIVE = (t: number): number => (1.5 * t) / (1 + 0.5 * t);

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
    const keys = Array.from({ length: found.totalKeys }, (_, i) =>
      found.keyAt(i),
    ).filter((key): key is DetectedKey => key !== null);
    const whites = keys.filter((key) => !key.black);
    expect(whites[0].semitone).toBe(0);
    expect(whites[7].semitone).toBe(12);
    expect(keys.find((key) => key.black)?.semitone).toBe(1);
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

  it("places the far keys from the rigid pattern once their white-key lines blur away", () => {
    const found = detectKeys(paintKeyboard(41, 45, PERSPECTIVE, 0.5));
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.whiteKeys).toBe(45);
    expect(found.phase).toBe("F");
    const whites = Array.from({ length: found.totalKeys }, (_, i) =>
      found.keyAt(i),
    ).filter((key): key is DetectedKey => key !== null && !key.black);
    const worst = Math.max(
      ...whites.map((key, w) =>
        Math.abs(key.bar[0].x - PERSPECTIVE(w / 45) * WIDTH),
      ),
    );
    expect(worst).toBeLessThan((WIDTH / 45) * 0.15);
  });

  it("takes the black keys' parallax out and places them back on the plane", () => {
    // one strip pixel of parallax per millimetre of height, the same everywhere on the board
    const lift = () => (x: number, _y: number, raiseMm: number) => x + raiseMm;
    const found = detectKeys(paintKeyboard(41, 45, PERSPECTIVE, 1, 8), lift);
    expect(found.kind).toBe("read");
    if (found.kind !== "read") {
      return;
    }
    expect(found.phase).toBe("F");
    const blacks = Array.from({ length: found.totalKeys }, (_, i) =>
      found.keyAt(i),
    ).filter((key): key is DetectedKey => key?.black === true);
    const u0 = keyUnits(41).from;
    let pitch = 41;
    const worst = Math.max(
      ...blacks.map((key) => {
        do {
          pitch += 1;
        } while (!isBlack(pitch));
        const truth = PERSPECTIVE((keyUnits(pitch).from - u0) / 45) * WIDTH;
        return Math.abs(key.bar[0].x - truth);
      }),
    );
    expect(worst).toBeLessThan((WIDTH / 45) * 0.1);
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

/** The same case-then-keys strip, but with the boundary running from one depth at the left edge
 * to another at the right, the way a keybed looks when its far edge was fitted at a wrong angle. */
function paintTiltedCase(leftFraction: number, rightFraction: number): Strip {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  fill(data, 30);
  for (let x = 0; x < WIDTH; x += 1) {
    const at = x / (WIDTH - 1);
    const start = Math.round(
      (leftFraction + (rightFraction - leftFraction) * at) * HEIGHT,
    );
    for (let y = start; y < HEIGHT; y += 1) {
      const index = (y * WIDTH + x) * 4;
      const level = x % 40 < 20 ? 220 : 20;
      data[index] = level;
      data[index + 1] = level;
      data[index + 2] = level;
      data[index + 3] = 255;
    }
  }
  return { width: WIDTH, height: HEIGHT, data };
}

/** A case-then-keys strip whose boundary a camera has blurred: the case fades into the keys over
 * `blurRows` rows centred on `edgeFraction`, so the true edge is the middle of the fade. */
function paintBlurredCase(edgeFraction: number, blurRows: number): Strip {
  const data = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  const centre = edgeFraction * HEIGHT;
  for (let y = 0; y < HEIGHT; y += 1) {
    const keys = Math.min(1, Math.max(0, (y - centre) / blurRows + 0.5));
    for (let x = 0; x < WIDTH; x += 1) {
      const at = (y * WIDTH + x) * 4;
      const stripe = x % 40 < 20 ? 220 : 20;
      const level = 30 * (1 - keys) + stripe * keys;
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
    const edge = measureFarEdge(paintCaseThenKeys(0.2));
    expect(edge?.left).toBeCloseTo(0.2, 2);
    expect(edge?.right).toBeCloseTo(0.2, 2);
  });

  it("recovers a shallower case too", () => {
    const edge = measureFarEdge(paintCaseThenKeys(0.08));
    expect(edge?.left).toBeCloseTo(0.08, 2);
    expect(edge?.right).toBeCloseTo(0.08, 2);
  });

  it("places a blurred edge at its middle rather than where it starts to brighten", () => {
    const edge = measureFarEdge(paintBlurredCase(0.2, 16));
    expect(edge?.left).toBeCloseTo(0.2, 2);
    expect(edge?.right).toBeCloseTo(0.2, 2);
  });

  it("recovers a boundary that sits deeper at one end than the other", () => {
    const edge = measureFarEdge(paintTiltedCase(0.06, 0.26));
    expect(edge?.left).toBeCloseTo(0.06, 1);
    expect(edge?.right).toBeCloseTo(0.26, 1);
  });

  it("finds nothing on a strip with no keys", () => {
    expect(measureFarEdge(paintAllCase())).toBeNull();
  });
});

const flat = (depth: number): FarEdge => ({ left: depth, right: depth });

describe("boardConsensus", () => {
  const read = (whiteKeys: number, phase: string, confidence: number) => ({
    kind: "read" as const,
    whiteKeys,
    phase,
    confidence,
    totalKeys: 0,
    stripKeys: whiteKeys,
    blackRaiseMm: 0,
    blackShift: { meanX: 0, meanY: 0, slope: 0 },
    outline: PLAIN_OUTLINE,
    keyAt: () => null,
  });

  it("trusts the board most looks agree on over one more confident look", () => {
    const best = boardConsensus([
      read(45, "E", 0.6),
      read(46, "E", 0.9),
      read(45, "E", 0.7),
    ]);
    expect(best?.whiteKeys).toBe(45);
    expect(best?.confidence).toBe(0.7);
  });

  it("has nothing to trust before any look reads a board", () => {
    expect(boardConsensus([])).toBeNull();
  });
});

describe("trustedEdge", () => {
  it("waits for enough looks before trusting any", () => {
    expect(trustedEdge([flat(0.1), flat(0.1)])).toBeNull();
  });

  it("takes the median once the looks agree", () => {
    const edge = trustedEdge([
      { left: 0.1, right: 0.2 },
      { left: 0.12, right: 0.21 },
      { left: 0.11, right: 0.19 },
    ]);
    expect(edge).toEqual({ left: 0.11, right: 0.2 });
  });

  it("refuses a look caught while the camera was still settling", () => {
    expect(trustedEdge([flat(0.2), flat(0.1), flat(0.1)])).toBeNull();
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
    const trimmed = trimFarEdge(quad, flat(0.25));
    expect(trimmed[2]).toEqual(quad[2]);
    expect(trimmed[3]).toEqual(quad[3]);
  });

  it("leaves the quad untouched at fraction zero", () => {
    const trimmed = trimFarEdge(quad, flat(0));
    for (const [index, corner] of quad.entries()) {
      expect(trimmed[index].x).toBeCloseTo(corner.x, 6);
      expect(trimmed[index].y).toBeCloseTo(corner.y, 6);
    }
  });

  // an oblique view, where the far end of the board is smaller on screen than the near end, so
  // the two side edges converge and depth no longer maps linearly onto the image
  const oblique = [
    { x: 0.11, y: 0.52 },
    { x: 0.87, y: 0.24 },
    { x: 0.97, y: 0.36 },
    { x: 0.14, y: 0.78 },
  ];

  it("lands the far edge at one depth on the keybed plane", () => {
    const trimmed = trimFarEdge(oblique, flat(0.25));
    const toStrip = findHomography(oblique, [
      { x: 0, y: 0 },
      { x: WIDTH, y: 0 },
      { x: WIDTH, y: HEIGHT },
      { x: 0, y: HEIGHT },
    ]);
    const left = applyHomography(toStrip, trimmed[0].x, trimmed[0].y);
    const right = applyHomography(toStrip, trimmed[1].x, trimmed[1].y);
    expect(left.y).toBeCloseTo(HEIGHT * 0.25, 3);
    expect(right.y).toBeCloseTo(HEIGHT * 0.25, 3);
  });

  it("does not tilt the far edge the way an image-space lerp does", () => {
    const trimmed = trimFarEdge(oblique, flat(0.25));
    const lerp = (from: Point, to: Point): number =>
      from.y + (to.y - from.y) * 0.25;
    const nearEnd = Math.abs(trimmed[0].y - lerp(oblique[0], oblique[3]));
    const farEnd = Math.abs(trimmed[1].y - lerp(oblique[1], oblique[2]));
    expect(nearEnd).toBeGreaterThan(farEnd * 2);
  });
});
