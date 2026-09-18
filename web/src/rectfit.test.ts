import { describe, expect, test } from "vitest";
import type { Point } from "./homography";
import { DEPTH_UNITS, WHITE_KEY_COUNT } from "./pose";
import { boundaryPoints, endMovesOnlyInward, fitRectangle } from "./rectfit";

const WIDTH = 640;
const HEIGHT = 480;
const MASK = 144;
const CELL_PX = WIDTH / (MASK - 1);

function projected(
  focal: number,
  yaw: number,
  pitch: number,
  distance: number,
): Point[] {
  const world = [
    [0, 0, 0],
    [WHITE_KEY_COUNT, 0, 0],
    [WHITE_KEY_COUNT, DEPTH_UNITS, 0],
    [0, DEPTH_UNITS, 0],
  ].map(([x, y, z]) => [x - WHITE_KEY_COUNT / 2, y - DEPTH_UNITS / 2, z]);
  const cy = Math.cos(yaw),
    sy = Math.sin(yaw),
    cp = Math.cos(pitch),
    sp = Math.sin(pitch);
  const rotation = [
    [cy, sy * sp, sy * cp],
    [0, cp, -sp],
    [-sy, cy * sp, cy * cp],
  ];
  return world.map(([x, y, z]) => {
    const camera = rotation.map((row) => row[0] * x + row[1] * y + row[2] * z);
    camera[2] += distance;
    return {
      x: WIDTH / 2 + (focal * camera[0]) / camera[2],
      y: HEIGHT / 2 + (focal * camera[1]) / camera[2],
    };
  });
}

function inside(quad: Point[], x: number, y: number): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = quad[i],
      b = quad[(i + 1) % 4];
    const next = Math.sign((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x));
    if (next !== 0 && sign !== 0 && next !== sign) return false;
    if (next !== 0) sign = next;
  }
  return true;
}

function coverage(quad: Point[]): Float32Array {
  const map = new Float32Array(MASK * MASK),
    fine = 8;
  for (let row = 0; row < MASK; row += 1)
    for (let col = 0; col < MASK; col += 1) {
      let hits = 0;
      for (let i = 0; i < fine; i += 1)
        for (let j = 0; j < fine; j += 1) {
          const x = ((col + (j + 0.5) / fine - 0.5) / (MASK - 1)) * WIDTH;
          const y = ((row + (i + 0.5) / fine - 0.5) / (MASK - 1)) * HEIGHT;
          if (inside(quad, x, y)) hits += 1;
        }
      map[row * MASK + col] = hits / (fine * fine);
    }
  return map;
}

function shortFarEnd(quad: Point[], fraction: number): Point[] {
  const result = quad.map((point) => ({ ...point }));
  const farFirst =
    Math.hypot(result[2].x - result[1].x, result[2].y - result[1].y) <
    Math.hypot(result[3].x - result[0].x, result[3].y - result[0].y);
  const [i, j, k, m] = farFirst ? [1, 2, 0, 3] : [0, 3, 1, 2];
  for (const [target, source] of [
    [i, k],
    [j, m],
  ])
    result[target] = {
      x: result[target].x + (result[source].x - result[target].x) * fraction,
      y: result[target].y + (result[source].y - result[target].y) * fraction,
    };
  return result;
}

function worstCorner(quad: Point[], truth: Point[]): number {
  let best = Infinity;
  for (let roll = 0; roll < 4; roll += 1)
    for (const reverse of [false, true]) {
      const error = Math.max(
        ...truth.map((point, index) => {
          const candidate = quad[(roll + (reverse ? 4 - index : index)) % 4];
          return Math.hypot(candidate.x - point.x, candidate.y - point.y);
        }),
      );
      best = Math.min(best, error);
    }
  return best;
}

describe("fitRectangle", () => {
  test("a far end read short is placed by the outline and shape", () => {
    const truth = projected(750, 0.9, -0.8, 100),
      coarse = shortFarEnd(truth, 0.1);
    const fit = fitRectangle(
      boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, coarse),
      coarse,
      WIDTH,
      HEIGHT,
    );
    expect(worstCorner(coarse, truth)).toBeGreaterThan(15);
    expect(fit).not.toBeNull();
    expect(worstCorner(fit?.quad ?? coarse, truth)).toBeLessThan(1.5 * CELL_PX);
  });

  test("an end without evidence is placed from the held focal", () => {
    const truth = projected(750, 0.9, -0.8, 100),
      coarse = shortFarEnd(truth, 0.1);
    const points = boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, coarse);
    const farFirst =
      Math.hypot(truth[2].x - truth[1].x, truth[2].y - truth[1].y) <
      Math.hypot(truth[3].x - truth[0].x, truth[3].y - truth[0].y);
    const far = farFirst
      ? { x: (truth[1].x + truth[2].x) / 2, y: (truth[1].y + truth[2].y) / 2 }
      : { x: (truth[0].x + truth[3].x) / 2, y: (truth[0].y + truth[3].y) / 2 };
    const fit = fitRectangle(
      points.filter(
        (point) =>
          Math.hypot(point.x - far.x, point.y - far.y) >
          0.2 * Math.hypot(truth[1].x - truth[0].x, truth[1].y - truth[0].y),
      ),
      coarse,
      WIDTH,
      HEIGHT,
      750,
    );
    expect(fit).not.toBeNull();
    const seen = farFirst ? [truth[0], truth[3]] : [truth[1], truth[2]];
    for (const corner of seen)
      expect(
        Math.min(
          ...(fit?.quad ?? coarse).map((point) =>
            Math.hypot(point.x - corner.x, point.y - corner.y),
          ),
        ),
      ).toBeLessThan(1.5 * CELL_PX);
  });

  test("a correct quad stays put", () => {
    const truth = projected(900, -0.4, -0.7, 90);
    const fit = fitRectangle(
      boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, truth),
      truth,
      WIDTH,
      HEIGHT,
    );
    expect(fit).not.toBeNull();
    expect(worstCorner(fit?.quad ?? truth, truth)).toBeLessThan(1.5 * CELL_PX);
  });
});

describe("end refinement safety", () => {
  const rectangle = [
    { x: 0, y: 0 },
    { x: 10, y: 0 },
    { x: 10, y: 4 },
    { x: 0, y: 4 },
  ];
  test("accepts an end pulled toward the keybed", () =>
    expect(
      endMovesOnlyInward(rectangle, 1, [
        { corner: 1, point: { x: 8, y: 0 } },
        { corner: 2, point: { x: 8, y: 4 } },
      ]),
    ).toBe(true));
  test("rejects an end grown onto the casing", () =>
    expect(
      endMovesOnlyInward(rectangle, 1, [
        { corner: 1, point: { x: 12, y: 0 } },
        { corner: 2, point: { x: 12, y: 4 } },
      ]),
    ).toBe(false));
});
