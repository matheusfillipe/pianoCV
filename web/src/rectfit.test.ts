import { describe, expect, test } from "vitest";
import type { Point } from "./homography";
import { DEPTH_UNITS, WHITE_KEY_COUNT } from "./pose";
import { boundaryPoints, fitRectangle } from "./rectfit";

const WIDTH = 640;
const HEIGHT = 480;
const MASK = 144;
// the mask is 144 cells across the frame, so the boundary is known to about one cell
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
  const cy = Math.cos(yaw);
  const sy = Math.sin(yaw);
  const cp = Math.cos(pitch);
  const sp = Math.sin(pitch);
  const r = [
    [cy, sy * sp, sy * cp],
    [0, cp, -sp],
    [-sy, cy * sp, cy * cp],
  ];
  return world.map(([x, y, z]) => {
    const c = r.map((row) => row[0] * x + row[1] * y + row[2] * z);
    c[2] += distance;
    return {
      x: WIDTH / 2 + (focal * c[0]) / c[2],
      y: HEIGHT / 2 + (focal * c[1]) / c[2],
    };
  });
}

function inside(quad: Point[], x: number, y: number): boolean {
  let sign = 0;
  for (let i = 0; i < 4; i += 1) {
    const a = quad[i];
    const b = quad[(i + 1) % 4];
    const s = Math.sign((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x));
    if (s === 0) {
      continue;
    }
    if (sign === 0) {
      sign = s;
    } else if (s !== sign) {
      return false;
    }
  }
  return true;
}

// a probability map the way the net produces one: the coverage of each cell, so the
// half-probability line runs through the true edge, never along cell boundaries
function coverage(quad: Point[]): Float32Array {
  const fine = 8;
  const map = new Float32Array(MASK * MASK);
  for (let row = 0; row < MASK; row += 1) {
    for (let col = 0; col < MASK; col += 1) {
      let hits = 0;
      for (let i = 0; i < fine; i += 1) {
        for (let j = 0; j < fine; j += 1) {
          const x = ((col + (j + 0.5) / fine - 0.5) / (MASK - 1)) * WIDTH;
          const y = ((row + (i + 0.5) / fine - 0.5) / (MASK - 1)) * HEIGHT;
          if (inside(quad, x, y)) {
            hits += 1;
          }
        }
      }
      map[row * MASK + col] = hits / (fine * fine);
    }
  }
  return map;
}

function shortFarEnd(quad: Point[], fraction: number): Point[] {
  const q = quad.map((p) => ({ ...p }));
  const farFirst =
    Math.hypot(q[2].x - q[1].x, q[2].y - q[1].y) <
    Math.hypot(q[3].x - q[0].x, q[3].y - q[0].y);
  const [i, j, k, m] = farFirst ? [1, 2, 0, 3] : [0, 3, 1, 2];
  q[i] = {
    x: q[i].x + (q[k].x - q[i].x) * fraction,
    y: q[i].y + (q[k].y - q[i].y) * fraction,
  };
  q[j] = {
    x: q[j].x + (q[m].x - q[j].x) * fraction,
    y: q[j].y + (q[m].y - q[j].y) * fraction,
  };
  return q;
}

function worstCorner(quad: Point[], truth: Point[]): number {
  let best = Infinity;
  for (let roll = 0; roll < 4; roll += 1) {
    for (const reverse of [false, true]) {
      const ordered = truth.map(
        (_, i) => quad[(roll + (reverse ? 4 - i : i)) % 4],
      );
      const worst = Math.max(
        ...ordered.map((p, i) =>
          Math.hypot(p.x - truth[i].x, p.y - truth[i].y),
        ),
      );
      best = Math.min(best, worst);
    }
  }
  return best;
}

describe("fitRectangle", () => {
  test("a far end read short is placed by the outline and the shape", () => {
    const truth = projected(750, 0.9, -0.8, 100);
    const coarse = shortFarEnd(truth, 0.1);
    const points = boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, coarse);
    const before = worstCorner(coarse, truth);
    const fit = fitRectangle(points, coarse, WIDTH, HEIGHT);
    expect(before).toBeGreaterThan(15);
    expect(fit).not.toBeNull();
    expect(worstCorner(fit?.quad ?? coarse, truth)).toBeLessThan(1.5 * CELL_PX);
  });

  test("an end with no evidence is placed by the length once the focal is held", () => {
    const truth = projected(750, 0.9, -0.8, 100);
    const coarse = shortFarEnd(truth, 0.1);
    const points = boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, coarse);
    const farFirst =
      Math.hypot(truth[2].x - truth[1].x, truth[2].y - truth[1].y) <
      Math.hypot(truth[3].x - truth[0].x, truth[3].y - truth[0].y);
    const far = farFirst
      ? { x: (truth[1].x + truth[2].x) / 2, y: (truth[1].y + truth[2].y) / 2 }
      : { x: (truth[0].x + truth[3].x) / 2, y: (truth[0].y + truth[3].y) / 2 };
    const span = Math.hypot(truth[1].x - truth[0].x, truth[1].y - truth[0].y);
    const kept = points.filter(
      (p) => Math.hypot(p.x - far.x, p.y - far.y) > 0.2 * span,
    );
    const held = fitRectangle(kept, coarse, WIDTH, HEIGHT, 750);
    expect(held).not.toBeNull();
    // the seen end is exact; the unseen end lies on the long edges, but where along them is
    // only pinned by perspective, so its position is not promised
    const fitted = held?.quad ?? coarse;
    const seenEnd = farFirst ? [truth[0], truth[3]] : [truth[1], truth[2]];
    for (const corner of seenEnd) {
      const nearest = Math.min(
        ...fitted.map((p) => Math.hypot(p.x - corner.x, p.y - corner.y)),
      );
      expect(nearest).toBeLessThan(1.5 * CELL_PX);
    }
    const lines: [Point, Point][] = [
      [truth[0], truth[1]],
      [truth[3], truth[2]],
    ];
    for (const p of fitted) {
      const offLine = Math.min(
        ...lines.map(([a, b]) => {
          const len = Math.hypot(b.x - a.x, b.y - a.y);
          return Math.abs(
            ((b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)) / len,
          );
        }),
      );
      expect(offLine).toBeLessThan(1.5 * CELL_PX);
    }
  });

  test("a correct quad stays put", () => {
    const truth = projected(900, -0.4, -0.7, 90);
    const points = boundaryPoints(coverage(truth), MASK, WIDTH, HEIGHT, truth);
    const fit = fitRectangle(points, truth, WIDTH, HEIGHT);
    expect(fit).not.toBeNull();
    expect(worstCorner(fit?.quad ?? truth, truth)).toBeLessThan(1.5 * CELL_PX);
  });
});
