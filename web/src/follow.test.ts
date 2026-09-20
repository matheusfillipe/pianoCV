import { describe, expect, test } from "vitest";
import { estimateMotion, type GrayFrame } from "./follow";
import type { Point } from "./homography";

const WIDTH = 120;
const HEIGHT = 90;

// a smooth, non-repeating texture: enough local gradient for patch matching to be
// unambiguous, unlike a checkerboard whose tiles alias into each other under translation
function pattern(x: number, y: number): number {
  const raw =
    Math.sin(x * 0.31) * Math.cos(y * 0.27) +
    Math.sin((x + y) * 0.13) * 0.6 +
    Math.sin(x * 0.07 - y * 0.19) * 0.4;
  return raw / 4 + 0.5;
}

function frameFrom(
  width: number,
  height: number,
  at: (x: number, y: number) => Point,
): GrayFrame {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = at(x, y);
      data[y * width + x] = pattern(p.x, p.y);
    }
  }
  return { data, width, height };
}

// decorrelated from `pattern` at every pixel, standing in for a frame with no
// keybed content left to match at all (interference, a hand filling the view)
function noiseFrame(width: number, height: number): GrayFrame {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const n = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
      data[y * width + x] = n - Math.floor(n);
    }
  }
  return { data, width, height };
}

function rotate(p: Point, radians: number): Point {
  const c = Math.cos(radians);
  const s = Math.sin(radians);
  return { x: c * p.x - s * p.y, y: s * p.x + c * p.y };
}

function grid(cols: number, rows: number, margin: number): Point[] {
  const points: Point[] = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      points.push({
        x: margin + (col / (cols - 1)) * (WIDTH - 2 * margin),
        y: margin + (row / (rows - 1)) * (HEIGHT - 2 * margin),
      });
    }
  }
  return points;
}

describe("estimateMotion", () => {
  const source = frameFrom(WIDTH, HEIGHT, (x, y) => ({ x, y }));

  test("recovers a known pixel shift", () => {
    const dx = 4;
    const dy = -3;
    const target = frameFrom(WIDTH, HEIGHT, (x, y) => ({
      x: x - dx,
      y: y - dy,
    }));
    const estimate = estimateMotion(source, grid(6, 5, 20), target);
    expect(estimate).not.toBeNull();
    expect(estimate?.transform.tx).toBeCloseTo(dx, 0);
    expect(estimate?.transform.ty).toBeCloseTo(dy, 0);
    expect(estimate?.transform.scale).toBeCloseTo(1, 1);
    expect(estimate?.transform.rotation).toBeCloseTo(0, 1);
  });

  test("recovers a known rotation and scale", () => {
    // pivoting near the sampled patch rather than the frame origin keeps each point's
    // displacement inside the tracker's own search radius, the way a real small pan or
    // tilt does; a rotation about a far-off origin would fling distant points out of range
    const center = { x: WIDTH / 2, y: HEIGHT / 2 };
    const scale = 1.05;
    const angle = 0.08;
    const translation = { x: 1.5, y: -1 };
    const target = frameFrom(WIDTH, HEIGHT, (x, y) => {
      const q = {
        x: x - center.x - translation.x,
        y: y - center.y - translation.y,
      };
      const p = rotate({ x: q.x / scale, y: q.y / scale }, -angle);
      return { x: p.x + center.x, y: p.y + center.y };
    });
    const estimate = estimateMotion(source, grid(6, 5, 35), target);
    expect(estimate).not.toBeNull();
    expect(estimate?.transform.scale).toBeCloseTo(scale, 1);
    expect(estimate?.transform.rotation).toBeCloseTo(angle, 1);
  });

  test("rejects a frame of pure noise", () => {
    const target = noiseFrame(WIDTH, HEIGHT);
    expect(estimateMotion(source, grid(6, 5, 20), target)).toBeNull();
  });

  test("rejects a frame where the keybed has left the picture", () => {
    const narrowed = frameFrom(60, HEIGHT, (x, y) => ({ x, y }));
    const points = grid(6, 5, 5).filter((p) => p.x > 90);
    expect(points.length).toBeGreaterThan(0);
    expect(estimateMotion(source, points, narrowed)).toBeNull();
  });
});
