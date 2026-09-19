import { PerspectiveCamera, Vector3 } from "three";
import { describe, expect, it } from "vitest";
import { fitBoard } from "./boardfit";
import { calibrateBoard } from "./boardgeometry";
import { applyHomography, findHomography } from "./homography";
import { isBlack, keyUnits } from "./keys";

const width = 960;
const height = 640;
const unit = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

function scene(fov: number, azimuth: number, lowest: number, highest: number) {
  const origin = keyUnits(lowest).from;
  const span = keyUnits(highest).to - origin;
  const camera = new PerspectiveCamera(fov, width / height, 0.1, 500);
  const distance = span / (1.1 * Math.tan((fov * Math.PI) / 360));
  const angle = (azimuth * Math.PI) / 180;
  camera.position.set(
    distance * Math.sin(angle) * 0.7,
    distance * 0.7,
    distance * Math.cos(angle) * 0.7,
  );
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();
  const quad = unit.map(({ x, y }) => {
    const p = new Vector3((x - 0.5) * span, 0, (y - 0.5) * 5).project(camera);
    return { x: ((p.x + 1) * width) / 2, y: ((1 - p.y) * height) / 2 };
  });
  const inverse = findHomography(quad, unit);
  const blacks = Array.from(
    { length: highest - lowest + 1 },
    (_, i) => lowest + i,
  )
    .filter(isBlack)
    .map(keyUnits);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = applyHomography(inverse, x, y);
      const inside = p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
      const position = origin + p.x * span;
      const black =
        p.y < 0.62 &&
        blacks.some((k) => position >= k.from && position <= k.to);
      const value = inside ? (black ? 25 : 220) : 50;
      const at = (y * width + x) * 4;
      data[at] = value;
      data[at + 1] = value;
      data[at + 2] = value;
      data[at + 3] = 255;
    }
  }
  const pixels: ImageData = { width, height, data, colorSpace: "srgb" };
  return { pixels, quad, span, origin };
}

describe("synthetic calibration across camera intrinsics", () => {
  it("supports the generator's 96-key F0 through E8 range", () => {
    const rendered = scene(45, 0, 17, 112);
    const board = fitBoard(rendered.pixels, rendered.quad);
    expect(board?.lowest).toBe(17);
    expect(board?.highest).toBe(112);
    expect(board?.span).toBeCloseTo(56, 0);
    expect(board?.blackKeys?.length).toBeGreaterThan(0);
  });
  it("corrects a rear edge that expands into the body at one end", () => {
    const rendered = scene(45, 60, 36, 96);
    const h = findHomography(unit, rendered.quad);
    const proposal = [
      applyHomography(h, 0, -0.18),
      applyHomography(h, 1, -0.02),
      rendered.quad[2],
      rendered.quad[3],
    ];
    const fit = calibrateBoard(rendered.pixels, proposal);
    expect(fit).not.toBeNull();
    for (let i = 0; i < 4; i += 1) {
      const actual = fit?.quad[i];
      if (!actual) throw new Error("missing calibrated corner");
      expect(
        Math.hypot(
          actual.x - rendered.quad[i].x,
          actual.y - rendered.quad[i].y,
        ),
      ).toBeLessThan(3);
    }
  });
  it("never expands the front edge into a bright control-panel band", () => {
    const rendered = scene(45, 60, 36, 96);
    const imageToUnit = findHomography(rendered.quad, unit);
    const data = new Uint8ClampedArray(rendered.pixels.data);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const p = applyHomography(imageToUnit, x, y);
        if (p.y > 1 && p.y < 1.16) {
          const at = (y * width + x) * 4;
          data[at] = 220;
          data[at + 1] = 220;
          data[at + 2] = 220;
        }
      }
    }
    const fit = calibrateBoard(
      { ...rendered.pixels, data },
      rendered.quad,
    );
    expect(fit).not.toBeNull();
    if (!fit) throw new Error("calibration unexpectedly failed");
    expect(
      fit.quad.slice(2).every((point) => {
        return applyHomography(imageToUnit, point.x, point.y).y <= 1;
      }),
    ).toBe(true);
  });
  it("does not pull a correct rear edge onto an internal brightness seam", () => {
    const rendered = scene(45, 60, 36, 96);
    const imageToUnit = findHomography(rendered.quad, unit);
    const data = new Uint8ClampedArray(rendered.pixels.data);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const p = applyHomography(imageToUnit, x, y);
        if (p.y < 0) {
          const at = (y * width + x) * 4;
          data[at] = 200;
          data[at + 1] = 200;
          data[at + 2] = 200;
        }
        if (p.y > 0.16 && p.y < 0.2) {
          const at = (y * width + x) * 4;
          data[at] = 25;
          data[at + 1] = 25;
          data[at + 2] = 25;
        }
      }
    }
    const fit = calibrateBoard({ ...rendered.pixels, data }, rendered.quad);
    expect(fit).not.toBeNull();
    if (!fit) throw new Error("calibration unexpectedly failed");
    expect(
      Math.max(
        ...fit.quad.slice(0, 2).map((point, index) =>
          Math.hypot(
            point.x - rendered.quad[index].x,
            point.y - rendered.quad[index].y,
          ),
        ),
      ),
    ).toBeLessThan(3);
  });
  it("declines a cropped keybed instead of stretching a partial stripe", () => {
    const rendered = scene(45, 0, 36, 96);
    const cropped = rendered.quad.map((point, index) =>
      index === 1 ? { ...point, x: -1 } : point,
    );
    expect(calibrateBoard(rendered.pixels, cropped)).toBeNull();
  });
  for (const [lowest, highest] of [
    [21, 108], // 88 keys, A0 through C8
    [28, 103], // 76 keys, E1 through G7
    [36, 84], // 49 keys, C2 through C6
    [48, 72], // 25 keys, C3 through C5
  ] as const) {
    it(`keeps the pitch range for ${highest - lowest + 1} keys`, () => {
      const rendered = scene(45, 0, lowest, highest);
      const board = fitBoard(rendered.pixels, rendered.quad);
      expect(board).not.toBeNull();
      expect(board?.lowest).toBe(lowest);
      expect(board?.highest).toBe(highest);
    });
  }
  for (const fov of [25, 45, 70]) {
    for (const azimuth of [-60, 0, 60]) {
      it(`refines a skewed rear edge at ${fov} degree FOV and ${azimuth} degree azimuth`, () => {
        const rendered = scene(fov, azimuth, 36, 96);
        const h = findHomography(unit, rendered.quad);
        const proposal = [
          applyHomography(h, 0, -0.32),
          applyHomography(h, 1, -0.08),
          rendered.quad[2],
          rendered.quad[3],
        ];
        const fit = calibrateBoard(rendered.pixels, proposal);
        expect(fit).not.toBeNull();
        const worst = Math.max(
          ...rendered.quad.map((point, index) => {
            const actual = fit?.quad[index];
            return actual
              ? Math.hypot(actual.x - point.x, actual.y - point.y)
              : Infinity;
          }),
        );
        expect(worst).toBeLessThan(4);
      });
    }
  }
  for (const fov of [25, 45, 70]) {
    for (const azimuth of [-60, 0, 60]) {
      it(`reads a 61-key board at ${fov} degree FOV and ${azimuth} degree azimuth`, () => {
        const rendered = scene(fov, azimuth, 36, 96);
        const board = fitBoard(rendered.pixels, rendered.quad);
        expect(board).not.toBeNull();
        expect(board?.span).toBeCloseTo(rendered.span, 0);
        expect(board?.lowest).toBe(36);
        expect(board?.highest).toBe(96);
      });
    }
  }
});
