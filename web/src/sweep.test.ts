import { describe, expect, it } from "vitest";
import {
  BOARD_SIZES,
  between,
  CASE_BACK_DEPTH_MM,
  CASE_CHEEK_WIDTH_MM,
  noteName,
  pickBoardSize,
  pickCaseColorFamily,
  type Range,
  SWEEP_AZIMUTH_DEG,
  SWEEP_DISTANCE_FACTOR,
  SWEEP_ELEVATION_DEG,
  SWEEP_FOV_DEG,
  SWEEP_OFFSET_X_FACTOR,
  SWEEP_OFFSET_Y_FACTOR,
  SWEEP_ROLL_DEG,
  samplePose,
} from "./sweep";

const RANGES: Range[] = [
  SWEEP_ELEVATION_DEG,
  SWEEP_AZIMUTH_DEG,
  SWEEP_DISTANCE_FACTOR,
  SWEEP_ROLL_DEG,
  SWEEP_FOV_DEG,
  SWEEP_OFFSET_X_FACTOR,
  SWEEP_OFFSET_Y_FACTOR,
];

describe("between", () => {
  it("hits the low and high bound at the extremes of random()", () => {
    const range: Range = { min: -10, max: 30 };
    expect(between(() => 0, range)).toBe(range.min);
    expect(between(() => 1, range)).toBe(range.max);
    expect(between(() => 0.5, range)).toBe(10);
  });

  it("keeps every declared sweep range non-empty and ordered", () => {
    for (const range of RANGES) {
      expect(range.max).toBeGreaterThan(range.min);
    }
  });
});

describe("samplePose", () => {
  it("keeps every sampled field inside its declared range", () => {
    for (let i = 0; i < 200; i += 1) {
      const seed = i / 200;
      const pose = samplePose(() => seed);
      expect(pose.elevationDeg).toBeGreaterThanOrEqual(SWEEP_ELEVATION_DEG.min);
      expect(pose.elevationDeg).toBeLessThanOrEqual(SWEEP_ELEVATION_DEG.max);
      expect(pose.azimuthDeg).toBeGreaterThanOrEqual(SWEEP_AZIMUTH_DEG.min);
      expect(pose.azimuthDeg).toBeLessThanOrEqual(SWEEP_AZIMUTH_DEG.max);
      expect(pose.distanceFactor).toBeGreaterThanOrEqual(
        SWEEP_DISTANCE_FACTOR.min,
      );
      expect(pose.distanceFactor).toBeLessThanOrEqual(
        SWEEP_DISTANCE_FACTOR.max,
      );
      expect(pose.fovDeg).toBeGreaterThanOrEqual(SWEEP_FOV_DEG.min);
      expect(pose.fovDeg).toBeLessThanOrEqual(SWEEP_FOV_DEG.max);
    }
  });
});

describe("noteName", () => {
  it("names the standard board starts", () => {
    expect(noteName(21)).toBe("A0");
    expect(noteName(28)).toBe("E1");
    expect(noteName(36)).toBe("C2");
    expect(noteName(60)).toBe("C4");
  });
});

describe("BOARD_SIZES", () => {
  it("carries the four sizes the app is meant to generalise across", () => {
    const byKeys = new Map(BOARD_SIZES.map((board) => [board.keys, board]));
    expect(byKeys.get(49)?.whiteKeys).toBe(29);
    expect(byKeys.get(49)?.lowestNote).toBe("C2");
    expect(byKeys.get(61)?.whiteKeys).toBe(36);
    expect(byKeys.get(61)?.lowestNote).toBe("C2");
    expect(byKeys.get(76)?.whiteKeys).toBe(45);
    expect(byKeys.get(76)?.lowestNote).toBe("E1");
    expect(byKeys.get(88)?.whiteKeys).toBe(52);
    expect(byKeys.get(88)?.lowestNote).toBe("A0");
  });

  it("only ever picks a size that is actually declared", () => {
    for (let i = 0; i < 50; i += 1) {
      const board = pickBoardSize(() => i / 50);
      expect(BOARD_SIZES).toContain(board);
    }
  });
});

describe("case dressing ranges", () => {
  it("keeps the case depth and cheek width ranges non-empty and in millimetres", () => {
    expect(CASE_BACK_DEPTH_MM.min).toBe(60);
    expect(CASE_BACK_DEPTH_MM.max).toBe(250);
    expect(CASE_CHEEK_WIDTH_MM.min).toBe(20);
    expect(CASE_CHEEK_WIDTH_MM.max).toBe(80);
  });
});

describe("pickCaseColorFamily", () => {
  it("always returns one of the declared families with ordered ranges", () => {
    for (let i = 0; i < 50; i += 1) {
      const spec = pickCaseColorFamily(() => i / 50);
      expect(["black", "darkGrey", "silver", "white", "wood"]).toContain(
        spec.family,
      );
      for (const range of [
        spec.hue,
        spec.saturation,
        spec.lightness,
        spec.roughness,
        spec.metalness,
      ]) {
        expect(range.max).toBeGreaterThanOrEqual(range.min);
      }
    }
  });

  it("picks black or dark grey most of the time, matching a mostly-plastic case", () => {
    const counts = { dark: 0, other: 0 };
    const samples = 2000;
    for (let i = 0; i < samples; i += 1) {
      const spec = pickCaseColorFamily(() => (i + 0.5) / samples);
      if (spec.family === "black" || spec.family === "darkGrey") {
        counts.dark += 1;
      } else {
        counts.other += 1;
      }
    }
    expect(counts.dark / samples).toBeCloseTo(0.7, 1);
  });
});
