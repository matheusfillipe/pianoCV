import { readFileSync } from "node:fs";
import { beforeAll, describe, expect, test } from "vitest";
import { applyHomography, type Homography, type Point } from "./homography";
import { keyboardTemplate, type ScoredPoint } from "./keycore";
import { initSync } from "./keycore-wasm/keycore.js";
import {
  type FixedLabels,
  lacksRear,
  parseLabels,
  prefill,
  reproject,
  serialise,
} from "./labelfix";

const HOMOGRAPHY: Homography = [60, 8, 40, 3, 20, 100, 0.0004, 0.0002, 1];
const KEYS = 12;
const PHASE = "C";

beforeAll(() => {
  initSync({
    module: readFileSync(
      new URL("./keycore-wasm/keycore_bg.wasm", import.meta.url),
    ),
  });
});

function px(p: Point): [number, number] {
  const q = applyHomography(HOMOGRAPHY, p.x, p.y);
  return [q.x, q.y];
}

function exactLabels(): FixedLabels {
  const template = keyboardTemplate(KEYS, PHASE);
  const blackLow = template.blackLow.map(px);
  const blackHigh = template.blackHigh.map(px);
  const lifted = (points: [number, number][]): [number, number][] =>
    points.map(([x, y]) => [x + 1, y - 9]);
  return {
    width: 1000,
    height: 500,
    whiteKeys: KEYS,
    phase: PHASE,
    corners: template.corners.map(px),
    gaps: template.gaps.map(px),
    blackLow,
    blackHigh,
    blackTopLow: lifted(blackLow),
    blackTopHigh: lifted(blackHigh),
    backGaps: template.backGaps.map(px),
    blackBackLow: lifted(blackLow),
    blackBackHigh: lifted(blackHigh),
  };
}

function expectClose(
  actual: readonly number[] | null,
  expected: readonly number[],
): void {
  expect(actual).not.toBeNull();
  expect(actual?.[0]).toBeCloseTo(expected[0], 3);
  expect(actual?.[1]).toBeCloseTo(expected[1], 3);
}

describe("labelfix", () => {
  test("serialise round trips with hidden points", () => {
    const labels = exactLabels();
    labels.gaps[2] = null;
    labels.blackTopHigh[0] = null;
    expect(parseLabels(serialise(labels))).toEqual(labels);
  });

  test("shifting one key puts gap i on the projection of template gap i+1", () => {
    const template = keyboardTemplate(KEYS, PHASE);
    const shifted = reproject(exactLabels(), 1);
    for (let i = 0; i < template.gaps.length - 1; i += 1) {
      expectClose(shifted.gaps[i], px(template.gaps[i + 1]));
    }
    const back = reproject(exactLabels(), -1);
    expectClose(back.gaps[3], px(template.gaps[2]));
  });

  test("an old file parses with null rear arrays of the right length", () => {
    const { backGaps, blackBackLow, blackBackHigh, ...old } = exactLabels();
    const text = JSON.stringify(old);
    const parsed = parseLabels(text);
    expect(lacksRear(text)).toBe(true);
    expect(parsed.backGaps).toEqual(Array(KEYS - 1).fill(null));
    expect(parsed.blackBackLow).toHaveLength(parsed.blackLow.length);
    expect(parsed.blackBackHigh.every((p) => p === null)).toBe(true);
  });

  test("shifting one key moves back gaps by one key", () => {
    const template = keyboardTemplate(KEYS, PHASE);
    const shifted = reproject(exactLabels(), 1);
    for (let i = 0; i < template.backGaps.length - 1; i += 1) {
      expectClose(shifted.backGaps[i], px(template.backGaps[i + 1]));
    }
  });

  test("black tops move with their bottoms", () => {
    const labels = exactLabels();
    const shifted = reproject(labels, 1);
    const bottom = shifted.blackLow[0];
    const top = shifted.blackTopLow[0];
    expect(bottom).not.toBeNull();
    expect(top).not.toBeNull();
    expectClose(top, [(bottom?.[0] ?? 0) + 1, (bottom?.[1] ?? 0) - 9]);
  });

  test("refit leaves an exact projection in place", () => {
    const labels = exactLabels();
    const refit = reproject(labels, 0);
    expectClose(refit.gaps[4], labels.gaps[4] ?? []);
    expectClose(refit.blackHigh[1], labels.blackHigh[1] ?? []);
  });

  test("hidden points stay null", () => {
    const labels = exactLabels();
    labels.gaps[2] = null;
    labels.blackLow[1] = null;
    labels.blackTopLow[0] = null;
    const shifted = reproject(labels, 1);
    expect(shifted.gaps[2]).toBeNull();
    expect(shifted.blackLow[1]).toBeNull();
    expect(shifted.blackTopLow[0]).toBeNull();
    expect(reproject(labels, 1, true).gaps[2]).not.toBeNull();
  });

  test("prefill takes the nearest top peak within a white key, else null", () => {
    const size = { width: 1000, height: 500 };
    const fractions = (p: [number, number]): ScoredPoint => ({
      score: 1,
      x: (p[0] + 2) / size.width,
      y: (p[1] - 3) / size.height,
    });
    const template = keyboardTemplate(KEYS, PHASE);
    const near = fractions(px(template.blackLow[0]));
    const far = { x: 0.99, y: 0.99, score: 1 };
    const fit = {
      homography: HOMOGRAPHY.map((v, i) =>
        i < 3 ? v / size.width : i < 6 ? v / size.height : v,
      ),
      whiteKeys: KEYS,
      phase: PHASE,
      lift: null,
      blackDepth: keyboardTemplate(KEYS, PHASE).blackLow[0]?.y ?? 0,
    };
    const labels = prefill(
      fit,
      {
        blackTopLow: [near, far],
        blackTopHigh: [],
        blackBackLow: [],
        blackBackHigh: [],
      },
      size,
    );
    for (const [i, gap] of template.backGaps.entries()) {
      expectClose(labels.backGaps[i], px(gap));
    }
    expect(labels.blackBackLow.every((p) => p === null)).toBe(true);
    expect(labels.blackTopLow[0]).not.toBeNull();
    expect(labels.blackTopLow[1]).toBeNull();
    expect(labels.blackTopHigh[0]).toBeNull();
    expect(labels.gaps).toHaveLength(KEYS - 1);
  });
});
