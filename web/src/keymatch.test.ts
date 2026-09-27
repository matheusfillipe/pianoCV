import { describe, expect, it } from "vitest";
import { measureSlant, pairRuns, peaks, squaredOutline } from "./keymatch";

describe("measureSlant and squaredOutline", () => {
  it("follows a slant that grows past half a key and turns the outline's ends onto the keys", () => {
    // the gaps move 1% of their position between the two depths, so the far gaps move 12 px of a
    // 26 px key, a pairing by nearest gap alone would take the neighbour there
    const dips = Array.from({ length: 45 }, (_, i) => 13 + i * 26);
    const shallow = dips.map((x) => x + 0.01 * x);
    const moves = measureSlant(dips, shallow);
    expect(moves?.length).toBe(45);
    const outline = moves === null ? null : squaredOutline(moves);
    expect(outline?.[0].x).toBeCloseTo(0, 5);
    // at the far edge each line has leaned by its move over the white row's own depth
    expect(outline?.[1].x).toBeCloseTo(1200 + 12 / 0.16, 5);
    expect(outline?.[2].x).toBeCloseTo(1200 - (12 * 0.12) / (0.88 * 0.16), 5);
  });

  it("keeps the outline's ends where they are when the gaps do not move", () => {
    const dips = Array.from({ length: 45 }, (_, i) => 13 + i * 26);
    const moves = measureSlant(dips, dips);
    const outline = moves === null ? null : squaredOutline(moves, 40, 1160);
    expect(outline?.map((corner) => Math.round(corner.x))).toEqual([
      40, 1160, 1160, 40,
    ]);
  });
});

describe("peaks", () => {
  it("keeps confident local maxima and drops a weaker echo beside a stronger one", () => {
    const values = new Float32Array(40);
    values[10] = 0.9;
    values[12] = 0.6;
    values[25] = 0.5;
    values[33] = 0.2;
    expect(peaks(values)).toEqual([10, 25]);
  });
});

describe("pairRuns", () => {
  it("pairs each left edge with the next right edge and refuses one too wide for a black key", () => {
    const runs = pairRuns([10, 40, 90], [22, 55, 140], 20);
    expect(runs.map((run) => [run.start, run.end])).toEqual([
      [10, 22],
      [40, 55],
    ]);
  });

  it("refuses a sliver too narrow for a black key", () => {
    expect(pairRuns([10, 40], [13, 55], 20, 6).map((run) => run.start)).toEqual(
      [40],
    );
  });

  it("never pairs across a left edge that comes first", () => {
    expect(pairRuns([10, 15], [30], 40).map((run) => run.start)).toEqual([15]);
  });
});
