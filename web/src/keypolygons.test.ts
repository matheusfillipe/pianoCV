import { describe, expect, it } from "vitest";
import { keyPolygons } from "./keypolygons";
import { keyUnits } from "./keys";

describe("keyPolygons", () => {
  it("projects every physical key from A0 through C8", () => {
    const keys = keyPolygons(
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ],
      {
        lowest: 21,
        highest: 108,
        origin: keyUnits(21).from,
        span: keyUnits(108).to - keyUnits(21).from,
        blackDepth: 0.62,
      },
    );

    expect(keys).toHaveLength(88);
    expect(keys[0]).toMatchObject({ pitch: 21, black: false });
    expect(keys[87]).toMatchObject({ pitch: 108, black: false });
    expect(keys[0]?.points[0]).toEqual({ x: 0, y: 0 });
    expect(keys[87]?.points[2]).toEqual({ x: 1, y: 1 });
  });

  it("uses measured black-key geometry when it is available", () => {
    const board = {
      lowest: 60,
      highest: 61,
      origin: keyUnits(60).from,
      span: 7,
      blackDepth: 0.62,
      blackKeys: [
        { pitch: 61, u0: 0.12, u1: 0.2, depth: 0.48, confidence: 0.9 },
      ],
    };
    const legacy = keyPolygons(
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ],
      { ...board, blackKeys: undefined },
    );
    const measured = keyPolygons(
      [
        { x: 0, y: 0 },
        { x: 1, y: 0 },
        { x: 1, y: 1 },
        { x: 0, y: 1 },
      ],
      board,
    );
    const legacyBlack = legacy.find((key) => key.black);
    const measuredBlack = measured.find((key) => key.black);
    expect(legacyBlack).toBeDefined();
    expect(measuredBlack?.points).toEqual([
      { x: 0.12, y: 0 },
      { x: 0.2, y: 0 },
      { x: 0.2, y: 0.48 },
      { x: 0.12, y: 0.48 },
    ]);
  });
});
