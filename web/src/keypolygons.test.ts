import { describe, expect, it } from "vitest";
import { keyPolygons } from "./keypolygons";

describe("keyPolygons", () => {
  it("projects every physical key from A0 through C8", () => {
    const keys = keyPolygons([
      { x: 0, y: 0 },
      { x: 1, y: 0 },
      { x: 1, y: 1 },
      { x: 0, y: 1 },
    ]);

    expect(keys).toHaveLength(88);
    expect(keys[0]).toMatchObject({ pitch: 21, black: false });
    expect(keys[87]).toMatchObject({ pitch: 108, black: false });
    expect(keys[0]?.points[0]).toEqual({ x: 0, y: 0 });
    expect(keys[87]?.points[2]).toEqual({ x: 1, y: 1 });
  });
});
