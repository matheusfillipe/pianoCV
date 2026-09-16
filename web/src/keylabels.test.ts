import { describe, expect, test } from "vitest";
import { instanceMaskColor, keyIdSpecs } from "./keylabels";

describe("key ID labels", () => {
  test("covers the source model in pitch order", () => {
    const keys = keyIdSpecs();
    expect(keys).toHaveLength(96);
    expect(keys[0]).toMatchObject({ pitch: 17, label: "F0", black: false });
    expect(keys[1]).toMatchObject({ pitch: 18, label: "F#0", black: true });
    expect(keys.at(-1)).toMatchObject({
      pitch: 112,
      label: "E8",
      black: false,
    });
  });

  test("keeps every key inside the canonical keybed", () => {
    for (const key of keyIdSpecs()) {
      expect(key.u0).toBeGreaterThanOrEqual(0);
      expect(key.u1).toBeLessThanOrEqual(1);
      expect(key.u1).toBeGreaterThan(key.u0);
      expect(key.v1).toBeGreaterThan(0);
      expect(key.v1).toBeLessThanOrEqual(1);
    }
  });

  test("assigns a unique instance color to every source key", () => {
    const values = keyIdSpecs().map((_, index) => instanceMaskColor(index));
    expect(new Set(values).size).toBe(96);
    expect(values[0]).not.toBe(0);
    expect(values.at(-1)).not.toBe(0);
  });
});
