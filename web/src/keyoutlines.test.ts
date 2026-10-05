import { readFileSync } from "node:fs";
import { beforeAll, expect, test } from "vitest";
import steepBoard from "./fixtures/steep-board-peaks.json";
import type { KeyNetFit } from "./keycore";
import { keyOutlines } from "./keycore";
import * as keycore from "./keycore-wasm/keycore.js";

beforeAll(() => {
  keycore.initSync({
    module: readFileSync(
      new URL("./keycore-wasm/keycore_bg.wasm", import.meta.url),
    ),
  });
});

test("every key of a fitted 76-key board comes out once, low to high", () => {
  const fit: KeyNetFit = JSON.parse(
    keycore.fit_keyboard(
      JSON.stringify(steepBoard.peaks),
      steepBoard.size.width,
      steepBoard.size.height,
    ),
  );
  const keys = keyOutlines(fit);
  expect(keys).toHaveLength(76);
  expect(keys.map((key) => key.semitone)).toEqual(
    Array.from({ length: 76 }, (_, i) => i),
  );
  expect(keys[0]?.from).toBe(0);
  expect(keys.at(-1)?.to).toBe(fit.whiteKeys);
  for (const key of keys.filter((k) => k.black)) {
    expect(key.bar.length).toBeGreaterThanOrEqual(4);
  }
});
