import { readFileSync } from "node:fs";
import { beforeAll, expect, test } from "vitest";
import steepBoard from "./fixtures/steep-board-peaks.json";
import * as keycore from "./keycore-wasm/keycore.js";

beforeAll(() => {
  keycore.initSync({
    module: readFileSync(
      new URL("./keycore-wasm/keycore_bg.wasm", import.meta.url),
    ),
  });
});

test("wasm fits the real steep board from its peaks", () => {
  const fit = JSON.parse(
    keycore.fit_keyboard(
      JSON.stringify(steepBoard.peaks),
      steepBoard.size.width,
      steepBoard.size.height,
    ),
  );
  expect(fit.whiteKeys).toBe(steepBoard.whiteKeys);
});

test("wasm decodes a heatmap and builds a model input of the crop's size", () => {
  const identity = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const peaks = JSON.parse(
    keycore.decode_heatmaps(new Float32Array(9 * 2 * 2), 4, 4, identity),
  );
  expect(peaks.gaps).toEqual([]);
  const input = keycore.prepare_input(
    new Uint8Array(4 * 4 * 4),
    4,
    4,
    identity,
    4,
    4,
  );
  expect(input).toHaveLength(3 * 4 * 4);
});
