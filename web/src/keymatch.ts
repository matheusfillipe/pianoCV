import * as ort from "onnxruntime-web/wasm";
import type { RuntimeAssets } from "./assets";
import { applyHomography, findHomography, type Point } from "./homography";
import {
  fitLine,
  type KeyEvidence,
  type LinePoint,
  lineAt,
  lineResidual,
  outlineQuad,
  PLAIN_OUTLINE,
  type Run,
  rectifyStrip,
  type SourceImage,
  STRIP_HEIGHT,
  STRIP_WIDTH,
} from "./keystrip";

export const KEYMATCH_URL = "/keymatch.onnx";
const MATCH_WIDTH = 768;
const MATCH_HEIGHT = 64;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
/** How sure the matcher must be of an edge before we take it as one. */
const PEAK_THRESHOLD = 0.35;
/** Two peaks closer than this, in matcher columns, are one edge read twice. */
const PEAK_SPACING = 7;
/** The widest and narrowest a black key can be, against the spacing of the white-key gaps. */
const MOST_BLACK_WIDTH = 0.9;
const LEAST_BLACK_WIDTH = 0.3;
/** The strip row the matcher was trained to find white-key gaps on, as a share of its depth. */
const WHITE_ROW = 0.88;
/** How much of the keybed's depth the second, shallower look spans, so its white-key row sits
 * nearer the black keys and the same gaps are seen at a second depth. */
const SHALLOW_DEPTH = 0.84;
const SHALLOW_WHITE_ROW = WHITE_ROW * SHALLOW_DEPTH;
/** How far a gap may move between the two depths from where the previous gap moved, in keys. */
const SLANT_TRACK_KEYS = 0.3;
const SLANT_OUTLIER_KEYS = 0.08;
const SLANT_MIN_GAPS = 8;
/** We read a little past the outline's ends, as a share of its length, so keys the outline
 * cuts off still show their lines. */
const PAD_SHARE = 0.06;
/** We stop refining once a round moves no corner further than this, in strip pixels. */
const SETTLED_PX = 1;
const MOST_ROUNDS = 3;
/** A round that turns an outline end further than this, in strip pixels, misread the keys. */
const MOST_TURN_PX = 120;

export type Matched = {
  /** The outline the keys were read on: the given quad with its ends turned to run along the
   * white keys the matcher saw, as frame fractions. */
  readonly quad: readonly Point[];
  /** Where that outline's corners sit in the strip of the quad that was given, in strip pixels,
   * so the correction can follow the quad as it moves. */
  readonly outline: readonly Point[];
  readonly evidence: KeyEvidence;
};

export type KeyMatcher = {
  /** The keys the matcher sees on `quad`, once the quad is squared to the keys, or null when it
   * saw too little to stand in for the brightness rules. */
  readonly match: (
    source: SourceImage,
    quad: readonly Point[],
  ) => Promise<Matched | null>;
};

/** Local maxima above the threshold, strongest first, none within `PEAK_SPACING` of a stronger. */
export function peaks(values: Float32Array): number[] {
  const candidates: number[] = [];
  for (let i = 1; i < values.length - 1; i += 1) {
    if (
      values[i] >= PEAK_THRESHOLD &&
      values[i] >= values[i - 1] &&
      values[i] >= values[i + 1]
    ) {
      candidates.push(i);
    }
  }
  candidates.sort((a, b) => values[b] - values[a]);
  const kept: number[] = [];
  for (const candidate of candidates) {
    if (kept.every((other) => Math.abs(other - candidate) >= PEAK_SPACING)) {
      kept.push(candidate);
    }
  }
  return kept.sort((a, b) => a - b);
}

/** Pairs each black key's left edge with the first right edge after it, before the next left
 * edge, and as wide as a black key can be. */
export function pairRuns(
  lefts: readonly number[],
  rights: readonly number[],
  widest: number,
  narrowest = 0,
): Run[] {
  const runs: Run[] = [];
  lefts.forEach((start, i) => {
    const nextLeft = lefts[i + 1] ?? Number.POSITIVE_INFINITY;
    const end = rights.find((x) => x > start && x < nextLeft);
    if (
      end !== undefined &&
      end - start <= widest &&
      end - start >= narrowest
    ) {
      runs.push({ start, end, center: (start + end) / 2 });
    }
  });
  return runs;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

/** How far each white-key gap moves between the white row and the shallow row, as a line over
 * the strip. White keys lie flat, so this is the slant the quad's ends give every key line; we
 * follow the gaps from the start of the strip so a slant past half a key never pairs a gap with
 * its neighbour. */
export function measureSlant(
  dips: readonly number[],
  shallowDips: readonly number[],
): LinePoint[] | null {
  const keyWidth = median(dips.slice(1).map((x, i) => x - dips[i]));
  if (keyWidth <= 0) {
    return null;
  }
  const moves: LinePoint[] = [];
  let expected = 0;
  for (const dip of dips) {
    let nearest: number | null = null;
    for (const x of shallowDips) {
      if (
        nearest === null ||
        Math.abs(x - dip - expected) < Math.abs(nearest - dip - expected)
      ) {
        nearest = x;
      }
    }
    if (
      nearest !== null &&
      Math.abs(nearest - dip - expected) < keyWidth * SLANT_TRACK_KEYS
    ) {
      moves.push({ x: dip, y: nearest - dip });
      expected = nearest - dip;
    }
  }
  if (moves.length < SLANT_MIN_GAPS) {
    return null;
  }
  const first = fitLine(moves);
  const kept = moves.filter(
    (move) => lineResidual(first, move) <= keyWidth * SLANT_OUTLIER_KEYS,
  );
  return kept.length < SLANT_MIN_GAPS ? null : kept;
}

/** The corners, in strip pixels, of the outline whose ends at `left` and `right` run along the
 * white keys: every key line passes through its white-row gap and leans by the measured slant
 * per unit of depth. Key lines on a flat keybed meet at one vanishing point, which is the same
 * as their slant changing linearly along the strip, so one line fit carries the perspective. */
export function squaredOutline(
  moves: readonly LinePoint[],
  left = 0,
  right = STRIP_WIDTH,
): Point[] {
  const slant = fitLine(moves);
  const lean = (x: number, row: number): number =>
    x +
    (lineAt(slant, x) * (WHITE_ROW - row)) / (WHITE_ROW - SHALLOW_WHITE_ROW);
  return [
    { x: lean(left, 0), y: 0 },
    { x: lean(right, 0), y: 0 },
    { x: lean(right, 1), y: STRIP_HEIGHT },
    { x: lean(left, 1), y: STRIP_HEIGHT },
  ];
}

function shallowQuad(quad: readonly Point[]): Point[] {
  return outlineQuad(quad, [
    { x: 0, y: 0 },
    { x: STRIP_WIDTH, y: 0 },
    { x: STRIP_WIDTH, y: STRIP_HEIGHT * SHALLOW_DEPTH },
    { x: 0, y: STRIP_HEIGHT * SHALLOW_DEPTH },
  ]);
}

export async function createKeyMatcher(
  assets: RuntimeAssets,
  url: string = KEYMATCH_URL,
): Promise<KeyMatcher> {
  ort.env.wasm.wasmPaths = { wasm: assets.ortWasm };
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const input = new Float32Array(3 * MATCH_HEIGHT * MATCH_WIDTH);
  const plane = MATCH_HEIGHT * MATCH_WIDTH;
  const scale = STRIP_WIDTH / MATCH_WIDTH;

  const heatmaps = async (
    source: SourceImage,
    quad: readonly Point[],
  ): Promise<Float32Array | null> => {
    const strip = rectifyStrip(source, quad, MATCH_WIDTH, MATCH_HEIGHT);
    if (strip === null) {
      return null;
    }
    for (let i = 0; i < plane; i += 1) {
      for (let c = 0; c < 3; c += 1) {
        input[c * plane + i] = (strip.data[i * 4 + c] / 255 - MEAN[c]) / STD[c];
      }
    }
    const outputs = await session.run({
      [session.inputNames[0]]: new ort.Tensor("float32", input, [
        1,
        3,
        MATCH_HEIGHT,
        MATCH_WIDTH,
      ]),
    });
    const heat = outputs[session.outputNames[0]]?.data;
    return heat instanceof Float32Array ? heat : null;
  };
  const channel = (heat: Float32Array, c: number): number[] =>
    peaks(heat.subarray(c * MATCH_WIDTH, (c + 1) * MATCH_WIDTH)).map(
      (x) => x * scale,
    );
  const read = async (
    source: SourceImage,
    quad: readonly Point[],
  ): Promise<KeyEvidence | null> => {
    const heat = await heatmaps(source, quad);
    if (heat === null) {
      return null;
    }
    const dips = channel(heat, 0);
    const gaps = dips.slice(1).map((x, i) => x - dips[i]);
    const runs = pairRuns(
      channel(heat, 1),
      channel(heat, 2),
      median(gaps) * MOST_BLACK_WIDTH,
      median(gaps) * LEAST_BLACK_WIDTH,
    );
    return dips.length < 2 || runs.length === 0 ? null : { dips, runs };
  };
  const pad = STRIP_WIDTH * PAD_SHARE;
  const padded = [
    { x: -pad, y: 0 },
    { x: STRIP_WIDTH + pad, y: 0 },
    { x: STRIP_WIDTH + pad, y: STRIP_HEIGHT },
    { x: -pad, y: STRIP_HEIGHT },
  ];
  // where the unpadded outline's ends fall in the padded strip's own pixels
  const inPadded = (x: number): number =>
    ((x + pad) * STRIP_WIDTH) / (STRIP_WIDTH + 2 * pad);
  /** One round: read the padded outline at two depths and turn its ends onto the key lines. */
  const squareOnce = async (
    source: SourceImage,
    outline: readonly Point[],
  ): Promise<Point[] | null> => {
    const wide = outlineQuad(outline, padded);
    const [near, shallow] = await Promise.all([
      heatmaps(source, wide),
      heatmaps(source, shallowQuad(wide)),
    ]);
    const moves =
      near === null || shallow === null
        ? null
        : measureSlant(channel(near, 0), channel(shallow, 0));
    if (moves === null) {
      return null;
    }
    const corners = squaredOutline(moves, inPadded(0), inPadded(STRIP_WIDTH));
    const plain = [inPadded(0), inPadded(STRIP_WIDTH)];
    const turn = Math.max(
      ...corners.map((corner, i) =>
        Math.abs(corner.x - plain[i === 0 || i === 3 ? 0 : 1]),
      ),
    );
    return turn > MOST_TURN_PX ? null : outlineQuad(wide, corners);
  };

  return {
    match: async (source, quad) => {
      let outline: readonly Point[] = quad;
      for (let round = 0; round < MOST_ROUNDS; round += 1) {
        const next = await squareOnce(source, outline);
        if (next === null) {
          break;
        }
        const moved = Math.max(
          ...next.map((p, i) =>
            Math.hypot(
              (p.x - outline[i].x) * source.width,
              (p.y - outline[i].y) * source.height,
            ),
          ),
        );
        outline = next;
        if (moved < SETTLED_PX) {
          break;
        }
      }
      const evidence = await read(source, outline);
      if (evidence === null) {
        return null;
      }
      const toStrip = findHomography(
        quad.slice(0, 4).map((p) => ({ x: p.x, y: p.y })),
        [...PLAIN_OUTLINE],
      );
      return {
        quad: outline,
        outline: outline.map((p) => applyHomography(toStrip, p.x, p.y)),
        evidence,
      };
    },
  };
}
