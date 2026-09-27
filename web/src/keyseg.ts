import type { RuntimeAssets } from "./assets";
import type { Point } from "./homography";
import type { SegmentReply, SegmentRequest } from "./keyseg-worker";
import type { Size } from "./keyspace";
import {
  fitLine,
  type Line,
  type LinePoint,
  lineAt,
  lineResidual,
  type SourceImage,
} from "./keystrip";

export const KEYSEG_URL = "/keyseg.onnx";
export const CROP_WIDTH = 1024;
export const CROP_HEIGHT = 224;
const WHITE = 1;
const BLACK = 2;
const BOUNDARY = 3;
const CLASSES = 4;
// the same margins the model was trained with, as shares of the quad's length
const MARGIN_ALONG = 0.08;
const MARGIN_ACROSS = 0.15;
/** A column needs this many key pixels to say where the keybed's edges are. */
const LEAST_COLUMN_PX = 3;
const LEAST_EDGE_POINTS = 20;
/** How far, in crop pixels, an edge's own pixels may sit from its line. */
const EDGE_WITHIN_PX = 2.5;

/** Rotates and scales the frame so the keys run left to right with the player's edge at the
 * bottom, around a rough keybed quad; it never bends the frame, so keys keep their own shape. */
export type Crop = {
  readonly centre: Point;
  readonly along: Point;
  readonly across: Point;
  /** Frame pixels per crop pixel. */
  readonly scale: number;
};

export type Segmented = {
  /** The keybed outline the key pixels show, as frame fractions, or null when too few were
   * found to draw one. */
  readonly outline: Point[] | null;
  /** Every key the segmenter saw, with outlines in frame fractions. */
  readonly regions: readonly KeyRegion[];
};

export type KeySegmenter = {
  /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
  readonly backend: string;
  readonly gpuFailure: string | null;
  /** The keybed outline and the keys the segmenter sees around `quad`. */
  readonly segment: (
    frame: CanvasImageSource | SourceImage,
    size: Size,
    quad: readonly Point[],
  ) => Promise<Segmented>;
};

const unit = (p: Point): Point => {
  const length = Math.hypot(p.x, p.y) || 1;
  return { x: p.x / length, y: p.y / length };
};
const dot = (a: Point, b: Point): number => a.x * b.x + a.y * b.y;

/** The crop around `quad`, given in frame pixels, the same one the model was trained on. */
export function cropFor(quad: readonly Point[]): Crop {
  const [farLeft, farRight, nearRight, nearLeft] = quad;
  const along = unit({
    x: farRight.x - farLeft.x + nearRight.x - nearLeft.x,
    y: farRight.y - farLeft.y + nearRight.y - nearLeft.y,
  });
  const depth = {
    x: nearLeft.x - farLeft.x + nearRight.x - farRight.x,
    y: nearLeft.y - farLeft.y + nearRight.y - farRight.y,
  };
  const perpendicular = { x: -along.y, y: along.x };
  const across =
    dot(perpendicular, depth) < 0
      ? { x: -perpendicular.x, y: -perpendicular.y }
      : perpendicular;
  const centre = {
    x: quad.slice(0, 4).reduce((sum, p) => sum + p.x, 0) / 4,
    y: quad.slice(0, 4).reduce((sum, p) => sum + p.y, 0) / 4,
  };
  const spread = (axis: Point): number => {
    const values = quad
      .slice(0, 4)
      .map((p) => dot({ x: p.x - centre.x, y: p.y - centre.y }, axis));
    return Math.max(...values) - Math.min(...values);
  };
  const alongExtent = spread(along) * (1 + 2 * MARGIN_ALONG);
  const acrossExtent =
    spread(across) + (alongExtent * MARGIN_ACROSS) / (1 + 2 * MARGIN_ALONG);
  const scale = Math.max(
    alongExtent / CROP_WIDTH,
    acrossExtent / CROP_HEIGHT,
    1e-6,
  );
  return { centre, along, across, scale };
}

export function cropToFrame(crop: Crop, x: number, y: number): Point {
  const u = (x - CROP_WIDTH / 2) * crop.scale;
  const v = (y - CROP_HEIGHT / 2) * crop.scale;
  return {
    x: crop.centre.x + u * crop.along.x + v * crop.across.x,
    y: crop.centre.y + u * crop.along.y + v * crop.across.y,
  };
}

/** Each pixel's most likely class, row by row. */
export function classesOf(probabilities: Float32Array): Uint8Array {
  const plane = CROP_WIDTH * CROP_HEIGHT;
  const classes = new Uint8Array(plane);
  for (let i = 0; i < plane; i += 1) {
    let best = 0;
    for (let c = 1; c < CLASSES; c += 1) {
      if (probabilities[c * plane + i] > probabilities[best * plane + i]) {
        best = c;
      }
    }
    classes[i] = best;
  }
  return classes;
}

function median(values: number[]): number {
  const sorted = values.sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

/** Robust line through points: a Theil-Sen start, the median of pairwise slopes, which a
 * minority of stray points cannot tilt, then least-squares refits over the points within
 * `within` of the line. */
function robustLine(points: readonly LinePoint[], within: number): Line | null {
  if (points.length < LEAST_EDGE_POINTS) {
    return null;
  }
  const step = Math.max(1, Math.floor(points.length / 60));
  const slopes: number[] = [];
  for (let i = 0; i < points.length; i += step) {
    for (let j = i + step; j < points.length; j += step) {
      const dx = points[j].x - points[i].x;
      if (dx !== 0) {
        slopes.push((points[j].y - points[i].y) / dx);
      }
    }
  }
  const slope = median(slopes);
  const intercept = median(points.map((p) => p.y - slope * p.x));
  let line: Line = { meanX: 0, meanY: intercept, slope };
  for (let round = 0; round < 3; round += 1) {
    const kept = points.filter((p) => lineResidual(line, p) <= within);
    if (kept.length < LEAST_EDGE_POINTS) {
      return null;
    }
    line = fitLine(kept);
  }
  return line;
}

const isKey = (c: number): boolean =>
  c === WHITE || c === BLACK || c === BOUNDARY;
const isWhite = (c: number): boolean => c === WHITE || c === BOUNDARY;

/** The keybed outline the key pixels show, in crop pixels, as far-left, far-right, near-right,
 * near-left: the far edge through each column's first key pixel, the near edge through its last
 * white one, and each end through each row's first and last key pixel. */
export function keyOutline(classes: Uint8Array): Point[] | null {
  const tops: LinePoint[] = [];
  const bottoms: LinePoint[] = [];
  for (let x = 0; x < CROP_WIDTH; x += 1) {
    let top = -1;
    let bottom = -1;
    let count = 0;
    for (let y = 0; y < CROP_HEIGHT; y += 1) {
      const c = classes[y * CROP_WIDTH + x];
      if (isKey(c)) {
        count += 1;
        if (top < 0) {
          top = y;
        }
      }
      if (isWhite(c)) {
        bottom = y;
      }
    }
    if (count >= LEAST_COLUMN_PX && bottom > top) {
      tops.push({ x, y: top - 0.5 });
      bottoms.push({ x, y: bottom + 0.5 });
    }
  }
  const far = robustLine(tops, EDGE_WITHIN_PX);
  const near = robustLine(bottoms, EDGE_WITHIN_PX);
  if (far === null || near === null) {
    return null;
  }
  const lefts: LinePoint[] = [];
  const rights: LinePoint[] = [];
  const middle = CROP_WIDTH / 2;
  for (let y = 0; y < CROP_HEIGHT; y += 1) {
    if (y < lineAt(far, middle) || y > lineAt(near, middle)) {
      continue;
    }
    let left = -1;
    let right = -1;
    for (let x = 0; x < CROP_WIDTH; x += 1) {
      if (isKey(classes[y * CROP_WIDTH + x])) {
        if (left < 0) {
          left = x;
        }
        right = x;
      }
    }
    if (left >= 0) {
      // x as a function of y, since the ends run across the crop
      lefts.push({ x: y, y: left - 0.5 });
      rights.push({ x: y, y: right + 0.5 });
    }
  }
  const left = robustLine(lefts, EDGE_WITHIN_PX);
  const right = robustLine(rights, EDGE_WITHIN_PX);
  if (left === null || right === null) {
    return null;
  }
  // where y = far(x) meets x = end(y)
  const meet = (edge: Line, end: Line): Point => {
    const m = edge.slope;
    const c = edge.meanY - edge.slope * edge.meanX;
    const b = end.slope;
    const a = end.meanY - end.slope * end.meanX;
    const x = (a + b * c) / (1 - b * m);
    return { x, y: m * x + c };
  };
  return [
    meet(far, left),
    meet(far, right),
    meet(near, right),
    meet(near, left),
  ];
}

/** A region smaller than this many crop pixels is a speck, never a key. */
const LEAST_KEY_PX = 12;
/** How far, in crop pixels, a simplified key outline may stray from the key's own pixels. */
const OUTLINE_TOLERANCE_PX = 0.8;

/** One key's pixels as the segmenter saw them: its outline and where its pixels sit. */
export type KeyRegion = {
  readonly black: boolean;
  readonly bar: readonly Point[];
  readonly centre: Point;
  readonly area: number;
};

type Rows = Map<number, [number, number]>;

/** The 4-connected regions of white and of black pixels, each as its rows' extents; the
 * boundary class keeps two white keys that touch from reading as one. */
export function regionsOf(
  classes: Uint8Array,
): { readonly black: boolean; readonly rows: Rows; readonly size: number }[] {
  const seen = new Uint8Array(classes.length);
  const found: { black: boolean; rows: Rows; size: number }[] = [];
  const stack: number[] = [];
  for (let start = 0; start < classes.length; start += 1) {
    const label = classes[start];
    if (seen[start] || (label !== WHITE && label !== BLACK)) {
      continue;
    }
    const rows: Rows = new Map();
    let size = 0;
    stack.push(start);
    seen[start] = 1;
    while (stack.length > 0) {
      const at = stack.pop() ?? 0;
      const x = at % CROP_WIDTH;
      const y = (at - x) / CROP_WIDTH;
      size += 1;
      const row = rows.get(y);
      rows.set(y, row ? [Math.min(row[0], x), Math.max(row[1], x)] : [x, x]);
      for (const n of [
        x > 0 ? at - 1 : -1,
        x < CROP_WIDTH - 1 ? at + 1 : -1,
        y > 0 ? at - CROP_WIDTH : -1,
        y < CROP_HEIGHT - 1 ? at + CROP_WIDTH : -1,
      ]) {
        if (n >= 0 && !seen[n] && classes[n] === label) {
          seen[n] = 1;
          stack.push(n);
        }
      }
    }
    if (size >= LEAST_KEY_PX) {
      found.push({ black: label === BLACK, rows, size });
    }
  }
  return found;
}

function distanceToLine(p: Point, a: Point, b: Point): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const length = Math.hypot(dx, dy);
  return length === 0
    ? Math.hypot(p.x - a.x, p.y - a.y)
    : Math.abs(dy * p.x - dx * p.y + b.x * a.y - b.y * a.x) / length;
}

/** Douglas-Peucker: the fewest outline points that stay within `tolerance` of every original. */
export function simplify(points: readonly Point[], tolerance: number): Point[] {
  if (points.length <= 2) {
    return [...points];
  }
  let worst = 0;
  let at = 0;
  for (let i = 1; i < points.length - 1; i += 1) {
    const distance = distanceToLine(
      points[i],
      points[0],
      points[points.length - 1],
    );
    if (distance > worst) {
      worst = distance;
      at = i;
    }
  }
  if (worst <= tolerance) {
    return [points[0], points[points.length - 1]];
  }
  return [
    ...simplify(points.slice(0, at + 1), tolerance).slice(0, -1),
    ...simplify(points.slice(at), tolerance),
  ];
}

/** A region's outline in crop pixels: down its left edge row by row and back up its right, so
 * a white key keeps the notches the black keys cut into it. */
export function rowsOutline(rows: Rows): Point[] {
  const ys = [...rows.keys()].sort((a, b) => a - b);
  const left = ys.map((y) => ({ x: (rows.get(y)?.[0] ?? 0) - 0.5, y }));
  const right = ys
    .map((y) => ({ x: (rows.get(y)?.[1] ?? 0) + 0.5, y }))
    .reverse();
  const top = ys[0] - 0.5;
  const bottom = (ys.at(-1) ?? 0) + 0.5;
  return simplify(
    [
      { x: left[0].x, y: top },
      ...left,
      { x: left.at(-1)?.x ?? 0, y: bottom },
      { x: right[0].x, y: bottom },
      ...right,
      { x: right.at(-1)?.x ?? 0, y: top },
    ],
    OUTLINE_TOLERANCE_PX,
  );
}

function polygonArea(points: readonly Point[]): number {
  let twice = 0;
  points.forEach((p, i) => {
    const q = points[(i + 1) % points.length];
    twice += p.x * q.y - q.x * p.y;
  });
  return Math.abs(twice) / 2;
}

function inside(p: Point, polygon: readonly Point[]): boolean {
  let within = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i, i += 1) {
    const a = polygon[i];
    const b = polygon[j];
    if (
      a.y > p.y !== b.y > p.y &&
      p.x < ((b.x - a.x) * (p.y - a.y)) / (b.y - a.y) + a.x
    ) {
      within = !within;
    }
  }
  return within;
}

/** The convex hull of `points`, by the monotone chain. */
export function convexHull(points: readonly Point[]): Point[] {
  const sorted = [...points].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o: Point, a: Point, b: Point): number =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const half = (list: readonly Point[]): Point[] => {
    const out: Point[] = [];
    for (const p of list) {
      while (
        out.length >= 2 &&
        cross(out[out.length - 2], out[out.length - 1], p) <= 0
      ) {
        out.pop();
      }
      out.push(p);
    }
    return out.slice(0, -1);
  };
  return [...half(sorted), ...half([...sorted].reverse())];
}

/** A segmented key must cover at least, and at most, this share of its template key's area to
 * stand in for it; outside it the segmenter merged, split or lost the key. */
const SNAP_LEAST_AREA = 0.45;
const SNAP_MOST_AREA = 1.6;

/** Each template key takes the outline of the segmented key of its colour whose centre falls
 * inside it and whose size is close to its own, so keys follow the camera's real shapes where
 * the segmenter saw them and keep the template's shape where it did not. A region stands in for
 * one key at most. */
export function snapKeys<
  K extends { readonly black: boolean; readonly bar: readonly Point[] },
>(keys: readonly K[], regions: readonly KeyRegion[]): K[] {
  const used = new Set<KeyRegion>();
  return keys.map((key) => {
    const area = polygonArea(key.bar);
    const match = regions.find(
      (region) =>
        !used.has(region) &&
        region.black === key.black &&
        region.area >= area * SNAP_LEAST_AREA &&
        region.area <= area * SNAP_MOST_AREA &&
        inside(region.centre, key.bar),
    );
    if (match === undefined) {
      return key;
    }
    used.add(match);
    return { ...key, bar: match.bar };
  });
}

/** The key regions of a class map, as frame points, and their areas in the same units. */
export function keyRegions(
  classes: Uint8Array,
  toFrame: (x: number, y: number) => Point,
): KeyRegion[] {
  return regionsOf(classes).map((region) => {
    const bar = rowsOutline(region.rows).map((p) => toFrame(p.x, p.y));
    let sumX = 0;
    let sumY = 0;
    for (const [y, [from, to]] of region.rows) {
      const width = to - from + 1;
      sumX += ((from + to) / 2) * width;
      sumY += y * width;
    }
    return {
      black: region.black,
      bar,
      centre: toFrame(sumX / region.size, sumY / region.size),
      area: polygonArea(bar),
    };
  });
}

export async function createKeySegmenter(
  assets: RuntimeAssets,
  url: string = KEYSEG_URL,
): Promise<KeySegmenter> {
  const worker = new Worker(new URL("./keyseg-worker.ts", import.meta.url), {
    type: "module",
  });
  const pending = new Map<
    number,
    (reply: Extract<SegmentReply, { kind: "segmented" }>) => void
  >();
  let nextId = 0;
  const ready = new Promise<{
    readonly backend: string;
    readonly gpuFailure: string | null;
  }>((resolve, reject) => {
    worker.onmessage = (event: MessageEvent<SegmentReply>) => {
      const reply = event.data;
      if (reply.kind === "ready") {
        resolve(reply);
      } else if (reply.kind === "failed") {
        reject(new Error(reply.reason));
      } else {
        pending.get(reply.id)?.(reply);
        pending.delete(reply.id);
      }
    };
  });
  worker.postMessage({
    kind: "init",
    url: new URL(url, location.href).href,
    wasm: new URL(assets.ortGpuWasm, location.href).href,
  } satisfies SegmentRequest);
  const { backend, gpuFailure } = await ready;
  return {
    backend,
    gpuFailure,
    segment: async (frame, size, quad) => {
      if (quad.length < 4 || size.width === 0 || size.height === 0) {
        return { outline: null, regions: [] };
      }
      const crop = cropFor(
        quad
          .slice(0, 4)
          .map((p) => ({ x: p.x * size.width, y: p.y * size.height })),
      );
      const bitmap = await createImageBitmap(
        "data" in frame
          ? new ImageData(
              new Uint8ClampedArray(frame.data),
              frame.width,
              frame.height,
            )
          : frame,
      );
      nextId += 1;
      const id = nextId;
      const reply = await new Promise<
        Extract<SegmentReply, { kind: "segmented" }>
      >((resolve) => {
        pending.set(id, resolve);
        worker.postMessage(
          { kind: "segment", id, frame: bitmap, crop } satisfies SegmentRequest,
          [bitmap],
        );
      });
      const fraction = (p: Point): Point => ({
        x: p.x / size.width,
        y: p.y / size.height,
      });
      const areaScale = size.width * size.height;
      return {
        outline: reply.corners?.map(fraction) ?? null,
        regions: reply.regions.map((region) => ({
          black: region.black,
          bar: region.bar.map(fraction),
          centre: fraction(region.centre),
          area: region.area / areaScale,
        })),
      };
    },
  };
}
