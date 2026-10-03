import type { RuntimeAssets } from "./assets";
import { openModel } from "./gpu";
import type { Point } from "./homography";
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

/** The crop around `quad`, given in frame pixels, the same one the model was trained on. `width`
 * and `height` default to keyseg's own crop size; KeyNet's track mode passes its own, smaller
 * crop, at the same margins. */
export function cropFor(
  quad: readonly Point[],
  width: number = CROP_WIDTH,
  height: number = CROP_HEIGHT,
): Crop {
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
  const scale = Math.max(alongExtent / width, acrossExtent / height, 1e-6);
  return { centre, along, across, scale };
}

export function cropToFrame(
  crop: Crop,
  x: number,
  y: number,
  width: number = CROP_WIDTH,
  height: number = CROP_HEIGHT,
): Point {
  const u = (x - width / 2) * crop.scale;
  const v = (y - height / 2) * crop.scale;
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

/** How far an outline point may move onto the segmented outline, as a share of the key's width. */
const SNAP_REACH = 0.4;
/** How many points a drawn key's outline is carried as. */
const OUTLINE_POINTS = 40;
/** How many neighbours either side along the same edge we average a point's move over, so the
 * pixel steps of a segmented outline come out smooth. */
const SMOOTH_NEIGHBOURS = 2;
/** How far a drawn outline may stray from its points once we cut it to straight runs, as a share
 * of the frame's height. */
const DRAWN_TOLERANCE = 0.0012;
/** A black key's edge with at least this many points is refitted without its outliers, and a
 * point further off the first fit than this many times the typical one is an outlier. */
const EDGE_REFIT_LEAST = 4;
const EDGE_OUTLIER = 2.5;
/** Two edge lines closer to parallel than this, as the sine of their angle, never meet at a
 * corner we would draw. */
const EDGE_PARALLEL = 0.15;
/** How far a black key's corner may move to where its edges meet, as a share of its width. */
const CORNER_REACH = 0.5;
/** How far a key's held outline moves towards each new segmentation's. */
const SNAP_EASE = 0.15;

export function nearestOnOutline(p: Point, outline: readonly Point[]): Point {
  let best = p;
  let bestDistance = Infinity;
  outline.forEach((a, i) => {
    const b = outline[(i + 1) % outline.length];
    const dx = b.x - a.x;
    const dy = b.y - a.y;
    const length = dx * dx + dy * dy;
    const t =
      length === 0
        ? 0
        : Math.max(
            0,
            Math.min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length),
          );
    const q = { x: a.x + dx * t, y: a.y + dy * t };
    const distance = Math.hypot(q.x - p.x, q.y - p.y);
    if (distance < bestDistance) {
      best = q;
      bestDistance = distance;
    }
  });
  return best;
}

/** A key's width, as the short side of the rectangle with its perimeter and area, so it holds
 * however many points the outline is carried as. */
function keyWidth(bar: readonly Point[]): number {
  const half =
    bar.reduce((sum, a, i) => {
      const b = bar[(i + 1) % bar.length];
      return sum + Math.hypot(b.x - a.x, b.y - a.y);
    }, 0) / 2;
  return (
    (half - Math.sqrt(Math.max(0, half * half - 4 * polygonArea(bar)))) / 2
  );
}

function signedArea(points: readonly Point[]): number {
  let twice = 0;
  points.forEach((p, i) => {
    const q = points[(i + 1) % points.length];
    twice += p.x * q.y - q.x * p.y;
  });
  return twice / 2;
}

/** The outline of a black key's top and front faces together, starting at the top's first
 * corner and wound the same way, so each point of it is the same spot of the key every frame. */
export function keyHull(
  top: readonly Point[],
  front: readonly Point[],
): Point[] {
  const hull = convexHull([...top, ...front]);
  const wound =
    Math.sign(signedArea(hull)) === Math.sign(signedArea(top))
      ? hull
      : [...hull].reverse();
  let start = 0;
  wound.forEach((p, i) => {
    if (
      Math.hypot(p.x - top[0].x, p.y - top[0].y) <
      Math.hypot(wound[start].x - top[0].x, wound[start].y - top[0].y)
    ) {
      start = i;
    }
  });
  return [...wound.slice(start), ...wound.slice(0, start)];
}

/** `count` points around a closed outline from its first corner: every corner, and the rest
 * spread over the edges by their length. `edges` holds the edge each point lies on. */
export function sampleOutline(
  bar: readonly Point[],
  count: number,
): { points: Point[]; edges: number[] } {
  const lengths = bar.map((p, i) => {
    const q = bar[(i + 1) % bar.length];
    return Math.hypot(q.x - p.x, q.y - p.y);
  });
  const perimeter = lengths.reduce((sum, l) => sum + l, 0);
  const share = (l: number): number =>
    perimeter === 0 ? 1 : 1 + ((count - bar.length) * l) / perimeter;
  const shares = lengths.map((l) => Math.floor(share(l)));
  // whole points go to the edges by their length, and the few left over to the longest edges
  const byLength = lengths
    .map((l, i) => ({ i, rest: share(l) - Math.floor(share(l)) }))
    .sort((x, y) => y.rest - x.rest || x.i - y.i);
  for (let k = 0; shares.reduce((a, b) => a + b, 0) < count; k += 1) {
    shares[byLength[k % byLength.length].i] += 1;
  }
  const points: Point[] = [];
  const edges: number[] = [];
  bar.forEach((p, edge) => {
    const q = bar[(edge + 1) % bar.length];
    for (let j = 0; j < shares[edge]; j += 1) {
      const t = j / shares[edge];
      points.push({ x: p.x + (q.x - p.x) * t, y: p.y + (q.y - p.y) * t });
      edges.push(edge);
    }
  });
  return { points, edges };
}

export function outlinePoints(bar: readonly Point[], count: number): Point[] {
  return sampleOutline(bar, count).points;
}

/** Each move averaged with its neighbours on the same edge, so an edge comes out smooth and a
 * corner stays sharp. */
function smoothAlongEdges(
  moves: readonly Point[],
  edges: readonly number[] | undefined,
): Point[] {
  const n = moves.length;
  return moves.map((_, i) => {
    let x = 0;
    let y = 0;
    let taken = 0;
    for (let k = -SMOOTH_NEIGHBOURS; k <= SMOOTH_NEIGHBOURS; k += 1) {
      const j = (i + k + n) % n;
      if (edges !== undefined && edges[j] !== edges[i]) {
        continue;
      }
      x += moves[j].x;
      y += moves[j].y;
      taken += 1;
    }
    return { x: x / taken, y: y / taken };
  });
}

/** Each template key moves onto the segmented key of its colour whose centre falls inside it and
 * whose size is close to its own: every point of its outline moves onto the nearest point of the
 * segmented outline, as far as its reach, and the moves are smoothed along each edge. A black
 * key carried with its `edges` then gets straight edges and sharp corners. A key keeps the
 * template's outline wherever the segmented one is out of reach, comes back with the points it
 * was given, and comes back unchanged when nothing matched it. `aspect` is the frame's width over
 * its height, since points are frame fractions. A region stands in for one key at most. */
export function snapKeys<
  K extends {
    readonly black: boolean;
    readonly bar: readonly Point[];
    readonly edges?: readonly number[];
  },
>(keys: readonly K[], regions: readonly KeyRegion[], aspect = 1): K[] {
  const used = new Set<KeyRegion>();
  const square = (p: Point): Point => ({ x: p.x * aspect, y: p.y });
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
    const bar = key.bar.map(square);
    const reach = SNAP_REACH * keyWidth(bar);
    const outline = match.bar.map(square);
    const moves = smoothAlongEdges(
      bar.map((p) => {
        const q = nearestOnOutline(p, outline);
        return Math.hypot(q.x - p.x, q.y - p.y) <= reach
          ? { x: q.x - p.x, y: q.y - p.y }
          : { x: 0, y: 0 };
      }),
      key.edges,
    );
    const moved = bar.map((p, i) => ({
      x: p.x + moves[i].x,
      y: p.y + moves[i].y,
    }));
    const shaped =
      key.black && key.edges
        ? straightenEdges(moved, key.edges, keyWidth(bar))
        : moved;
    return {
      ...key,
      bar: shaped.map((p) => ({ x: p.x / aspect, y: p.y })),
    };
  });
}

type FittedLine = { readonly at: Point; readonly along: Point };

/** The straight line through the points, fitted again without the ones far off the first fit,
 * the way a shadow the segmenter took for key bulges off a black key's edge. */
function edgeLine(points: readonly Point[]): FittedLine | null {
  const through = (pts: readonly Point[]): FittedLine | null => {
    if (pts.length < 2) {
      return null;
    }
    const at = {
      x: pts.reduce((sum, p) => sum + p.x, 0) / pts.length,
      y: pts.reduce((sum, p) => sum + p.y, 0) / pts.length,
    };
    let xx = 0;
    let xy = 0;
    let yy = 0;
    for (const p of pts) {
      xx += (p.x - at.x) ** 2;
      xy += (p.x - at.x) * (p.y - at.y);
      yy += (p.y - at.y) ** 2;
    }
    const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
    return { at, along: { x: Math.cos(angle), y: Math.sin(angle) } };
  };
  const first = through(points);
  if (first === null || points.length < EDGE_REFIT_LEAST) {
    return first;
  }
  const off = points.map((p) => offLine(p, first));
  const typical = [...off].sort((a, b) => a - b)[Math.floor(off.length / 2)];
  const kept = points.filter((_, i) => off[i] <= typical * EDGE_OUTLIER);
  return kept.length >= 2 ? through(kept) : first;
}

function offLine(p: Point, line: FittedLine): number {
  return Math.abs(
    (p.x - line.at.x) * line.along.y - (p.y - line.at.y) * line.along.x,
  );
}

function ontoLine(p: Point, line: FittedLine): Point {
  const t = (p.x - line.at.x) * line.along.x + (p.y - line.at.y) * line.along.y;
  return { x: line.at.x + line.along.x * t, y: line.at.y + line.along.y * t };
}

function meet(a: FittedLine, b: FittedLine): Point | null {
  const denominator = a.along.x * b.along.y - a.along.y * b.along.x;
  if (Math.abs(denominator) < EDGE_PARALLEL) {
    return null;
  }
  const t =
    ((b.at.x - a.at.x) * b.along.y - (b.at.y - a.at.y) * b.along.x) /
    denominator;
  return { x: a.at.x + a.along.x * t, y: a.at.y + a.along.y * t };
}

/** Each edge of the outline laid on its own straight line and each corner where its two edges'
 * lines meet, so a key keeps straight sides and sharp corners wherever its edges lead. */
function straightenEdges(
  points: readonly Point[],
  edges: readonly number[],
  width: number,
): Point[] {
  const reach = CORNER_REACH * width;
  const count = Math.max(...edges) + 1;
  const lines = Array.from({ length: count }, (_, e) => {
    const on = points.filter((_, i) => edges[i] === e);
    const end = points[(edges.lastIndexOf(e) + 1) % points.length];
    return edgeLine([...on, end]);
  });
  return points.map((p, i) => {
    const line = lines[edges[i]];
    if (line === null) {
      return p;
    }
    const corner = i === 0 || edges[i - 1] !== edges[i];
    const before = lines[(edges[i] - 1 + count) % count];
    const met = corner && before !== null ? meet(before, line) : null;
    return met !== null && Math.hypot(met.x - p.x, met.y - p.y) <= reach
      ? met
      : ontoLine(p, line);
  });
}

export type KeySnap = {
  /** The keys drawn as smooth outlines. A black key follows its segmented key: its outline is
   * carried as evenly spread points, each fresh segmentation pulls its held moves part of the way
   * to what it shows, and a black key the segmenter did not find this time keeps the moves it
   * had, so it never flips between its template and its segmented shape frame to frame. A white
   * key keeps the template's outline, whose sides it shares with its white neighbours; the black
   * keys over it are cut out of it where it is drawn. */
  readonly apply: <
    K extends {
      readonly black: boolean;
      readonly semitone: number;
      readonly bar: readonly Point[];
    },
  >(
    keys: readonly K[],
    regions: readonly KeyRegion[],
    aspect: number,
  ) => K[];
  readonly reset: () => void;
};

export function createKeySnap(): KeySnap {
  let held = new Map<string, Point[]>();
  let seen: readonly KeyRegion[] | null = null;
  return {
    reset: () => {
      held = new Map();
      seen = null;
    },
    apply: (keys, regions, aspect) => {
      const dense = keys.map((key) => {
        const { points, edges } = sampleOutline(key.bar, OUTLINE_POINTS);
        return { ...key, bar: points, edges };
      });
      const id = (key: { black: boolean; semitone: number }): string =>
        `${key.black}${key.semitone}`;
      if (regions !== seen) {
        seen = regions;
        const blacks = dense.filter((key) => key.black);
        const snapped = snapKeys(blacks, regions, aspect);
        blacks.forEach((key, i) => {
          if (snapped[i] === key) {
            return;
          }
          const moves = snapped[i].bar.map((p, j) => ({
            x: p.x - key.bar[j].x,
            y: p.y - key.bar[j].y,
          }));
          const before = held.get(id(key));
          held.set(
            id(key),
            before === undefined
              ? moves
              : before.map((h, j) => ({
                  x: h.x + (moves[j].x - h.x) * SNAP_EASE,
                  y: h.y + (moves[j].y - h.y) * SNAP_EASE,
                })),
          );
        });
      }
      const square = (p: Point): Point => ({ x: p.x * aspect, y: p.y });
      const moved = dense.map((key) => {
        const moves = held.get(id(key));
        return {
          ...key,
          bar: (moves
            ? key.bar.map((p, j) => ({
                x: p.x + moves[j].x,
                y: p.y + moves[j].y,
              }))
            : key.bar
          ).map(square),
        };
      });
      return moved.map((key) => ({
        ...key,
        bar: simplify([...key.bar, key.bar[0]], DRAWN_TOLERANCE)
          .slice(0, -1)
          .map((p) => ({ x: p.x / aspect, y: p.y })),
      }));
    },
  };
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
  const model = await openModel(assets, url);
  return {
    backend: model.backend,
    gpuFailure: model.gpuFailure,
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
      const reply = await model.worker.ask(
        { kind: "segment", url: model.url, frame: bitmap, crop },
        [bitmap],
      );
      if (reply.kind !== "segmented") {
        return { outline: null, regions: [] };
      }
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
