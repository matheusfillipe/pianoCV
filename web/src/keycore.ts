import type { Homography, Point } from "./homography";
import init, * as core from "./keycore-wasm/keycore.js";
import type { Bar, Size } from "./keyspace";

export type DetectedKey = {
  readonly bar: Bar;
  readonly black: boolean;
  /** Semitones above the board's first white key, which is what maps a MIDI note onto it. */
  readonly semitone: number;
};

export type ScoredPoint = Point & { readonly score: number };

export type KeyNetPeaks = {
  /** Back-low, back-high, front-high, front-low: the best peak per channel, or null where the
   * channel had nothing above threshold. */
  readonly corners: readonly (ScoredPoint | null)[];
  readonly gaps: readonly ScoredPoint[];
  readonly blackLow: readonly ScoredPoint[];
  readonly blackHigh: readonly ScoredPoint[];
  /** Where each black key's front face meets its top, low-pitch and high-pitch side. */
  readonly blackTopLow: readonly ScoredPoint[];
  readonly blackTopHigh: readonly ScoredPoint[];
  /** White-key gaps at the back edge, and the black keys' raised back-top corners, low-pitch and
   * high-pitch side. Empty for a model without the back channels. */
  readonly backGaps: readonly ScoredPoint[];
  readonly blackBackLow: readonly ScoredPoint[];
  readonly blackBackHigh: readonly ScoredPoint[];
};

export type KeyNetFit = {
  /** Maps the template plane (white-key units) onto the frame (fractions). */
  readonly homography: Homography;
  readonly quad: readonly [Point, Point, Point, Point];
  readonly whiteKeys: number;
  readonly phase: string;
  /** How the black keys' tops sit over the keybed in this view, null until their top corners
   * have been seen. */
  readonly lift: readonly [number, number, number] | null;
};

export type KeyboardTemplate = {
  /** Back-low, back-high, front-high, front-low, in white-key units: x runs 0 to `whiteKeys`
   * along the board, y runs 0 (back, under the black keys) to 1 (the player's edge). */
  readonly corners: readonly [Point, Point, Point, Point];
  /** Every white-key boundary at the front edge, low to high. */
  readonly gaps: readonly Point[];
  /** Every white-key boundary at the back edge, low to high. */
  readonly backGaps: readonly Point[];
  readonly blackLow: readonly Point[];
  readonly blackHigh: readonly Point[];
};

export type CropRequest = {
  readonly mode: "search" | "track";
  readonly width: number;
  readonly height: number;
  /** Maps a model input pixel to a frame fraction. */
  readonly matrix: Homography;
};

export type KeyNetStep = {
  readonly fit: KeyNetFit | null;
  /** Whether this step found the board, after one or more without it. */
  readonly acquired: boolean;
};

let loading: Promise<unknown> | null = null;

export function loadKeycore(): Promise<unknown> {
  loading ??= init();
  return loading;
}

export function keyboardTemplate(
  whiteKeys: number,
  phase: string,
): KeyboardTemplate {
  return JSON.parse(core.keyboard_template(whiteKeys, phase));
}

export function liftPoints(
  homography: Homography,
  lift: readonly [number, number, number],
  points: readonly Point[],
): Point[] {
  return JSON.parse(
    core.lift_points(
      new Float64Array(homography),
      new Float64Array(lift),
      JSON.stringify(points),
    ),
  );
}

export function refineHomography(
  pairs: readonly { readonly src: Point; readonly dst: Point }[],
): Homography | null {
  return JSON.parse(core.refine_homography(JSON.stringify(pairs)));
}

/** The keys to draw: flat on the keybed until the fit has a lift for the black keys' tops. */
export function keyNetFaces(fit: KeyNetFit): DetectedKey[] {
  return JSON.parse(
    core.key_net_faces(
      JSON.stringify(fit),
      new Float64Array(fit.lift ?? [0, 0, 0]),
    ),
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

/** The convex hull of `points`, by the monotone chain. */
function convexHull(points: readonly Point[]): Point[] {
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

export function decodeHeatmaps(
  heat: Float32Array,
  width: number,
  height: number,
  toFrame: Homography,
  offsets: Float32Array | null,
): KeyNetPeaks {
  return JSON.parse(
    core.decode_heatmaps(
      heat,
      width,
      height,
      new Float64Array(toFrame),
      offsets,
    ),
  );
}

export function prepareInput(
  rgba: Uint8ClampedArray,
  pixels: Size,
  crop: CropRequest,
): Float32Array {
  return core.prepare_input(
    new Uint8Array(rgba.buffer, rgba.byteOffset, rgba.byteLength),
    pixels.width,
    pixels.height,
    new Float64Array(crop.matrix),
    crop.width,
    crop.height,
  );
}

export class KeyNetLoop {
  private readonly session: core.KeyNetSession;

  constructor(rectified: boolean) {
    this.session = new core.KeyNetSession(rectified);
  }

  nextCrop(frame: Size): CropRequest {
    return JSON.parse(this.session.next_crop(frame.width, frame.height));
  }

  step(
    presence: number,
    peaks: KeyNetPeaks,
    frame: Size,
    nowMs: number,
  ): KeyNetStep {
    return JSON.parse(
      this.session.step(
        presence,
        JSON.stringify(peaks),
        frame.width,
        frame.height,
        nowMs,
      ),
    );
  }
}
