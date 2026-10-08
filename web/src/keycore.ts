import type { Homography, Point } from "./homography";
import init, * as core from "./keycore-wasm/keycore.js";
import type { Bar, Size } from "./keyspace";

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
  /** How much of the keybed's depth this board's black keys take, measured while it is
   * tracked. */
  readonly blackDepth: number;
  /** The bend the board was fitted under: the homography, quad and lift live in the frame
   * with it taken out, and `bendPoint` carries a point of them into the picture. */
  readonly lens: Lens;
};

/** How the camera's lens bends straight lines, as one radial term about the middle of the
 * frame: a point at `r` from the middle, measured so the frame's corners sit at 1, is pulled
 * in to `r / (1 + k r²)` once the bend is taken out. */
export type Lens = {
  readonly k: number;
  /** The frame's width over its height, so the bend is round in pixels. */
  readonly aspect: number;
};

function lensReach(lens: Lens): number {
  return Math.hypot(lens.aspect, 1) / 2;
}

/** Where a point of the picture, in frame fractions, lies once the lens's bend is taken out. */
export function straightenPoint(lens: Lens, p: Point): Point {
  const dx = (p.x - 0.5) * lens.aspect;
  const dy = p.y - 0.5;
  const r2 = (dx * dx + dy * dy) / lensReach(lens) ** 2;
  const scale = 1 / (1 + lens.k * r2);
  return { x: 0.5 + (dx * scale) / lens.aspect, y: 0.5 + dy * scale };
}

/** Where a straightened point lands in the picture the camera films, the inverse of
 * `straightenPoint`. */
export function bendPoint(lens: Lens, p: Point): Point {
  const reach = lensReach(lens);
  const dx = ((p.x - 0.5) * lens.aspect) / reach;
  const dy = (p.y - 0.5) / reach;
  const s = Math.hypot(dx, dy);
  const room = 1 - 4 * lens.k * s * s;
  if (s < 1e-12 || Math.abs(lens.k) < 1e-12 || room < 0) {
    return p;
  }
  const scale = (1 - Math.sqrt(room)) / (2 * lens.k * s) / s;
  return {
    x: 0.5 + (dx * scale * reach) / lens.aspect,
    y: 0.5 + dy * scale * reach,
  };
}

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

/** The template for a board whose black keys take `blackDepth` of the keybed: a fit's own
 * `blackDepth`, or a typical board's when none is given. */
export function keyboardTemplate(
  whiteKeys: number,
  phase: string,
  blackDepth?: number,
): KeyboardTemplate {
  return JSON.parse(core.keyboard_template(whiteKeys, phase, blackDepth));
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

export type KeyOutline = {
  readonly black: boolean;
  /** Semitones above the board's first white key. */
  readonly semitone: number;
  /** Where the key starts and ends along the board, in white keys from its first key. */
  readonly from: number;
  readonly to: number;
  /** The key's outline in frame fractions, a black key's raised top and footprint together. */
  readonly bar: readonly Point[];
  /** The face the key shows on top: a white key's own face and a black key's raised top. */
  readonly top: Bar;
};

/** Every key of the fitted board once, low to high, flat on the keybed until the fit has a lift. */
export function keyOutlines(fit: KeyNetFit): KeyOutline[] {
  return JSON.parse(core.key_outlines(JSON.stringify(fit)));
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
