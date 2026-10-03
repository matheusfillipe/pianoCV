import type { Homography, Point } from "./homography";
import init, * as core from "./keycore-wasm/keycore.js";
import type { Size } from "./keyspace";
import type { DetectedKey } from "./keystrip";

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
