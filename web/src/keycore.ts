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

/** A point in the keyboard's own space, in white-key widths: `x` along the board from its first
 * key, `depth` from the keybed's far edge towards the player, `height` up off the white keys. */
export type SpacePoint = {
  readonly x: number;
  readonly depth: number;
  readonly height: number;
};

/** The keyboard's space as the camera sees it. */
export type KeySpace = {
  /** Where a point lands in the frame, in fractions, or null where it sits on the lens or behind
   * it. */
  readonly project: (point: SpacePoint) => Point | null;
  /** Where the camera stands, or null for a view so far off that its rays are parallel. */
  readonly camera: SpacePoint | null;
  /** White-key widths from the keybed's far edge to the player's edge. */
  readonly keybedDepth: number;
};

type SpaceJson = {
  readonly projection: readonly number[];
  readonly camera: SpacePoint | null;
  readonly keybedDepth: number;
  readonly keybedDistance: number;
  readonly nearestShare: number;
};

function keySpaceOf(space: SpaceJson): KeySpace {
  const m = space.projection;
  return {
    camera: space.camera,
    keybedDepth: space.keybedDepth,
    project: ({ x, depth, height }) => {
      const row = (r: number): number =>
        m[4 * r] * x +
        m[4 * r + 1] * depth +
        m[4 * r + 2] * height +
        m[4 * r + 3];
      const s = row(2);
      return s / space.keybedDistance >= space.nearestShare
        ? { x: row(0) / s, y: row(1) / s }
        : null;
    },
  };
}

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

export type KeyOutline = {
  readonly black: boolean;
  /** Semitones above the board's first white key. */
  readonly semitone: number;
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

  space(): KeySpace | null {
    const space: SpaceJson | null = JSON.parse(this.session.space());
    return space === null ? null : keySpaceOf(space);
  }
}
