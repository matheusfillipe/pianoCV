import type { Point } from "./homography";
import { isBlack, keyUnits } from "./keys";
import {
  keybedDepth,
  projectSpace,
  solvePose,
  spaceDepth,
  WHITE_KEY_COUNT,
} from "./pose";

/** The keyboard in front of the camera, as the MIDI pitches of its end keys. */
export type PitchRange = {
  readonly lowest: number;
  readonly highest: number;
};

/** The keyboard the camera is looking at, fitted to where it sits inside the
 * corners found for it. */
export type Board = PitchRange & {
  /** The white-key position, in `keyUnits`, of the keybed's own first corner. */
  readonly origin: number;
  readonly span: number;
};

export type Size = {
  readonly width: number;
  readonly height: number;
};

/** Corners as fractions of the frame, ordered so 0 to 1 runs along the keys
 * and 1 to 2 crosses the keybed's depth, with the player at the near edge. */
export type Keybed = {
  readonly quad: readonly Point[];
};

/** The keys the pose spans, which is what `along` is a share of. A board with
 * another key count is still this wide. */
export const spanInKeys = WHITE_KEY_COUNT;

/** Nearer than this a point is behind the lens or on top of it, with no
 * picture to be drawn in. */
const nearestDrawable = 1;

export type KeybedSpace = {
  /** Where a point of the keys lands in the frame: `across` runs from the far
   * edge at 0, where the black keys end, to the player's edge at 1. */
  readonly onKeys: (along: number, across: number) => Point | null;
};

/** The far edge orders first and the player's edge last, matching the pose's
 * own world frame. */
function poseCorners(keybed: Keybed, frame: Size): Point[] {
  return keybed.quad.map((corner) => ({
    x: corner.x * frame.width,
    y: corner.y * frame.height,
  }));
}

export function keybedSpace(keybed: Keybed, frame: Size): KeybedSpace | null {
  const corners = poseCorners(keybed, frame);
  if (corners.length < 4 || frame.width === 0 || frame.height === 0) {
    return null;
  }
  const pose = solvePose(corners, frame.width, frame.height);
  const project = (u: number, v: number, w: number): Point | null => {
    if (spaceDepth(pose, u, v, w) < nearestDrawable) {
      return null;
    }
    return projectSpace(pose, u, v, w, frame.width, frame.height);
  };
  return {
    onKeys: (along, across) =>
      project(along * WHITE_KEY_COUNT, across * keybedDepth(), 0),
  };
}

/** How many white keys a board carries, which is what its span is measured in. */
export function whiteKeysOf(range: PitchRange): number {
  return keyUnits(range.highest).to - keyUnits(range.lowest).from;
}

/** Where a key sits across the keybed, as a share of the corners it was
 * found in. */
export function keyBand(
  pitch: number,
  board: Board,
): { readonly from: number; readonly to: number } {
  const units = keyUnits(pitch);
  return {
    from: (units.from - board.origin) / board.span,
    to: (units.to - board.origin) / board.span,
  };
}

/** The board a keybed carries when its corners are taken to be its ends,
 * which is what a range alone can say. */
export function boardOf(range: PitchRange): Board {
  return {
    ...range,
    origin: keyUnits(range.lowest).from,
    span: whiteKeysOf(range),
  };
}

/** Four corners of a key's face, in image points. */
export type Bar = readonly [Point, Point, Point, Point];

/** How much of the keybed's depth a black key takes, measured from the far
 * edge where the black keys start. */
const blackKeyDepth = 0.62;

function quad(corners: readonly (Point | null)[]): Bar | null {
  const [a, b, c, d] = corners;
  if (
    a === undefined ||
    b === undefined ||
    c === undefined ||
    d === undefined ||
    a === null ||
    b === null ||
    c === null ||
    d === null
  ) {
    return null;
  }
  return [a, b, c, d];
}

/** The face of one key, flat on the instrument. */
export function keyFace(
  space: KeybedSpace,
  pitch: number,
  board: Board,
): Bar | null {
  const band = keyBand(pitch, board);
  const near = isBlack(pitch) ? blackKeyDepth : 1;
  return quad([
    space.onKeys(band.from, 0),
    space.onKeys(band.to, 0),
    space.onKeys(band.to, near),
    space.onKeys(band.from, near),
  ]);
}

/** Every pitch the board carries, low to high. */
export function keysOf(range: PitchRange): readonly number[] {
  const keys: number[] = [];
  for (let pitch = range.lowest; pitch <= range.highest; pitch += 1) {
    keys.push(pitch);
  }
  return keys;
}
