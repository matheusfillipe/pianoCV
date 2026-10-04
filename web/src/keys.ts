const WHITE_PC_INDEX = [0, -1, 1, -1, 2, 3, -1, 4, -1, 5, -1, 6];
const BLACK_OFFSETS = [-1, 0.6, -1, 1.75, -1, -1, 3.6, -1, 4.63, -1, 5.66, -1];
const BLACK_WIDTH = 0.58;

export interface KeyUnits {
  from: number;
  to: number;
}

export function isBlack(pitch: number): boolean {
  return BLACK_OFFSETS[pitch % 12] >= 0;
}

export function whiteIndex(pitch: number): number {
  return 7 * Math.floor(pitch / 12) - 12 + WHITE_PC_INDEX[pitch % 12];
}

/** Where a key sits on any keyboard, in white-key widths from the same origin
 * `whiteIndex` counts from. The black keys carry the offsets a real instrument
 * has, where a black key straddles the join between two whites rather than
 * sitting over the middle of it. */
export function keyUnits(pitch: number): KeyUnits {
  const offset = BLACK_OFFSETS[pitch % 12];
  if (offset >= 0) {
    const from = 7 * Math.floor(pitch / 12) - 12 + offset;
    return { from, to: from + BLACK_WIDTH };
  }
  const from = whiteIndex(pitch);
  return { from, to: from + 1 };
}
