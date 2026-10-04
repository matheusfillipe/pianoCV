import { lowestPitchFor } from "./effects";
import type { Point } from "./homography";
import { type KeyNetFit, keyHull, keyNetFaces } from "./keycore";

export type KeyOutline = {
  readonly black: boolean;
  /** Semitones above the board's first white key. */
  readonly semitone: number;
  /** The MIDI note, or null for a board whose lowest note is not known. */
  readonly note: number | null;
  /** The key's outline in frame fractions, a black key's raised top and footprint together. */
  readonly bar: readonly Point[];
};

/** Every key of the fitted board once, low to high, with its note and on-screen outline. */
export function keyOutlines(fit: KeyNetFit): KeyOutline[] {
  const lowest = lowestPitchFor(fit.whiteKeys, fit.phase);
  const faces = keyNetFaces(fit);
  const keys: KeyOutline[] = [];
  for (let i = 0; i < faces.length; i += 1) {
    const face = faces[i];
    const footprint = faces[i + 1];
    const merged =
      face.black && footprint?.black && footprint.semitone === face.semitone;
    keys.push({
      black: face.black,
      semitone: face.semitone,
      note: lowest === null ? null : lowest + face.semitone,
      bar: merged ? keyHull(face.bar, footprint.bar) : face.bar,
    });
    if (merged) {
      i += 1;
    }
  }
  return keys;
}
