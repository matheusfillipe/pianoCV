import type { Point } from "./homography";
import { type KeyNetFit } from "./keycore";
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
export declare function keyOutlines(fit: KeyNetFit): KeyOutline[];
