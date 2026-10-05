import type { Point } from "./homography";
import { type KeyNetFit } from "./keycore";
import type { Bar } from "./keyspace";
export type KeyOutline = {
    readonly black: boolean;
    /** Semitones above the board's first white key. */
    readonly semitone: number;
    readonly note: number;
    /** The key's outline in frame fractions, a black key's raised top and footprint together. */
    readonly bar: readonly Point[];
    /** The face the key shows on top, in frame fractions, far edge first: a white key's own face
     * and a black key's raised top. */
    readonly top: Bar;
};
/** Every key of the fitted board once, low to high, with its note and on-screen outline. */
export declare function keyOutlines(fit: KeyNetFit): KeyOutline[];
