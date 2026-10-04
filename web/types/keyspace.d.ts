import type { Point } from "./homography";
/** The keyboard in front of the camera, as the MIDI pitches of its end keys. */
export type PitchRange = {
    readonly lowest: number;
    readonly highest: number;
};
export type Size = {
    readonly width: number;
    readonly height: number;
};
/** How many white keys a board carries, which is what its span is measured in. */
export declare function whiteKeysOf(range: PitchRange): number;
/** Four corners of a key's face, in image points. */
export type Bar = readonly [Point, Point, Point, Point];
