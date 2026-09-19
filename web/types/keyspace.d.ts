import type { Point } from "./homography";
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
export declare const spanInKeys: number;
export type KeybedSpace = {
    /** Where a point of the keys lands in the frame: `across` runs from the far
     * edge at 0, where the black keys end, to the player's edge at 1. */
    readonly onKeys: (along: number, across: number) => Point | null;
};
export declare function keybedSpace(keybed: Keybed, frame: Size): KeybedSpace | null;
/** How many white keys a board carries, which is what its span is measured in. */
export declare function whiteKeysOf(range: PitchRange): number;
/** Where a key sits across the keybed, as a share of the corners it was
 * found in. */
export declare function keyBand(pitch: number, board: Board): {
    readonly from: number;
    readonly to: number;
};
/** The board a keybed carries when its corners are taken to be its ends,
 * which is what a range alone can say. */
export declare function boardOf(range: PitchRange): Board;
/** Four corners of a key's face, in image points. */
export type Bar = readonly [Point, Point, Point, Point];
/** The face of one key, flat on the instrument. */
export declare function keyFace(space: KeybedSpace, pitch: number, board: Board): Bar | null;
/** Every pitch the board carries, low to high. */
export declare function keysOf(range: PitchRange): readonly number[];
