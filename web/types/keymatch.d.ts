import type { RuntimeAssets } from "./assets";
import { type Point } from "./homography";
import { type KeyEvidence, type LinePoint, type Run, type SourceImage } from "./keystrip";
export declare const KEYMATCH_URL = "/keymatch.onnx";
export type Matched = {
    /** The outline the keys were read on: the given quad with its ends turned to run along the
     * white keys the matcher saw, as frame fractions. */
    readonly quad: readonly Point[];
    /** Where that outline's corners sit in the strip of the quad that was given, in strip pixels,
     * so the correction can follow the quad as it moves. */
    readonly outline: readonly Point[];
    readonly evidence: KeyEvidence;
};
export type KeyMatcher = {
    /** The keys the matcher sees on `quad`, once the quad is squared to the keys, or null when it
     * saw too little to stand in for the brightness rules. */
    readonly match: (source: SourceImage, quad: readonly Point[]) => Promise<Matched | null>;
};
/** Local maxima above the threshold, strongest first, none within `PEAK_SPACING` of a stronger. */
export declare function peaks(values: Float32Array): number[];
/** Pairs each black key's left edge with the first right edge after it, before the next left
 * edge, and as wide as a black key can be. */
export declare function pairRuns(lefts: readonly number[], rights: readonly number[], widest: number, narrowest?: number): Run[];
/** How far each white-key gap moves between the white row and the shallow row, as a line over
 * the strip. White keys lie flat, so this is the slant the quad's ends give every key line; we
 * follow the gaps from the start of the strip so a slant past half a key never pairs a gap with
 * its neighbour. */
export declare function measureSlant(dips: readonly number[], shallowDips: readonly number[]): LinePoint[] | null;
/** The corners, in strip pixels, of the outline whose ends at `left` and `right` run along the
 * white keys: every key line passes through its white-row gap and leans by the measured slant
 * per unit of depth. Key lines on a flat keybed meet at one vanishing point, which is the same
 * as their slant changing linearly along the strip, so one line fit carries the perspective. */
export declare function squaredOutline(moves: readonly LinePoint[], left?: number, right?: number): Point[];
export declare function createKeyMatcher(assets: RuntimeAssets, url?: string): Promise<KeyMatcher>;
