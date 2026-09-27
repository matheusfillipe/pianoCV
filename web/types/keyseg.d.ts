import type { RuntimeAssets } from "./assets";
import type { Point } from "./homography";
import type { Size } from "./keyspace";
import { type SourceImage } from "./keystrip";
export declare const KEYSEG_URL = "/keyseg.onnx";
export declare const CROP_WIDTH = 1024;
export declare const CROP_HEIGHT = 224;
/** Rotates and scales the frame so the keys run left to right with the player's edge at the
 * bottom, around a rough keybed quad; it never bends the frame, so keys keep their own shape. */
export type Crop = {
    readonly centre: Point;
    readonly along: Point;
    readonly across: Point;
    /** Frame pixels per crop pixel. */
    readonly scale: number;
};
export type Segmented = {
    /** The keybed outline the key pixels show, as frame fractions, or null when too few were
     * found to draw one. */
    readonly outline: Point[] | null;
    /** Every key the segmenter saw, with outlines in frame fractions. */
    readonly regions: readonly KeyRegion[];
};
export type KeySegmenter = {
    /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
    readonly backend: string;
    readonly gpuFailure: string | null;
    /** The keybed outline and the keys the segmenter sees around `quad`. */
    readonly segment: (frame: CanvasImageSource | SourceImage, size: Size, quad: readonly Point[]) => Promise<Segmented>;
};
/** The crop around `quad`, given in frame pixels, the same one the model was trained on. */
export declare function cropFor(quad: readonly Point[]): Crop;
export declare function cropToFrame(crop: Crop, x: number, y: number): Point;
/** Each pixel's most likely class, row by row. */
export declare function classesOf(probabilities: Float32Array): Uint8Array;
/** The keybed outline the key pixels show, in crop pixels, as far-left, far-right, near-right,
 * near-left: the far edge through each column's first key pixel, the near edge through its last
 * white one, and each end through each row's first and last key pixel. */
export declare function keyOutline(classes: Uint8Array): Point[] | null;
/** One key's pixels as the segmenter saw them: its outline and where its pixels sit. */
export type KeyRegion = {
    readonly black: boolean;
    readonly bar: readonly Point[];
    readonly centre: Point;
    readonly area: number;
};
type Rows = Map<number, [number, number]>;
/** The 4-connected regions of white and of black pixels, each as its rows' extents; the
 * boundary class keeps two white keys that touch from reading as one. */
export declare function regionsOf(classes: Uint8Array): {
    readonly black: boolean;
    readonly rows: Rows;
    readonly size: number;
}[];
/** Douglas-Peucker: the fewest outline points that stay within `tolerance` of every original. */
export declare function simplify(points: readonly Point[], tolerance: number): Point[];
/** A region's outline in crop pixels: down its left edge row by row and back up its right, so
 * a white key keeps the notches the black keys cut into it. */
export declare function rowsOutline(rows: Rows): Point[];
/** The convex hull of `points`, by the monotone chain. */
export declare function convexHull(points: readonly Point[]): Point[];
/** Each template key takes the outline of the segmented key of its colour whose centre falls
 * inside it and whose size is close to its own, so keys follow the camera's real shapes where
 * the segmenter saw them and keep the template's shape where it did not. A region stands in for
 * one key at most. */
export declare function snapKeys<K extends {
    readonly black: boolean;
    readonly bar: readonly Point[];
}>(keys: readonly K[], regions: readonly KeyRegion[]): K[];
/** The key regions of a class map, as frame points, and their areas in the same units. */
export declare function keyRegions(classes: Uint8Array, toFrame: (x: number, y: number) => Point): KeyRegion[];
export declare function createKeySegmenter(assets: RuntimeAssets, url?: string): Promise<KeySegmenter>;
export {};
