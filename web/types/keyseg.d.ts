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
export declare function nearestOnOutline(p: Point, outline: readonly Point[]): Point;
/** The outline of a black key's top and front faces together, starting at the top's first
 * corner and wound the same way, so each point of it is the same spot of the key every frame. */
export declare function keyHull(top: readonly Point[], front: readonly Point[]): Point[];
/** `count` points around a closed outline from its first corner: every corner, and the rest
 * spread over the edges by their length. `edges` holds the edge each point lies on. */
export declare function sampleOutline(bar: readonly Point[], count: number): {
    points: Point[];
    edges: number[];
};
export declare function outlinePoints(bar: readonly Point[], count: number): Point[];
/** Each template key moves onto the segmented key of its colour whose centre falls inside it and
 * whose size is close to its own: every point of its outline moves onto the nearest point of the
 * segmented outline, as far as its reach, and the moves are smoothed along each edge. A black
 * key carried with its `edges` then gets straight edges and sharp corners. A key keeps the
 * template's outline wherever the segmented one is out of reach, comes back with the points it
 * was given, and comes back unchanged when nothing matched it. `aspect` is the frame's width over
 * its height, since points are frame fractions. A region stands in for one key at most. */
export declare function snapKeys<K extends {
    readonly black: boolean;
    readonly bar: readonly Point[];
    readonly edges?: readonly number[];
}>(keys: readonly K[], regions: readonly KeyRegion[], aspect?: number): K[];
export type KeySnap = {
    /** The keys drawn as smooth outlines. A black key follows its segmented key: its outline is
     * carried as evenly spread points, each fresh segmentation pulls its held moves part of the way
     * to what it shows, and a black key the segmenter did not find this time keeps the moves it
     * had, so it never flips between its template and its segmented shape frame to frame. A white
     * key keeps the template's outline, whose sides it shares with its white neighbours; the black
     * keys over it are cut out of it where it is drawn. */
    readonly apply: <K extends {
        readonly black: boolean;
        readonly semitone: number;
        readonly bar: readonly Point[];
    }>(keys: readonly K[], regions: readonly KeyRegion[], aspect: number) => K[];
    readonly reset: () => void;
};
export declare function createKeySnap(): KeySnap;
/** The key regions of a class map, as frame points, and their areas in the same units. */
export declare function keyRegions(classes: Uint8Array, toFrame: (x: number, y: number) => Point): KeyRegion[];
export declare function createKeySegmenter(assets: RuntimeAssets, url?: string): Promise<KeySegmenter>;
export {};
