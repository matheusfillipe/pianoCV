import { type Point } from "./homography";
import type { Size } from "./keyspace";
import { type Bar } from "./keyspace";
declare global {
    interface Window {
        kvtKeyStrip?: {
            width: number;
            height: number;
            data: Uint8ClampedArray;
        };
        kvtKeyRead?: KeyRead;
    }
}
export declare const STRIP_WIDTH = 1200;
export declare const STRIP_HEIGHT = 150;
export type SourceImage = {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
};
export type Strip = {
    readonly width: number;
    readonly height: number;
    readonly data: Uint8ClampedArray;
};
export type DetectedKey = {
    readonly bar: Bar;
    readonly black: boolean;
};
export type KeyRead = {
    readonly kind: "read";
    readonly whiteKeys: number;
    readonly totalKeys: number;
    /** The white key the strip's left edge starts on, C to B. */
    readonly phase: string;
    readonly confidence: number;
    readonly keyAt: (index: number) => DetectedKey | null;
} | {
    readonly kind: "unsure";
    readonly reason: string;
    readonly confidence: number;
};
/** Warps the held quad flat: the far edge, where the black keys end, lands at the top, and the
 * player's edge lands at the bottom, so the rest of this module can work in undistorted pixels. */
export declare function rectifyStrip(source: SourceImage, quad: readonly Point[], width?: number, height?: number): Strip | null;
/** Maps a detection's keys, given in strip pixels, into the current frame through the same
 * homography a fresh rectify would use, so drawing tracks the followed quad every frame without
 * rebuilding the strip. */
export declare function projectKeys(read: KeyRead, quad: readonly Point[]): DetectedKey[];
/** Where the keys begin inside a rectified strip, as a fraction of its depth: the case is dark
 * and flat, the keys alternate black and white, so the first row bright and varied enough is
 * where the case ends. Null when the strip carries too little evidence to trust: no such row,
 * an implausible fraction, too few rows read as keys, or the keys not reaching the near edge. */
export declare function measureFarEdge(strip: Strip): number | null;
/** Moves the far edge (corners 0 and 1) toward the near edge (corners 2 and 3) by a fraction of
 * the depth, in the plane the quad is already drawn in. The near edge never moves: only the far
 * edge was ever measured onto the case instead of the keys. */
export declare function trimFarEdge(quad: readonly Point[], fraction: number): Point[];
/** Finds the keys in a rectified strip: the white-key separator lines near the player's edge and
 * the black-key runs near the far edge, walked and refined rather than fit to one assumed grid,
 * so a leftover lens warp does not have to be modelled to be tolerated. */
export declare function detectKeys(strip: Strip): KeyRead;
export type KeyReader = {
    readonly last: () => KeyRead | null;
    /** The far-edge trim this reader currently trusts, as a fraction of depth, or 0 before any
     * try has found evidence for one. */
    readonly fraction: () => number;
    readonly look: (frame: CanvasImageSource, quad: readonly Point[], size: Size, now: number) => void;
    readonly moved: (now: number) => void;
};
export declare function createKeyReader(): KeyReader;
