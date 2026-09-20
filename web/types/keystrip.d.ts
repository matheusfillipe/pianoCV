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
/** Finds the keys in a rectified strip: the white-key separator lines near the player's edge and
 * the black-key runs near the far edge, walked and refined rather than fit to one assumed grid,
 * so a leftover lens warp does not have to be modelled to be tolerated. */
export declare function detectKeys(strip: Strip): KeyRead;
export type KeyReader = {
    readonly last: () => KeyRead | null;
    readonly look: (frame: CanvasImageSource, quad: readonly Point[], size: Size, now: number) => void;
    readonly moved: (now: number) => void;
};
export declare function createKeyReader(): KeyReader;
