import type { Point } from "./homography";
import type { Size } from "./keyspace";
declare global {
    interface Window {
        kvtFollowMs?: number[];
    }
}
export interface GrayFrame {
    readonly data: Float32Array;
    readonly width: number;
    readonly height: number;
}
export interface Similarity {
    readonly scale: number;
    readonly rotation: number;
    readonly tx: number;
    readonly ty: number;
}
export interface MotionEstimate {
    readonly transform: Similarity;
    /** Where every point that still matched anything landed, so the caller can
     * keep tracking them next frame without re-picking features. */
    readonly trackedPoints: readonly Point[];
    readonly inliers: number;
}
export declare function applySimilarity(s: Similarity, p: Point): Point;
/** Least-squares similarity (uniform scale, rotation, translation) between two point sets,
 * the closed-form solution to minimizing total squared point-to-point distance. */
export declare function fitSimilarity(from: readonly Point[], to: readonly Point[]): Similarity | null;
/** How the picture moved between two frames, read off a handful of tracked points, or null when
 * too few of them still match anything or the motion they agree on does not fit them well. */
export declare function estimateMotion(source: GrayFrame, points: readonly Point[], target: GrayFrame): MotionEstimate | null;
/** High-contrast points inside the quad, one per grid cell so they spread across it rather than
 * clumping on whichever edge is sharpest; a keybed's black-key edges dominate the picks. */
export declare function pickFeatures(frame: GrayFrame, quad: readonly Point[]): Point[];
export interface Follower {
    /** Anchors the follower to the tracker's own corners, discarding whatever it had estimated
     * before. Call this whenever the tracker (re)confirms the keybed. */
    readonly reset: (quad: readonly Point[], frame: CanvasImageSource, size: Size) => void;
    /** The followed quad for this frame, in the same frame-fraction coordinates as reset()'s
     * quad. Returns null until reset() has been called at least once. */
    readonly update: (frame: CanvasImageSource, size: Size) => readonly Point[] | null;
}
export declare function createFollower(): Follower;
