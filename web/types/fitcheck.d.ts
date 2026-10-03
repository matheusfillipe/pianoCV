import { type KeyNetFit, type KeyNetPeaks } from "./keycore";
import { type KeyNetRunner } from "./keynetrunner";
/** A labelled frame's keypoints in frame pixels, low key to high, as `make lab-keynet-eval
 * ARGS="--points-out ..."` writes them. */
type LabelPoints = {
    readonly width: number;
    readonly height: number;
    readonly whiteKeys: number;
    readonly corners: readonly (readonly number[])[];
    readonly gaps: readonly (readonly number[])[];
    readonly blackLow: readonly (readonly number[])[];
    readonly blackHigh: readonly (readonly number[])[];
};
export type FitCheck = {
    readonly frame: string;
    readonly acquired: boolean;
    readonly board: string | null;
    readonly rightBoard: boolean;
    /** How many white keys the fitted numbering sits off the labels, 0 when it is right. */
    readonly shift: number | null;
    /** Median errors in white-key widths with the numbering as fitted; corners are the front two. */
    readonly gapKeys: number | null;
    readonly blackKeys: number | null;
    readonly cornerKeys: number | null;
};
/** How the fitted board lands on one frame's labels. */
export declare function checkFit(frame: string, fit: KeyNetFit | null, labels: LabelPoints): FitCheck;
/** Runs the app's own KeyNet loop on a still from a cold start as a camera would begin, and
 * returns the board it settles on with the peaks of the last run. */
export declare function fitStill(keyNet: KeyNetRunner, image: ImageBitmap): Promise<{
    fit: KeyNetFit | null;
    peaks: KeyNetPeaks | null;
}>;
/** Runs the app's own KeyNet loop on every labelled frame of the given recordings, each from a
 * cold start as a camera would begin, and scores the board it settles on. */
export declare function checkFits(clips: readonly string[], modelUrl?: string): Promise<FitCheck[]>;
export type FitSummary = {
    readonly frames: number;
    readonly acquired: number;
    readonly rightBoard: number;
    readonly rightNumbering: number;
    readonly gapKeys: number | null;
    readonly blackKeys: number | null;
    readonly cornerKeys: number | null;
};
export declare function summarise(checks: readonly FitCheck[]): FitSummary;
export {};
