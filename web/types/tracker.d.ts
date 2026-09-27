import type { Detector } from "./detector";
import type { Point } from "./homography";
import { type Measurement } from "./measure";
import { type Stillness } from "./stillness";
/** Hunting runs the model often since nothing is held yet. Held only has to
 * ask whether the keyboard moved, so a glance runs far less often. */
export declare const searchEveryMs = 120;
export declare const glanceEveryMs = 500;
export declare const confirmEveryMs = 2500;
export declare const trustStillnessForMs = 30000;
export declare const readsToHold = 4;
export declare const agreeWithin = 0.02;
export declare const missesBeforeLost = 3;
export declare const missesWhileHeld = 20;
export declare const staysWithin = 0.12;
export type Measured = Extract<Measurement, {
    kind: "depth" | "focal";
}>;
export type Progress = {
    readonly agreed: number;
    readonly reason: string;
};
/** A keybed the tracker is still confirming, one it holds fixed, or one it
 * lost and stopped looking for. Corners are never moved once held: a held
 * keybed is held or it is gone. */
export type TrackerState = {
    readonly kind: "hunting";
    readonly progress: Progress;
} | {
    readonly kind: "held";
    readonly quad: Point[];
    readonly byHand: boolean;
} | {
    readonly kind: "lost";
    readonly reason: string;
};
export type Reading = {
    readonly latencyMs: number;
    readonly coverage: number;
    readonly confidence: number;
};
export type TrackerOptions = {
    /** Told when a held keybed stops being there, so whatever draws it can
     * stop. */
    readonly onLost?: () => void;
    /** Told what a hand-placed quad measured, so the caller can persist it. */
    readonly onMeasured?: (measurement: Measured) => void;
    readonly stillness?: Stillness;
};
export interface Tracker {
    readonly look: (frame: HTMLVideoElement, now: number) => Promise<void>;
    readonly state: () => TrackerState;
    readonly reading: () => Reading | null;
    /** Takes the corners a person dragged, which hold until release(). */
    readonly hold: (quad: readonly Point[]) => void;
    readonly release: () => void;
}
export declare function createTracker(detector: Detector, options?: TrackerOptions): Tracker;
