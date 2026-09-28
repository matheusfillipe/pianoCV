import type { Point } from "./homography";
type LabelKey = {
    readonly pitch: number;
    readonly black: boolean;
    readonly top: readonly Point[];
    readonly front: readonly Point[] | null;
};
/** A frame the page labelled itself, in the sidecar format of the synthetic renders. */
export type KeyLabel = {
    readonly png: string;
    /** White where a hand or sleeve covers the frame, which training leaves out. */
    readonly ignorePng: string;
    readonly sidecar: {
        readonly kind: "real-keys";
        readonly imageWidth: number;
        readonly imageHeight: number;
        readonly corners: readonly Point[];
        readonly keys: readonly LabelKey[];
        readonly fit: number;
        readonly confidence: number;
    };
};
export type KeyLabels = {
    readonly kind: "labelled";
    readonly labels: readonly KeyLabel[];
} | {
    readonly kind: "skipped";
    readonly reason: string;
};
/** Waits for the key reader to settle, then, when its chosen read is trusted and fits its
 * picture well, grabs `count` frames `everyMs` apart with the keys the page lays over each. */
export declare function captureKeyLabels(count?: number, everyMs?: number): Promise<KeyLabels>;
export {};
