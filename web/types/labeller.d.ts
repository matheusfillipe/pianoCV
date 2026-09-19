import type { BoardRead } from "./board";
import type { Corners } from "./calibrate";
import { type Stillness } from "./stillness";
import type { TrackerState } from "./tracker";
export declare const saveEveryMs = 3000;
export declare const savesPerSession = 300;
export declare const agreementFloor = 0.9;
type Source = "hand" | "board";
export interface LabelSidecar {
    readonly kind: "auto";
    readonly startedAt: number;
    readonly durationMs: number;
    readonly corners: Corners;
    readonly imageWidth: number;
    readonly imageHeight: number;
    readonly mimeType: string;
    /** Shared by every frame saved from one held keyboard, so a training split never tears one
     * hold across train and validation. */
    readonly session: string;
    readonly source: Source;
    readonly agreement?: number;
}
export type Save = (frame: HTMLVideoElement, sidecar: LabelSidecar) => void;
export interface LabellerOptions {
    readonly save?: Save;
    readonly stillness?: Stillness;
    readonly newSession?: () => string;
}
export interface Labeller {
    readonly saved: () => number;
    readonly look: (frame: HTMLVideoElement, tracker: TrackerState, board: BoardRead | null, enabled: boolean, now: number) => void;
}
/** Turns ordinary use of a held keyboard into training frames: every frame the demo app drives
 * through here, saving stills whenever the tracker and the board reader agree the corners are
 * trustworthy, spaced out so a still camera does not produce a pile of near-duplicates. */
export declare function createLabeller(options?: LabellerOptions): Labeller;
export {};
