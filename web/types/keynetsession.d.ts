import { type KeyNetFit, type KeyNetPeaks, type KeyNetStep } from "./keycore";
import type { KeyNetRunner } from "./keynetrunner";
import type { Size } from "./keyspace";
declare global {
    interface Window {
        pianocvKeyNetMs?: {
            search: number[];
            track: number[];
        };
    }
}
export type KeyNetSession = {
    readonly fit: () => KeyNetFit | null;
    /** The peaks the latest run decoded. */
    readonly peaks: () => KeyNetPeaks | null;
    /** Runs the model once on the frame and moves the session on; null while a run is in flight. */
    readonly step: (frame: CanvasImageSource, size: Size, now: number) => Promise<KeyNetStep | null>;
};
export declare function createKeyNetSession(keyNet: KeyNetRunner): KeyNetSession;
