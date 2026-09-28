import type { Point } from "./homography";
import { type KeyRegion } from "./keyseg";
declare global {
    interface Window {
        pianocvHeldQuad?: Point[];
        pianocvFollowedQuad?: Point[];
        pianocvLiveKeys?: {
            readonly outline: Point[] | null;
            readonly regions: readonly KeyRegion[];
            readonly current: boolean;
        };
        pianocvDrawnKeys?: readonly DrawnKey[];
        pianocvTemplateKeys?: readonly DrawnKey[];
    }
}
type DrawnKey = {
    readonly black: boolean;
    readonly semitone: number;
    readonly bar: readonly Point[];
};
export {};
