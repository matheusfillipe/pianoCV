import type { Point } from "./homography";
declare global {
    interface Window {
        pianocvHeldQuad?: Point[];
        pianocvFollowedQuad?: Point[];
        pianocvLiveKeys?: {
            readonly outline: Point[] | null;
            readonly regions: number;
            readonly current: boolean;
        };
    }
}
