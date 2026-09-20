import type { Point } from "./homography";
declare global {
    interface Window {
        kvtHeldQuad?: Point[];
        kvtFollowedQuad?: Point[];
    }
}
