import type { Point } from "./homography";
export interface Stillness {
    readonly changed: (frame: CanvasImageSource, keybed: readonly Point[] | null) => boolean;
    readonly forget: () => void;
}
export declare function pointInQuad(quad: readonly Point[], at: Point): boolean;
/** Whether the camera is looking at the same scene it was. A held keybed does
 * not move while it is played, so the model only has to run again once the
 * picture around it does. */
export declare function createStillness(): Stillness;
