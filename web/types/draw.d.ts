import { type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import type { Point } from "./homography";
export declare function drawHands(ctx: CanvasRenderingContext2D, hands: HandLandmarkerResult, w: number, h: number): void;
export declare function drawQuad(ctx: CanvasRenderingContext2D, quad: Point[], w: number, h: number, color: string, label?: string): void;
type Key = {
    readonly black: boolean;
    readonly bar: readonly Point[];
};
export declare function tracePolygon(ctx: CanvasRenderingContext2D, bar: readonly Point[], w: number, h: number): void;
/** Runs `draw` with every black key cut out of the canvas, so what it draws of the white keys
 * stops at the black keys' edges the way the keys themselves do. */
export declare function outsideBlackKeys(ctx: CanvasRenderingContext2D, keys: readonly Key[], w: number, h: number, draw: () => void): void;
export declare function drawKeys(ctx: CanvasRenderingContext2D, keys: readonly Key[], w: number, h: number): void;
export {};
