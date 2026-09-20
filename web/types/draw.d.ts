import { type HandLandmarkerResult } from "@mediapipe/tasks-vision";
import type { Point } from "./homography";
import type { DetectedKey } from "./keystrip";
export declare function drawHands(ctx: CanvasRenderingContext2D, hands: HandLandmarkerResult, w: number, h: number): void;
export declare function drawQuad(ctx: CanvasRenderingContext2D, quad: Point[], w: number, h: number, color: string, label?: string): void;
export declare function drawKeys(ctx: CanvasRenderingContext2D, keys: readonly DetectedKey[], w: number, h: number): void;
export declare function drawModelInput(ctx: CanvasRenderingContext2D, gray: Float32Array, size: number, quad: Point[], box: number): void;
