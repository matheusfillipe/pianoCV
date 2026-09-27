import { type ImageSegmenterResult } from "@mediapipe/tasks-vision";
import { type RuntimeAssets } from "./assets";
export interface OcclusionMask {
    segment(frame: HTMLVideoElement, timestampMs: number): ImageSegmenterResult;
}
export declare function createOcclusionMask(assets: RuntimeAssets): Promise<OcclusionMask>;
/** Draws the camera's own pixels back on top wherever the mask says hand, arm or sleeve, so
 * whatever was drawn before this call (the key glow) ends up sitting behind the player's hands. */
export declare function drawOcclusion(ctx: CanvasRenderingContext2D, video: HTMLVideoElement, result: ImageSegmenterResult, w: number, h: number): void;
