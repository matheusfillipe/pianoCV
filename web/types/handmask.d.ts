import { type HandLandmarkerResult, type ImageSegmenterResult, type ImageSource } from "@mediapipe/tasks-vision";
import { type MediaPipeAssets } from "./assets";
export interface SkinSegmenter {
    segment(frame: ImageSource, timestampMs: number): ImageSegmenterResult;
    close(): void;
}
export declare function createSkinSegmenter(assets: MediaPipeAssets): Promise<SkinSegmenter>;
export interface Box {
    x0: number;
    y0: number;
    x1: number;
    y1: number;
}
export declare function handBoxes(hands: HandLandmarkerResult | null, padding: number): Box[];
export interface HandAlpha {
    canvas: HTMLCanvasElement;
    coverage: number;
}
export declare function buildSkinAlpha(category: Uint8Array, width: number, height: number, boxes: Box[], gateOnHands: boolean): HandAlpha | null;
