import { type HandLandmarkerResult, type ImageSource } from "@mediapipe/tasks-vision";
import { type MediaPipeAssets } from "./assets";
export interface HandTracker {
    detect(frame: ImageSource, timestampMs: number): HandLandmarkerResult;
}
export declare function createHandTracker(assets: MediaPipeAssets): Promise<HandTracker>;
