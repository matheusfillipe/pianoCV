import type { RuntimeAssets } from "./assets";
import type { Point } from "./homography";
declare global {
    interface Window {
        pianocvFit?: string;
        pianocvPoints?: Point[];
        pianocvPlain?: Point[];
        pianocvQuad?: Point[];
        pianocvRect?: Point[];
        pianocvProbability?: Float32Array;
        pianocvGray?: {
            gray: Float32Array;
            width: number;
            height: number;
        };
    }
}
export declare const INPUT_SIZE = 288;
export declare const MODEL_URL = "/keybed_seg2.onnx";
export declare const MASK_SIZE = 144;
export declare const MASK_THRESHOLD = 0.5;
export declare function boxAverage(gray: Float32Array, out: Float32Array): void;
export declare function medianChange(a: Float32Array, b: Float32Array, scratch: Float32Array): number;
export interface Detection {
    quad: Point[] | null;
    proposalQuad: Point[] | null;
    motion: number;
    still: boolean;
    inputQuad: Point[] | null;
    mask: Uint8Array;
    maskSize: number;
    coverage: number;
    confidence: number;
    latencyMs: number;
    gray: Float32Array;
}
export interface Detector {
    detect(frame: CanvasImageSource): Promise<Detection>;
    reset(): void;
}
export declare function grayscale(data: Uint8ClampedArray, out: Float32Array, scale?: number): void;
export declare function createDetector(assets: RuntimeAssets, url?: string): Promise<Detector>;
