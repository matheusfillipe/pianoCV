/** Files the runtime loads at runtime rather than imports. Each bundler names
 * them its own way, so the app that embeds this hands them over. */
export interface RuntimeAssets {
  /** onnxruntime-web's WebGPU build's wasm. */
  readonly ortGpuWasm: string;
}

/** MediaPipe's loader script and its wasm, which only the demo's hand models load. */
export interface MediaPipeAssets {
  readonly mediapipeLoader: string;
  readonly mediapipeWasm: string;
}

export const handModelUrl =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

export const skinModelUrl =
  "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite";
