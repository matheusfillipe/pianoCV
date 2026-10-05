import wasmLoaderUrl from "@mediapipe/tasks-vision/vision_wasm_internal.js?url";
import wasmBinaryUrl from "@mediapipe/tasks-vision/vision_wasm_internal.wasm?url";
import ortGpuWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import type { MediaPipeAssets, RuntimeAssets } from "./assets";

export const viteAssets: RuntimeAssets & MediaPipeAssets = {
  ortGpuWasm: ortGpuWasmUrl,
  mediapipeLoader: wasmLoaderUrl,
  mediapipeWasm: wasmBinaryUrl,
};
