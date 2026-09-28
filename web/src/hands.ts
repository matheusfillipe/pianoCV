import {
  HandLandmarker,
  type HandLandmarkerResult,
  type ImageSource,
} from "@mediapipe/tasks-vision";
import { handModelUrl, type RuntimeAssets } from "./assets";

export interface HandTracker {
  detect(frame: ImageSource, timestampMs: number): HandLandmarkerResult;
}

export async function createHandTracker(
  assets: RuntimeAssets,
): Promise<HandTracker> {
  const start = (delegate: "GPU" | "CPU"): Promise<HandLandmarker> =>
    HandLandmarker.createFromOptions(
      {
        wasmLoaderPath: assets.mediapipeLoader,
        wasmBinaryPath: assets.mediapipeWasm,
      },
      {
        baseOptions: { modelAssetPath: handModelUrl, delegate },
        numHands: 2,
        runningMode: "VIDEO",
      },
    );
  // a browser without a working WebGL context still finds hands, only slower
  const handLandmarker = await start("GPU").catch(() => start("CPU"));
  return {
    detect: (frame, timestampMs) =>
      handLandmarker.detectForVideo(frame, timestampMs),
  };
}
