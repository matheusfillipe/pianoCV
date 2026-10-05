import {
  FilesetResolver,
  type HandLandmarkerResult,
  ImageSegmenter,
  type ImageSegmenterResult,
  type ImageSource,
} from "@mediapipe/tasks-vision";
import { type MediaPipeAssets, skinModelUrl } from "./assets";

// selfie_multiclass categories: 0 background, 1 hair, 2 body skin, 3 face skin, 4 clothes, 5 other
const BODY_SKIN = 2;

export interface SkinSegmenter {
  segment(frame: ImageSource, timestampMs: number): ImageSegmenterResult;
  close(): void;
}

export async function createSkinSegmenter(
  assets: MediaPipeAssets,
): Promise<SkinSegmenter> {
  const vision = await FilesetResolver.forVisionTasks();
  const files = {
    ...vision,
    wasmLoaderPath: assets.mediapipeLoader,
    wasmBinaryPath: assets.mediapipeWasm,
  };
  const options = {
    runningMode: "VIDEO" as const,
    outputCategoryMask: true,
    outputConfidenceMasks: false,
  };
  // The GPU runs this model in a fraction of the time the CPU delegate takes,
  // and a machine without one still has the CPU delegate behind it.
  const segmenter = await ImageSegmenter.createFromOptions(files, {
    ...options,
    baseOptions: { modelAssetPath: skinModelUrl, delegate: "GPU" },
  }).catch(() =>
    ImageSegmenter.createFromOptions(files, {
      ...options,
      baseOptions: { modelAssetPath: skinModelUrl, delegate: "CPU" },
    }),
  );
  return {
    segment: (video, timestampMs) =>
      segmenter.segmentForVideo(video, timestampMs),
    close: () => segmenter.close(),
  };
}

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

// skin reaches past the landmarks at the wrist and the outside of the palm
export function handBoxes(
  hands: HandLandmarkerResult | null,
  padding: number,
): Box[] {
  if (!hands) {
    return [];
  }
  return hands.landmarks.map((landmarks) => {
    let x0 = 1;
    let y0 = 1;
    let x1 = 0;
    let y1 = 0;
    for (const point of landmarks) {
      x0 = Math.min(x0, point.x);
      y0 = Math.min(y0, point.y);
      x1 = Math.max(x1, point.x);
      y1 = Math.max(y1, point.y);
    }
    const padX = (x1 - x0) * padding;
    const padY = (y1 - y0) * padding;
    return { x0: x0 - padX, y0: y0 - padY, x1: x1 + padX, y1: y1 + padY };
  });
}

export interface HandAlpha {
  canvas: HTMLCanvasElement;
  coverage: number;
}

let alphaCanvas: HTMLCanvasElement | null = null;
let alphaCtx: CanvasRenderingContext2D | null = null;
let alphaImage: ImageData | null = null;

function ensureAlpha(width: number, height: number): boolean {
  if (!alphaCanvas) {
    alphaCanvas = document.createElement("canvas");
  }
  if (alphaCanvas.width !== width || alphaCanvas.height !== height) {
    alphaCanvas.width = width;
    alphaCanvas.height = height;
    alphaCtx = alphaCanvas.getContext("2d");
    alphaImage = alphaCtx?.createImageData(width, height) ?? null;
  } else if (!alphaCtx) {
    alphaCtx = alphaCanvas.getContext("2d");
    alphaImage = alphaCtx?.createImageData(width, height) ?? null;
  }
  return alphaCtx !== null && alphaImage !== null;
}

// the segmenter finds every patch of skin in the frame; the boxes say which of it is a hand
export function buildSkinAlpha(
  category: Uint8Array,
  width: number,
  height: number,
  boxes: Box[],
  gateOnHands: boolean,
): HandAlpha | null {
  if (!ensureAlpha(width, height) || !alphaCtx || !alphaImage || !alphaCanvas) {
    return null;
  }
  const data = alphaImage.data;
  data.fill(0);
  let kept = 0;
  if (gateOnHands && boxes.length === 0) {
    alphaCtx.putImageData(alphaImage, 0, 0);
    return { canvas: alphaCanvas, coverage: 0 };
  }
  // walking only the rows and columns a hand covers keeps this off the critical path
  const regions = gateOnHands ? boxes : [{ x0: 0, y0: 0, x1: 1, y1: 1 }];
  for (const box of regions) {
    const startX = Math.max(0, Math.floor(box.x0 * width));
    const endX = Math.min(width, Math.ceil(box.x1 * width));
    const startY = Math.max(0, Math.floor(box.y0 * height));
    const endY = Math.min(height, Math.ceil(box.y1 * height));
    for (let y = startY; y < endY; y += 1) {
      const row = y * width;
      for (let x = startX; x < endX; x += 1) {
        if (category[row + x] !== BODY_SKIN) {
          continue;
        }
        const index = (row + x) * 4;
        data[index] = 255;
        data[index + 1] = 255;
        data[index + 2] = 255;
        data[index + 3] = 255;
        kept += 1;
      }
    }
  }
  alphaCtx.putImageData(alphaImage, 0, 0);
  return { canvas: alphaCanvas, coverage: kept / category.length };
}
