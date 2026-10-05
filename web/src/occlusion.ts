import {
  ImageSegmenter,
  type ImageSegmenterResult,
} from "@mediapipe/tasks-vision";
import { type MediaPipeAssets, skinModelUrl } from "./assets";

export interface OcclusionMask {
  segment(frame: HTMLVideoElement, timestampMs: number): ImageSegmenterResult;
}

export async function createOcclusionMask(
  assets: MediaPipeAssets,
): Promise<OcclusionMask> {
  const segmenter = await ImageSegmenter.createFromOptions(
    {
      wasmLoaderPath: assets.mediapipeLoader,
      wasmBinaryPath: assets.mediapipeWasm,
    },
    {
      baseOptions: { modelAssetPath: skinModelUrl, delegate: "GPU" },
      runningMode: "VIDEO",
      outputCategoryMask: true,
      outputConfidenceMasks: false,
    },
  );
  return {
    segment: (frame, timestampMs) =>
      segmenter.segmentForVideo(frame, timestampMs),
  };
}

/** body-skin (hands and arms) and clothes (sleeves) both occlude the glow. */
export const OCCLUDING_CATEGORIES: ReadonlySet<number> = new Set([2, 4]);

let maskCanvas: HTMLCanvasElement | null = null;
let compositeCanvas: HTMLCanvasElement | null = null;

function sized(canvas: HTMLCanvasElement, width: number, height: number): void {
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

/** Draws the camera's own pixels back on top wherever the mask says hand, arm or sleeve, so
 * whatever was drawn before this call (the key glow) ends up sitting behind the player's hands. */
export function drawOcclusion(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  result: ImageSegmenterResult,
  w: number,
  h: number,
): void {
  const categoryMask = result.categoryMask;
  if (!categoryMask) {
    return;
  }
  const data = categoryMask.getAsUint8Array();
  const mw = categoryMask.width;
  const mh = categoryMask.height;

  maskCanvas ??= document.createElement("canvas");
  sized(maskCanvas, mw, mh);
  const maskCtx = maskCanvas.getContext("2d");
  if (!maskCtx) {
    return;
  }
  const image = maskCtx.createImageData(mw, mh);
  for (let i = 0; i < data.length; i += 1) {
    image.data[i * 4 + 3] = OCCLUDING_CATEGORIES.has(data[i]) ? 255 : 0;
  }
  maskCtx.putImageData(image, 0, 0);

  compositeCanvas ??= document.createElement("canvas");
  sized(compositeCanvas, w, h);
  const compositeCtx = compositeCanvas.getContext("2d");
  if (!compositeCtx) {
    return;
  }
  compositeCtx.clearRect(0, 0, w, h);
  compositeCtx.drawImage(video, 0, 0, w, h);
  compositeCtx.globalCompositeOperation = "destination-in";
  compositeCtx.drawImage(maskCanvas, 0, 0, w, h);
  compositeCtx.globalCompositeOperation = "source-over";

  ctx.drawImage(compositeCanvas, 0, 0, w, h);
}
