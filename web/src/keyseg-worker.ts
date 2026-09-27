import * as ort from "onnxruntime-web/webgpu";
import {
  CROP_HEIGHT,
  CROP_WIDTH,
  type Crop,
  classesOf,
  cropToFrame,
  type KeyRegion,
  keyOutline,
  keyRegions,
} from "./keyseg";

export type SegmentRequest =
  | {
      readonly kind: "init";
      readonly url: string;
      readonly wasm: string;
    }
  | {
      readonly kind: "segment";
      readonly id: number;
      readonly frame: ImageBitmap;
      readonly crop: Crop;
    };

export type SegmentReply =
  | {
      readonly kind: "ready";
      readonly backend: string;
      /** Why the GPU could not run the model, when it could not. */
      readonly gpuFailure: string | null;
    }
  | { readonly kind: "failed"; readonly reason: string }
  | {
      readonly kind: "segmented";
      readonly id: number;
      /** The outline's corners in frame pixels, or null when too few key pixels were found. */
      readonly corners: { x: number; y: number }[] | null;
      /** Every key region, in frame pixels. */
      readonly regions: KeyRegion[];
    };

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const plane = CROP_WIDTH * CROP_HEIGHT;
const input = new Float32Array(3 * plane);
const canvas = new OffscreenCanvas(CROP_WIDTH, CROP_HEIGHT);
const ctx = canvas.getContext("2d", { willReadFrequently: true });
let session: ort.InferenceSession | null = null;

const post = (reply: SegmentReply): void => {
  self.postMessage(reply);
};

async function start(url: string, wasm: string): Promise<void> {
  // the bundled build carries its own WebGPU loader, which an overridden script would replace
  ort.env.wasm.wasmPaths = { wasm };
  // a GPU adapter onnxruntime rejects fails the WebGPU start, and this worker's own runtime then
  // falls back to wasm without touching the page's other models
  let gpuFailure: string | null = null;
  for (const providers of [["webgpu"], ["wasm"]]) {
    try {
      session = await ort.InferenceSession.create(url, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      post({ kind: "ready", backend: providers[0], gpuFailure });
      return;
    } catch (error) {
      if (providers[0] === "wasm") {
        post({ kind: "failed", reason: `${gpuFailure} then ${String(error)}` });
      } else {
        gpuFailure = String(error);
      }
    }
  }
}

/** Draws the crop with the canvas's own transform; the half-pixel shift matches OpenCV's
 * warpAffine, which puts pixel centres on whole numbers where a canvas puts them halfway. */
function drawCrop(
  target: OffscreenCanvasRenderingContext2D,
  frame: ImageBitmap,
  crop: Crop,
): void {
  const a = crop.along.x / crop.scale;
  const c = crop.along.y / crop.scale;
  const b = crop.across.x / crop.scale;
  const d = crop.across.y / crop.scale;
  const e =
    CROP_WIDTH / 2 -
    (crop.along.x * crop.centre.x + crop.along.y * crop.centre.y) / crop.scale;
  const f =
    CROP_HEIGHT / 2 -
    (crop.across.x * crop.centre.x + crop.across.y * crop.centre.y) /
      crop.scale;
  target.setTransform(1, 0, 0, 1, 0, 0);
  target.fillStyle = "#000";
  target.fillRect(0, 0, CROP_WIDTH, CROP_HEIGHT);
  target.setTransform(
    a,
    b,
    c,
    d,
    e - 0.5 * (a + c) + 0.5,
    f - 0.5 * (b + d) + 0.5,
  );
  target.drawImage(frame, 0, 0);
}

async function segment(
  frame: ImageBitmap,
  crop: Crop,
): Promise<{
  corners: { x: number; y: number }[] | null;
  regions: KeyRegion[];
}> {
  if (session === null || ctx === null) {
    frame.close();
    return { corners: null, regions: [] };
  }
  drawCrop(ctx, frame, crop);
  frame.close();
  const pixels = ctx.getImageData(0, 0, CROP_WIDTH, CROP_HEIGHT).data;
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      input[c * plane + i] = (pixels[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
  const outputs = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", input, [
      1,
      3,
      CROP_HEIGHT,
      CROP_WIDTH,
    ]),
  });
  const probabilities = await outputs[session.outputNames[0]]?.getData();
  if (!(probabilities instanceof Float32Array)) {
    return { corners: null, regions: [] };
  }
  const classes = classesOf(probabilities);
  const toFrame = (x: number, y: number) => cropToFrame(crop, x, y);
  return {
    corners: keyOutline(classes)?.map((p) => toFrame(p.x, p.y)) ?? null,
    regions: keyRegions(classes, toFrame),
  };
}

// messages arrive one at a time and we await each, so one session never runs two inferences
let queue: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<SegmentRequest>) => {
  const request = event.data;
  queue = queue.then(async () => {
    if (request.kind === "init") {
      await start(request.url, request.wasm);
      return;
    }
    const found = await segment(request.frame, request.crop).catch(() => ({
      corners: null,
      regions: [],
    }));
    post({ kind: "segmented", id: request.id, ...found });
  });
};
