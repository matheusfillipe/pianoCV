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

export type GpuRequest =
  | {
      readonly kind: "open";
      readonly id: number;
      readonly url: string;
      readonly wasm: string;
    }
  | {
      readonly kind: "run";
      readonly id: number;
      readonly url: string;
      readonly input: Float32Array;
      readonly dims: readonly number[];
    }
  | {
      readonly kind: "segment";
      readonly id: number;
      readonly url: string;
      readonly frame: ImageBitmap;
      readonly crop: Crop;
    }
  | {
      readonly kind: "keynet";
      readonly id: number;
      readonly url: string;
      readonly input: Float32Array;
      readonly dims: readonly number[];
    };

export type GpuReply =
  | {
      readonly kind: "opened";
      readonly id: number;
      /** Where the model runs: "webgpu", or "wasm" where the GPU could not start it. */
      readonly backend: string;
      /** Why the GPU could not run the model, when it could not. */
      readonly gpuFailure: string | null;
    }
  | { readonly kind: "failed"; readonly id: number; readonly reason: string }
  | {
      readonly kind: "ran";
      readonly id: number;
      /** The model's first output, or null when it gave none. */
      readonly output: Float32Array | null;
    }
  | {
      readonly kind: "segmented";
      readonly id: number;
      /** The outline's corners in frame pixels, or null when too few key pixels were found. */
      readonly corners: { x: number; y: number }[] | null;
      /** Every key region, in frame pixels. */
      readonly regions: KeyRegion[];
    }
  | {
      readonly kind: "keynetresult";
      readonly id: number;
      /** The model's raw outputs, or null when it gave no heatmaps. */
      readonly outputs: KeyNetOutputs | null;
    };

export type KeyNetOutputs = {
  readonly heat: Float32Array;
  readonly presence: number;
  readonly offsets: Float32Array | null;
};

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const plane = CROP_WIDTH * CROP_HEIGHT;
const input = new Float32Array(3 * plane);
const canvas = new OffscreenCanvas(CROP_WIDTH, CROP_HEIGHT);
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const sessions = new Map<string, ort.InferenceSession>();

const post = (reply: GpuReply, transfer: Transferable[] = []): void => {
  self.postMessage(reply, { transfer });
};

async function open(id: number, url: string, wasm: string): Promise<void> {
  // the bundled build carries its own WebGPU loader, which an overridden script would replace
  ort.env.wasm.wasmPaths = { wasm };
  // a GPU adapter onnxruntime rejects fails the WebGPU start, and this worker's runtime then
  // falls back to wasm for that model
  let gpuFailure: string | null = null;
  for (const providers of [["webgpu"], ["wasm"]]) {
    try {
      const session = await ort.InferenceSession.create(url, {
        executionProviders: providers,
        graphOptimizationLevel: "all",
      });
      sessions.set(url, session);
      post({ kind: "opened", id, backend: providers[0], gpuFailure });
      return;
    } catch (error) {
      if (providers[0] === "wasm") {
        post({
          kind: "failed",
          id,
          reason: `${gpuFailure} then ${String(error)}`,
        });
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
  width: number,
  height: number,
): void {
  const a = crop.along.x / crop.scale;
  const c = crop.along.y / crop.scale;
  const b = crop.across.x / crop.scale;
  const d = crop.across.y / crop.scale;
  const e =
    width / 2 -
    (crop.along.x * crop.centre.x + crop.along.y * crop.centre.y) / crop.scale;
  const f =
    height / 2 -
    (crop.across.x * crop.centre.x + crop.across.y * crop.centre.y) /
      crop.scale;
  target.setTransform(1, 0, 0, 1, 0, 0);
  target.fillStyle = "#000";
  target.fillRect(0, 0, width, height);
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

async function runKeyNet(
  session: ort.InferenceSession,
  input: Float32Array,
  dims: readonly number[],
): Promise<KeyNetOutputs | null> {
  // the offset head trains to zero unless a recipe turns it on, so we skip reading it back
  const outputs = await session.run(
    { [session.inputNames[0]]: new ort.Tensor("float32", input, dims) },
    ["heatmaps", "presence"],
  );
  const heat = await outputs.heatmaps?.getData();
  const presence = await outputs.presence?.getData();
  if (!(heat instanceof Float32Array)) {
    return null;
  }
  return {
    heat,
    presence: presence instanceof Float32Array ? presence[0] : 0,
    offsets: null,
  };
}

/** ImageNet-normalises `pixels` (RGBA, `plane` pixels) into `out` as planar float32 RGB. */
function normalizeInto(
  pixels: Uint8ClampedArray,
  plane: number,
  out: Float32Array,
): void {
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      out[c * plane + i] = (pixels[i * 4 + c] / 255 - MEAN[c]) / STD[c];
    }
  }
}

async function firstOutput(
  session: ort.InferenceSession,
  input: Float32Array,
  dims: readonly number[],
): Promise<Float32Array | null> {
  const outputs = await session.run({
    [session.inputNames[0]]: new ort.Tensor("float32", input, dims),
  });
  const data = await outputs[session.outputNames[0]]?.getData();
  return data instanceof Float32Array ? data : null;
}

async function segment(
  session: ort.InferenceSession | undefined,
  frame: ImageBitmap,
  crop: Crop,
): Promise<{
  corners: { x: number; y: number }[] | null;
  regions: KeyRegion[];
}> {
  if (session === undefined || ctx === null) {
    frame.close();
    return { corners: null, regions: [] };
  }
  drawCrop(ctx, frame, crop, CROP_WIDTH, CROP_HEIGHT);
  frame.close();
  const pixels = ctx.getImageData(0, 0, CROP_WIDTH, CROP_HEIGHT).data;
  normalizeInto(pixels, plane, input);
  const probabilities = await firstOutput(session, input, [
    1,
    3,
    CROP_HEIGHT,
    CROP_WIDTH,
  ]);
  if (probabilities === null) {
    return { corners: null, regions: [] };
  }
  const classes = classesOf(probabilities);
  const toFrame = (x: number, y: number) => cropToFrame(crop, x, y);
  return {
    corners: keyOutline(classes)?.map((p) => toFrame(p.x, p.y)) ?? null,
    regions: keyRegions(classes, toFrame),
  };
}

// messages arrive one at a time and we await each, so the GPU never runs two inferences at once
let queue: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<GpuRequest>) => {
  const request = event.data;
  queue = queue.then(async () => {
    if (request.kind === "open") {
      await open(request.id, request.url, request.wasm);
      return;
    }
    const session = sessions.get(request.url);
    if (request.kind === "run") {
      const output = session
        ? await firstOutput(session, request.input, request.dims).catch(
            () => null,
          )
        : null;
      post(
        { kind: "ran", id: request.id, output },
        output ? [output.buffer] : [],
      );
      return;
    }
    if (request.kind === "keynet") {
      const outputs = session
        ? await runKeyNet(session, request.input, request.dims).catch(
            () => null,
          )
        : null;
      post({ kind: "keynetresult", id: request.id, outputs });
      return;
    }
    const found = await segment(session, request.frame, request.crop).catch(
      () => ({ corners: null, regions: [] }),
    );
    post({ kind: "segmented", id: request.id, ...found });
  });
};
