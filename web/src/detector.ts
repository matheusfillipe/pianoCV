import * as ort from "onnxruntime-web/wasm";
import type { RuntimeAssets } from "./assets";
import { quadFromMask } from "./fitquad";
import type { Point } from "./homography";
import { cameraFocalFraction } from "./pose";
import {
  boundaryPoints,
  boundaryResidual,
  fitRectangle,
  lastDecline,
  principalBox,
  refineEnds,
  snapToGradient,
} from "./rectfit";
import { refineQuad } from "./refineedges";

declare global {
  interface Window {
    // why the rectangle fit last declined, and the evidence it was given, for a lab session
    // watching a recording
    kvtFit?: string;
    kvtPoints?: Point[];
    kvtPlain?: Point[];
    kvtQuad?: Point[];
    kvtRect?: Point[];
    kvtProbability?: Float32Array;
    kvtGray?: { gray: Float32Array; width: number; height: number };
  }
}

export const INPUT_SIZE = 288;
export const MODEL_URL = "/keybed_seg2.onnx";
// the backbone is ImageNet pretrained, so it wants rgb on those statistics rather than grey
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
export const MASK_SIZE = 144;
export const MASK_THRESHOLD = 0.5;
// edges are refined on an unsquashed frame so each edge normal is really perpendicular,
// capped at the width the 10 px search radius was measured on
const REFINE_WIDTH = 640;
// COEP would be needed for ORT threads and would also block the cross-origin hand landmarker
// model, so the wasm build stays single threaded and the work goes to the GPU instead
const THREADS = 1;
// when nothing in the frame moves, the mask is averaged over this many detections before it
// is fitted: measured on static clips it cuts corner jitter 5-12x and jumps 19 -> 4
const ACCUMULATE = 8;
// the rectangle may explain the boundary this much worse than the free quad and still show:
// a smaller margin made the two trade places frame to frame, since both snap to the same gradients
const FIT_SLACK_PX = 2;
// how far a still scene's new solve pulls the held rectangle per detection, and the corner
// move beyond which the solve is a correction the held rectangle follows at once
const STILL_BLEND = 0.25;
const RESOLVE_PX = 20;
// the scene counts as still while the MEDIAN per-pixel change on a 72x72 average stays under
// this: a hand changes only a patch (0.0039 max), while a 1 px camera shift already scores that
const STILL_BELOW = 0.006;
const MOTION_BOX = 4;
const MOTION_SIZE = INPUT_SIZE / MOTION_BOX;

export function boxAverage(gray: Float32Array, out: Float32Array): void {
  const inv = 1 / (MOTION_BOX * MOTION_BOX);
  for (let y = 0; y < MOTION_SIZE; y += 1) {
    for (let x = 0; x < MOTION_SIZE; x += 1) {
      let sum = 0;
      for (let dy = 0; dy < MOTION_BOX; dy += 1) {
        const row = (y * MOTION_BOX + dy) * INPUT_SIZE + x * MOTION_BOX;
        for (let dx = 0; dx < MOTION_BOX; dx += 1) {
          sum += gray[row + dx];
        }
      }
      out[y * MOTION_SIZE + x] = sum * inv;
    }
  }
}

export function medianChange(
  a: Float32Array,
  b: Float32Array,
  scratch: Float32Array,
): number {
  for (let i = 0; i < a.length; i += 1) {
    scratch[i] = Math.abs(a[i] - b[i]);
  }
  scratch.sort();
  return scratch[scratch.length >> 1];
}

function sourceSize(frame: CanvasImageSource): {
  width: number;
  height: number;
} {
  if (frame instanceof HTMLVideoElement) {
    return { width: frame.videoWidth, height: frame.videoHeight };
  }
  if (frame instanceof HTMLImageElement) {
    return { width: frame.naturalWidth, height: frame.naturalHeight };
  }
  if (frame instanceof HTMLCanvasElement || frame instanceof ImageBitmap) {
    return { width: frame.width, height: frame.height };
  }
  return { width: INPUT_SIZE, height: INPUT_SIZE };
}

export interface Detection {
  quad: Point[] | null;
  // median per-pixel change since the last detection, and whether that counted as still
  motion: number;
  still: boolean;
  // the same corners inside the model's own square input, which is what `gray` holds
  inputQuad: Point[] | null;
  mask: Uint8Array;
  maskSize: number;
  coverage: number;
  // mean mask probability inside the drawn quad: a real keybed is near 1, a guess is not
  confidence: number;
  latencyMs: number;
  gray: Float32Array;
}

export interface Detector {
  detect(frame: CanvasImageSource): Promise<Detection>;
}

function grayscale(
  data: Uint8ClampedArray,
  out: Float32Array,
  scale = 1,
): void {
  for (let i = 0; i < out.length; i += 1) {
    const p = i * 4;
    out[i] =
      ((0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2]) / 255) *
      scale;
  }
}

// planar rgb, one channel after another, which is the layout the exported graph expects
function planarRgb(
  data: Uint8ClampedArray,
  out: Float32Array,
  size: number,
): void {
  const plane = size * size;
  for (let i = 0; i < plane; i += 1) {
    const p = i * 4;
    for (let c = 0; c < 3; c += 1) {
      out[c * plane + i] = (data[p + c] / 255 - MEAN[c]) / STD[c];
    }
  }
}

function floats(value: ort.Tensor | undefined, name: string): Float32Array {
  if (!value || !(value.data instanceof Float32Array)) {
    throw new Error(`model output ${name} is not a float tensor`);
  }
  return value.data;
}

// mean mask probability over the grid cells whose centres lie inside the quad
function meanInside(
  probability: Float32Array,
  size: number,
  quad: Point[],
  width: number,
  height: number,
): number {
  let total = 0;
  let count = 0;
  for (let row = 0; row < size; row += 1) {
    const y = (row / (size - 1)) * height;
    for (let col = 0; col < size; col += 1) {
      const x = (col / (size - 1)) * width;
      let sign = 0;
      let inside = true;
      for (let i = 0; i < 4 && inside; i += 1) {
        const a = quad[i];
        const b = quad[(i + 1) % 4];
        const s = Math.sign((b.x - a.x) * (y - a.y) - (b.y - a.y) * (x - a.x));
        if (s === 0) {
          continue;
        }
        if (sign === 0) {
          sign = s;
        } else if (s !== sign) {
          inside = false;
        }
      }
      if (inside) {
        total += probability[row * size + col];
        count += 1;
      }
    }
  }
  return count ? total / count : 0;
}

export async function createDetector(
  assets: RuntimeAssets,
  url: string = MODEL_URL,
): Promise<Detector> {
  ort.env.wasm.wasmPaths = { wasm: assets.ortWasm };
  ort.env.wasm.numThreads = THREADS;
  // The wasm backend, measured against the alternative: onnxruntime's WebGPU
  // build fails to start at all on a machine whose adapter it dislikes, and a
  // failed start leaves the wasm backend unable to start either, so the whole
  // detector goes with it. The model runs here and the picture is polled rarely
  // instead.
  const session = await ort.InferenceSession.create(url, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  const canvas = document.createElement("canvas");
  canvas.width = INPUT_SIZE;
  canvas.height = INPUT_SIZE;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) {
    throw new Error("2d canvas context unavailable");
  }
  ctx.imageSmoothingQuality = "high";
  const gray = new Float32Array(INPUT_SIZE * INPUT_SIZE);
  const rgb = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);

  const full = document.createElement("canvas");
  const fullCtx = full.getContext("2d", { willReadFrequently: true });
  if (!fullCtx) {
    throw new Error("2d canvas context unavailable");
  }
  let fullGray = new Float32Array(0);
  // the grey frame averaged over the same still frames as the mask, since a single frame's
  // far-end noise re-tilted the whole rectangle 3-5 px on a near-top view
  let steadyGray = new Float32Array(0);
  let steadyCount = 0;
  const coarse = new Float32Array(MOTION_SIZE * MOTION_SIZE);
  const previous = new Float32Array(MOTION_SIZE * MOTION_SIZE);
  const scratch = new Float32Array(MOTION_SIZE * MOTION_SIZE);
  let havePrevious = false;
  const recent: Float32Array[] = [];
  const averaged = new Float32Array(MASK_SIZE * MASK_SIZE);
  const probability = new Float32Array(MASK_SIZE * MASK_SIZE);

  let previousFit: Point[] | null = null;

  // the plain quad refitted as the keybed rectangle against the mask boundary, or kept
  // when the rectangle explains that boundary no better
  const constrain = (
    quad: Point[] | null,
    gray: Float32Array,
    width: number,
    height: number,
    still: boolean,
  ): Point[] | null => {
    // the last frame's rectangle is the best start and gate for boundary crossings: gating by
    // the plain quad instead, a different spike each frame, moved the rectangle 3-6 px per detection
    const raw = boundaryPoints(
      probability,
      MASK_SIZE,
      width,
      height,
      previousFit ?? quad,
    );
    if (raw.length < 8) {
      window.kvtFit = "no boundary";
      return quad;
    }
    // with no plain quad at all (a folded blob the quad builder gave up on) the boundary
    // still exists, and its own box stands in for the plain quad as the start
    const start = previousFit ?? quad ?? principalBox(raw);
    const points = snapToGradient(gray, width, height, raw, start);
    window.kvtPoints = points;
    window.kvtPlain = quad ?? undefined;
    window.kvtProbability = probability.slice();
    window.kvtGray = { gray: gray.slice(), width, height };
    const focal = cameraFocalFraction() * width;
    const frame = { gray, width, height };
    let held = fitRectangle(points, start, width, height, focal);
    if (!held && previousFit && quad) {
      held = fitRectangle(points, quad, width, height, focal);
    }
    if (!held) {
      window.kvtFit = lastDecline;
      previousFit = null;
      return quad;
    }
    // while nothing moves the keybed cannot have moved, so each new solve only pulls the held
    // rectangle part of the way; on a near-top view mask noise alone re-tilts a fresh solve 3-5 px
    let rectangle = held.rectangle;
    const settled =
      still &&
      previousFit !== null &&
      previousFit.every(
        (p, i) =>
          Math.hypot(p.x - rectangle[i].x, p.y - rectangle[i].y) <= RESOLVE_PX,
      );
    if (settled && previousFit) {
      const anchor = previousFit;
      rectangle = anchor.map((p, i) => ({
        x: p.x + (rectangle[i].x - p.x) * STILL_BLEND,
        y: p.y + (rectangle[i].y - p.y) * STILL_BLEND,
      }));
    }
    previousFit = rectangle;
    const fitted = refineEnds(frame, rectangle);
    window.kvtQuad = fitted;
    window.kvtRect = rectangle;
    if (!quad) {
      window.kvtFit = `fit at focal ${focal.toFixed(0)} from the boundary alone`;
      return fitted;
    }
    // the rectangle earns its place by explaining the boundary at least as well as the free
    // quad; at a wrong focal it cannot, and the free quad is the better picture
    const plainResidual = boundaryResidual(points, quad);
    const heldResidual = boundaryResidual(points, fitted);
    const wins = heldResidual <= plainResidual + FIT_SLACK_PX;
    window.kvtFit = `${wins ? "fit" : "plain"} at focal ${focal.toFixed(0)}: residual fit ${heldResidual.toFixed(2)} plain ${plainResidual.toFixed(2)}`;
    return wins ? fitted : quad;
  };

  const motion = (): number => {
    boxAverage(gray, coarse);
    const change = havePrevious
      ? medianChange(coarse, previous, scratch)
      : Infinity;
    previous.set(coarse);
    havePrevious = true;
    return change;
  };

  return {
    detect: async (frame) => {
      // canvas bilinear aliases a 4x downscale badly and the key pattern is exactly that fine,
      // so ask the browser for a properly filtered resample the way cv2 INTER_AREA does
      const source = sourceSize(frame);
      // the frame goes in whole and squashed. Letterboxing it to the 4:3 of the training
      // frames measured worse (12.1 px against 5.7): the net has never seen a black bar.
      const bitmap = await createImageBitmap(frame, {
        resizeWidth: INPUT_SIZE,
        resizeHeight: INPUT_SIZE,
        resizeQuality: "high",
      });
      ctx.drawImage(bitmap, 0, 0);
      bitmap.close();
      const pixels = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data;
      // the grey copy still feeds the front/back test and the model-input overlay
      grayscale(pixels, gray);
      planarRgb(pixels, rgb, INPUT_SIZE);
      const started = performance.now();
      const outputs = await session.run({
        image: new ort.Tensor("float32", rgb, [1, 3, INPUT_SIZE, INPUT_SIZE]),
      });
      const single = floats(outputs.mask, "mask");
      const change = motion();
      const still = change <= STILL_BELOW;
      if (!still) {
        recent.length = 0;
      }
      recent.push(Float32Array.from(single));
      if (recent.length > ACCUMULATE) {
        recent.shift();
      }
      averaged.fill(0);
      for (const map of recent) {
        for (let i = 0; i < averaged.length; i += 1) {
          averaged[i] += map[i];
        }
      }
      const scale = 1 / recent.length;
      const mask = new Uint8Array(probability.length);
      let kept = 0;
      for (let i = 0; i < probability.length; i += 1) {
        probability[i] = averaged[i] * scale;
        if (probability[i] > MASK_THRESHOLD) {
          mask[i] = 1;
          kept += 1;
        }
      }
      const refineScale = Math.min(1, REFINE_WIDTH / Math.max(source.width, 1));
      const refineWidth = Math.max(1, Math.round(source.width * refineScale));
      const refineHeight = Math.max(1, Math.round(source.height * refineScale));
      if (full.width !== refineWidth || full.height !== refineHeight) {
        full.width = refineWidth;
        full.height = refineHeight;
        fullGray = new Float32Array(refineWidth * refineHeight);
      }
      fullCtx.drawImage(frame, 0, 0, refineWidth, refineHeight);
      grayscale(
        fullCtx.getImageData(0, 0, refineWidth, refineHeight).data,
        fullGray,
        255,
      );
      if (!still || steadyGray.length !== fullGray.length) {
        steadyGray = Float32Array.from(fullGray);
        steadyCount = 1;
      } else {
        steadyCount = Math.min(steadyCount + 1, ACCUMULATE);
        const share = 1 / steadyCount;
        for (let i = 0; i < steadyGray.length; i += 1) {
          steadyGray[i] += (fullGray[i] - steadyGray[i]) * share;
        }
      }
      const coarse = quadFromMask(mask, MASK_SIZE, MASK_SIZE);
      // a 144 mask pixel is 2 input pixels wide, so the edges are snapped onto the input's
      // own gradients before the quad is handed on
      const refined = coarse
        ? refineQuad(
            steadyGray,
            refineWidth,
            refineHeight,
            coarse.map((p) => ({
              x: (p.x / (MASK_SIZE - 1)) * refineWidth,
              y: (p.y / (MASK_SIZE - 1)) * refineHeight,
            })),
          )
        : null;
      const constrained = constrain(
        refined,
        steadyGray,
        refineWidth,
        refineHeight,
        still,
      );
      return {
        motion: change,
        still,
        // the mask grid is square while the frame is not, so corners come back in frame fractions
        quad: constrained
          ? constrained.map((p) => ({
              x: p.x / refineWidth,
              y: p.y / refineHeight,
            }))
          : null,
        inputQuad: constrained
          ? constrained.map((p) => ({
              x: p.x / refineWidth,
              y: p.y / refineHeight,
            }))
          : null,
        mask,
        maskSize: MASK_SIZE,
        coverage: kept / probability.length,
        confidence: constrained
          ? meanInside(
              probability,
              MASK_SIZE,
              constrained.map((p) => ({
                x: p.x / refineWidth,
                y: p.y / refineHeight,
              })),
              1,
              1,
            )
          : 0,
        latencyMs: performance.now() - started,
        gray: gray.slice(),
      };
    },
  };
}
