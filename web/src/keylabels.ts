import type { MPMask } from "@mediapipe/tasks-vision";
import type { Point } from "./homography";
import { convexHull } from "./keyseg";
import type { DetectedKey } from "./keystrip";
import { TRUSTED_READ } from "./keystrip";
import { createOcclusionMask, OCCLUDING_CATEGORIES } from "./occlusion";
import { viteAssets } from "./viteassets";

/** A read this well fitted to its picture is trusted to label frames for training. */
const LABEL_FIT = 0.85;
const SETTLE_READS = 12;
const SETTLE_WAIT_MS = 40_000;
/** How far we grow a hand's landmark hull about its centre, since the landmarks sit on the
 * joints and the hand reaches past them. */
const HAND_GROWTH = 1.35;
const HANDS_WAIT_MS = 5000;

type LabelKey = {
  readonly pitch: number;
  readonly black: boolean;
  readonly top: readonly Point[];
  readonly front: readonly Point[] | null;
};

/** A frame the page labelled itself, in the sidecar format of the synthetic renders. */
export type KeyLabel = {
  readonly png: string;
  /** White where a hand or sleeve covers the frame, which training leaves out. */
  readonly ignorePng: string;
  readonly sidecar: {
    readonly kind: "real-keys";
    readonly imageWidth: number;
    readonly imageHeight: number;
    readonly corners: readonly Point[];
    readonly keys: readonly LabelKey[];
    readonly fit: number;
    readonly confidence: number;
  };
};

export type KeyLabels =
  | { readonly kind: "labelled"; readonly labels: readonly KeyLabel[] }
  | { readonly kind: "skipped"; readonly reason: string };

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/** The template's faces as one entry per key, a black key's front face after its top. */
function labelKeys(faces: readonly DetectedKey[]): LabelKey[] {
  const keys: LabelKey[] = [];
  for (let i = 0; i < faces.length; i += 1) {
    const face = faces[i];
    const front = faces[i + 1];
    if (face.black && front?.black && front.semitone === face.semitone) {
      keys.push({
        pitch: face.semitone,
        black: true,
        top: face.bar,
        front: front.bar,
      });
      i += 1;
    } else {
      keys.push({
        pitch: face.semitone,
        black: face.black,
        top: face.bar,
        front: null,
      });
    }
  }
  return keys;
}

/** Each hand the page found, as its landmarks' hull grown about its centre. */
function handOutlines(): Point[][] {
  return (window.pianocvHands?.landmarks ?? []).map((hand) => {
    const hull = convexHull(hand.map((p) => ({ x: p.x, y: p.y })));
    const centre = {
      x: hull.reduce((sum, p) => sum + p.x, 0) / hull.length,
      y: hull.reduce((sum, p) => sum + p.y, 0) / hull.length,
    };
    return hull.map((p) => ({
      x: centre.x + (p.x - centre.x) * HAND_GROWTH,
      y: centre.y + (p.y - centre.y) * HAND_GROWTH,
    }));
  });
}

/** Paints the canvas black, then white wherever the person segmenter sees skin or a sleeve and
 * over each hand the hand tracker found, since either alone misses some hands. */
function drawIgnored(
  ctx: CanvasRenderingContext2D,
  categories: MPMask | undefined,
): void {
  const { width, height } = ctx.canvas;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, width, height);
  if (categories !== undefined) {
    const data = categories.getAsUint8Array();
    const small = new ImageData(categories.width, categories.height);
    for (let i = 0; i < data.length; i += 1) {
      const covered = OCCLUDING_CATEGORIES.has(data[i]) ? 255 : 0;
      small.data.set([covered, covered, covered, 255], i * 4);
    }
    const mask = document.createElement("canvas");
    mask.width = categories.width;
    mask.height = categories.height;
    mask.getContext("2d")?.putImageData(small, 0, 0);
    ctx.drawImage(mask, 0, 0, width, height);
  }
  ctx.fillStyle = "#fff";
  for (const hand of handOutlines()) {
    ctx.beginPath();
    hand.forEach((p, i) => {
      if (i === 0) {
        ctx.moveTo(p.x * width, p.y * height);
      } else {
        ctx.lineTo(p.x * width, p.y * height);
      }
    });
    ctx.closePath();
    ctx.fill();
  }
}

/** Waits for the key reader to settle, then, when its chosen read is trusted and fits its
 * picture well, grabs `count` frames `everyMs` apart with the keys the page lays over each. */
export async function captureKeyLabels(
  count = 20,
  everyMs = 400,
): Promise<KeyLabels> {
  const press = (toggle: string): void => {
    [...document.querySelectorAll("button")]
      .find((b) => b.textContent?.trim() === toggle)
      ?.click();
  };
  if (!window.pianocvKeyFaces?.length) {
    press("keys");
  }
  if (window.pianocvHands === undefined) {
    press("hands");
  }
  for (
    let waited = 0;
    window.pianocvHands === undefined && waited < HANDS_WAIT_MS;
    waited += 100
  ) {
    await wait(100);
  }
  const started = performance.now();
  while (
    (window.pianocvReadFits?.length ?? 0) < SETTLE_READS &&
    performance.now() - started < SETTLE_WAIT_MS
  ) {
    await wait(250);
  }
  // a label drawn over a hand would teach the model that hands are keys
  if (window.pianocvHands === undefined) {
    const status = document.body.innerText
      .split("\n")
      .find((line) => line.includes("hand tracker"));
    return {
      kind: "skipped",
      reason: `no hand tracker to mask hands with: ${status ?? "never started"}`,
    };
  }
  const chosen = window.pianocvReadFits?.find((read) => read.chosen);
  if (chosen === undefined) {
    return { kind: "skipped", reason: "no board read" };
  }
  if (chosen.confidence < TRUSTED_READ || chosen.fit < LABEL_FIT) {
    return {
      kind: "skipped",
      reason: `read at ${chosen.confidence.toFixed(2)} confidence, ${chosen.fit.toFixed(2)} fit`,
    };
  }
  const video = document.querySelector("video");
  if (video === null || video.videoWidth === 0) {
    return { kind: "skipped", reason: "no video" };
  }
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return { kind: "skipped", reason: "no canvas" };
  }
  const occlusion = await createOcclusionMask(viteAssets);
  const labels: KeyLabel[] = [];
  for (let i = 0; i < count; i += 1) {
    const faces = window.pianocvKeyFaces ?? [];
    const corners = window.pianocvFollowedQuad;
    if (faces.length > 0 && corners?.length === 4) {
      ctx.drawImage(video, 0, 0);
      const png = canvas.toDataURL("image/png");
      drawIgnored(
        ctx,
        occlusion.segment(video, performance.now()).categoryMask,
      );
      labels.push({
        png,
        ignorePng: canvas.toDataURL("image/png"),
        sidecar: {
          kind: "real-keys",
          imageWidth: canvas.width,
          imageHeight: canvas.height,
          corners,
          keys: labelKeys(faces),
          fit: chosen.fit,
          confidence: chosen.confidence,
        },
      });
    }
    await wait(everyMs);
  }
  return { kind: "labelled", labels };
}
