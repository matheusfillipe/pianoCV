import type { Point } from "./homography";
import {
  detectKeys,
  type FarEdge,
  measureFarEdge,
  rectifyStrip,
  type SourceImage,
  type Strip,
  trimFarEdge,
} from "./keystrip";

/** Where the black keys read, in strip-height fractions, matching the live app's own band. */
const BLACK_ROWS: readonly [number, number] = [0.12, 0.42];
/** Where the white-key separators read, near the player's edge. */
const NEAR_ROWS: readonly [number, number] = [0.8, 0.96];

type FrameResult =
  | {
      readonly status: "trimmed";
      readonly quad: Point[];
      readonly edge: FarEdge;
      readonly whiteKeys: number | null;
      readonly phase: string | null;
      readonly confidence: number;
    }
  | { readonly status: "not-trimmed" }
  | { readonly status: "unresolved" };

type RecordingResult =
  | {
      readonly corners: Point[];
      readonly trim: FarEdge;
      readonly whiteKeys: number | null;
      readonly phase: string | null;
      readonly confidence: number;
    }
  | { readonly skipped: string };

function luminance(data: Uint8ClampedArray, at: number): number {
  return 0.299 * data[at] + 0.587 * data[at + 1] + 0.114 * data[at + 2];
}

function bandMean(
  strip: Strip,
  fromFrac: number,
  toFrac: number,
): number | null {
  const fromY = Math.round(fromFrac * strip.height);
  const toY = Math.round(toFrac * strip.height);
  let sum = 0;
  let count = 0;
  for (let y = fromY; y < toY; y += 1) {
    for (let x = 0; x < strip.width; x += 1) {
      const at = (y * strip.width + x) * 4;
      if (strip.data[at + 3] === 0) {
        continue;
      }
      sum += luminance(strip.data, at);
      count += 1;
    }
  }
  return count === 0 ? null : sum / count;
}

/** True when the strip's far band reads darker than its near band, the signature of black
 * keys sitting at the top the way rectifyStrip expects; null when either band is unreadable. */
function blackKeysAtTop(strip: Strip): boolean | null {
  const black = bandMean(strip, BLACK_ROWS[0], BLACK_ROWS[1]);
  const near = bandMean(strip, NEAR_ROWS[0], NEAR_ROWS[1]);
  return black === null || near === null ? null : black < near;
}

type Oriented = { readonly quad: Point[]; readonly strip: Strip };

/** Hand labels do not guarantee the far edge comes first, so we rectify both readings of the
 * quad and keep whichever one alone shows black keys at the top. */
function resolveOrientation(
  source: SourceImage,
  quad: readonly Point[],
): Oriented | null {
  const forward = quad.slice(0, 4);
  const reversed = [quad[3], quad[2], quad[1], quad[0]];
  const forwardStrip = rectifyStrip(source, forward);
  const reversedStrip = rectifyStrip(source, reversed);
  const forwardOk = forwardStrip ? blackKeysAtTop(forwardStrip) : null;
  const reversedOk = reversedStrip ? blackKeysAtTop(reversedStrip) : null;
  if (forwardOk === true && reversedOk !== true && forwardStrip) {
    return { quad: forward, strip: forwardStrip };
  }
  if (reversedOk === true && forwardOk !== true && reversedStrip) {
    return { quad: reversed, strip: reversedStrip };
  }
  return null;
}

function computeFrame(
  source: SourceImage,
  quad: readonly Point[],
): FrameResult {
  const oriented = resolveOrientation(source, quad);
  if (oriented === null) {
    return { status: "unresolved" };
  }
  const edge = measureFarEdge(oriented.strip);
  if (edge === null) {
    return { status: "not-trimmed" };
  }
  const trimmedQuad = trimFarEdge(oriented.quad, edge);
  const trimmedStrip = rectifyStrip(source, trimmedQuad);
  const read = trimmedStrip ? detectKeys(trimmedStrip) : null;
  return {
    status: "trimmed",
    quad: trimmedQuad,
    edge,
    whiteKeys: read?.kind === "read" ? read.whiteKeys : null,
    phase: read?.kind === "read" ? read.phase : null,
    confidence: read?.confidence ?? 0,
  };
}

function frameToRecordingResult(frame: FrameResult): RecordingResult {
  if (frame.status !== "trimmed") {
    return {
      skipped:
        frame.status === "unresolved"
          ? "corner order could not be resolved"
          : "far edge could not be measured",
    };
  }
  return {
    corners: frame.quad,
    trim: frame.edge,
    whiteKeys: frame.whiteKeys,
    phase: frame.phase,
    confidence: frame.confidence,
  };
}

function canvasSource(canvas: HTMLCanvasElement): SourceImage {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (ctx === null) {
    throw new Error("cannot get a 2d canvas context");
  }
  const { data, width, height } = ctx.getImageData(
    0,
    0,
    canvas.width,
    canvas.height,
  );
  return { width, height, data };
}

async function loadImage(url: string): Promise<SourceImage> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`cannot fetch ${url}: ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    throw new Error("cannot get a 2d canvas context");
  }
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvasSource(canvas);
}

/** A recorder's webm often reports no duration until something seeks past its end, a quirk of
 * MediaRecorder output rather than of the clip itself. */
async function readableDuration(video: HTMLVideoElement): Promise<number> {
  if (Number.isFinite(video.duration) && video.duration > 0) {
    return video.duration;
  }
  return new Promise<number>((resolve) => {
    const onChange = (): void => {
      if (Number.isFinite(video.duration) && video.duration > 0) {
        video.removeEventListener("durationchange", onChange);
        resolve(video.duration);
      }
    };
    video.addEventListener("durationchange", onChange);
    video.currentTime = 1e7;
  });
}

function waitForEvent(
  target: HTMLVideoElement,
  event: string,
  label: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    target.addEventListener(event, () => resolve(), { once: true });
    target.addEventListener(
      "error",
      () => reject(new Error(`cannot ${label}`)),
      { once: true },
    );
  });
}

async function loadVideoMiddleFrame(url: string): Promise<SourceImage> {
  const video = document.createElement("video");
  video.muted = true;
  video.src = url;
  video.load();
  await waitForEvent(video, "loadedmetadata", "load video metadata");
  const duration = await readableDuration(video);
  video.currentTime = duration / 2;
  await waitForEvent(video, "seeked", "seek video");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    throw new Error("cannot get a 2d canvas context");
  }
  ctx.drawImage(video, 0, 0);
  return canvasSource(canvas);
}

type SidecarSummary = { readonly corners: { x: number; y: number }[] };

async function computeRecording(stem: string): Promise<RecordingResult> {
  const sidecar = (await (
    await fetch(
      `/relabel-lab/recordings/sidecar/${encodeURIComponent(stem)}.json`,
    )
  ).json()) as SidecarSummary;
  const source = await loadVideoMiddleFrame(
    `/relabel-lab/recordings/video/${encodeURIComponent(stem)}.webm`,
  );
  return frameToRecordingResult(computeFrame(source, sidecar.corners));
}

async function relabelRealSeg2(): Promise<Record<string, FrameResult>> {
  const corners = (await (
    await fetch("/relabel-lab/real-seg2/corners.json")
  ).json()) as Record<string, [number, number][]>;
  const results: Record<string, FrameResult> = {};
  for (const [stem, quad] of Object.entries(corners)) {
    try {
      const source = await loadImage(
        `/relabel-lab/real-seg2/frame/${encodeURIComponent(stem)}.png`,
      );
      const points = quad.map(([x, y]) => ({ x, y }));
      results[stem] = computeFrame(source, points);
    } catch (error) {
      console.error(`real-seg2 frame ${stem} failed`, error);
      results[stem] = { status: "unresolved" };
    }
  }
  return results;
}

async function relabelRecordings(): Promise<Record<string, RecordingResult>> {
  const stems = (await (
    await fetch("/relabel-lab/recordings/list.json")
  ).json()) as string[];
  const results: Record<string, RecordingResult> = {};
  for (const stem of stems) {
    try {
      results[stem] = await computeRecording(stem);
    } catch (error) {
      results[stem] = {
        skipped: `error: ${error instanceof Error ? error.message : String(error)}`,
      };
    }
  }
  return results;
}

async function main(): Promise<void> {
  const element = document.querySelector<HTMLPreElement>("#report");
  try {
    const realSeg2 = await relabelRealSeg2();
    const recordings = await relabelRecordings();
    const body = JSON.stringify({ realSeg2, recordings });
    if (element) {
      element.textContent = body;
    }
    const run =
      new URLSearchParams(window.location.search).get("run") ?? "relabel-keys";
    const response = await fetch(
      `/relabel-lab/report/${encodeURIComponent(run)}.json`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      },
    );
    if (!response.ok) {
      throw new Error(`cannot save report: ${response.status}`);
    }
    document.title = "relabel complete";
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    if (element) {
      element.textContent = message;
    }
    document.title = "relabel failed";
    throw error;
  }
}

void main();
