import {
  decodeHeatmaps,
  type KeyNetFit,
  KeyNetLoop,
  type KeyNetPeaks,
  type KeyNetStep,
  prepareInput,
} from "./keycore";
import type { KeyNetRunner } from "./keynetrunner";
import type { Size } from "./keyspace";

declare global {
  interface Window {
    // recent keynet round-trip costs in ms, per mode, for the lab to read off a live page
    pianocvKeyNetMs?: { search: number[]; track: number[] };
  }
}
const KEYNET_MS_HISTORY = 300;
// a frame larger than this is shrunk by the browser before we read its pixels
const MOST_GRAB_WIDTH = 1920;

export type KeyNetSession = {
  readonly fit: () => KeyNetFit | null;
  /** The peaks the latest run decoded. */
  readonly peaks: () => KeyNetPeaks | null;
  /** Runs the model once on the frame and moves the session on; null while a run is in flight. */
  readonly step: (
    frame: CanvasImageSource,
    size: Size,
    now: number,
  ) => Promise<KeyNetStep | null>;
};

let grabCanvas: OffscreenCanvas | null = null;
let grabContext: OffscreenCanvasRenderingContext2D | null = null;

function grab(frame: CanvasImageSource, size: Size): ImageData | null {
  grabCanvas ??= new OffscreenCanvas(1, 1);
  grabContext ??= grabCanvas.getContext("2d", { willReadFrequently: true });
  if (grabContext === null) {
    return null;
  }
  const width = Math.min(size.width, MOST_GRAB_WIDTH);
  const height = Math.round((size.height * width) / size.width);
  grabCanvas.width = width;
  grabCanvas.height = height;
  grabContext.drawImage(frame, 0, 0, width, height);
  return grabContext.getImageData(0, 0, width, height);
}

export function createKeyNetSession(keyNet: KeyNetRunner): KeyNetSession {
  const loop = new KeyNetLoop(
    new URLSearchParams(location.search).get("crop") === "rectified",
  );
  let fit: KeyNetFit | null = null;
  let peaks: KeyNetPeaks | null = null;
  let busy = false;
  return {
    fit: () => fit,
    peaks: () => peaks,
    step: async (frame, size, now) => {
      if (busy) {
        return null;
      }
      busy = true;
      try {
        const crop = loop.nextCrop(size);
        const started = performance.now();
        const pixels = grab(frame, size);
        const outputs =
          pixels === null
            ? null
            : await keyNet.run(
                prepareInput(pixels.data, pixels, crop),
                crop.width,
                crop.height,
              );
        if (outputs === null || pixels === null) {
          return { fit, acquired: false };
        }
        window.pianocvKeyNetMs ??= { search: [], track: [] };
        const history = window.pianocvKeyNetMs[crop.mode];
        history.push(performance.now() - started);
        if (history.length > KEYNET_MS_HISTORY) {
          history.shift();
        }
        peaks = decodeHeatmaps(
          outputs.heat,
          crop.width,
          crop.height,
          crop.matrix,
          outputs.offsets,
        );
        const result = loop.step(
          outputs.presence,
          peaks,
          pixels.data,
          pixels,
          size,
          now,
        );
        fit = result.fit;
        return result;
      } finally {
        busy = false;
      }
    },
  };
}
