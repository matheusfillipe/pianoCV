import type {
  HandLandmarkerResult,
  ImageSegmenterResult,
} from "@mediapipe/tasks-vision";
import {
  type Calibration,
  type Corners,
  canonicalQuad,
  createCalibration,
} from "./calibrate";
import { drawHands, drawKeys, drawQuad } from "./draw";
import { createEffects, lowestPitchFor } from "./effects";
import { createHandTracker, type HandTracker } from "./hands";
import { createHud, type Hud } from "./hud";
import type { KeyNetFit } from "./keycore";
import { createKeyNet, KEYNET_URL, type KeyNetRunner } from "./keynetrunner";
import { createKeyNetSession } from "./keynetsession";
import { type KeyOutline, keyOutlines } from "./keyoutlines";
import { createLab } from "./lab";
import {
  createOcclusionMask,
  drawOcclusion,
  type OcclusionMask,
} from "./occlusion";
import { viteAssets } from "./viteassets";

declare global {
  interface Window {
    // the keys drawn this frame, as frame fractions
    pianocvDrawnKeys?: readonly KeyOutline[];
  }
}

// the site can be served under a path, as on a project page, and the models sit beside it
function modelUrl(file: string): string {
  return `${import.meta.env.BASE_URL}${file.replace(/^\//, "")}`;
}

const MANUAL_COLOR = "rgba(56,189,248,0.9)";
const AUTO_COLOR = "#4ade80";
function createVideo(): HTMLVideoElement {
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.style.display = "none";
  document.body.appendChild(video);
  return video;
}

function createCanvas(): HTMLCanvasElement {
  const canvas = document.createElement("canvas");
  canvas.style.position = "fixed";
  canvas.style.inset = "0";
  canvas.style.width = "100vw";
  canvas.style.height = "100vh";
  document.body.appendChild(canvas);
  return canvas;
}

function renderError(canvas: HTMLCanvasElement, message: string): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    return;
  }
  ctx.fillStyle = "#050505";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#e5e5e5";
  ctx.font = "16px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

function errorMessage(err: unknown): string {
  if (err instanceof DOMException && err.name === "NotAllowedError") {
    return "camera access denied";
  }
  return err instanceof Error ? err.message : String(err);
}

async function startCamera(video: HTMLVideoElement): Promise<void> {
  // ?clip=<recording>.webm plays a saved recording in place of the camera, so the whole
  // pipeline can be watched in a browser with no camera at all
  const clip = new URLSearchParams(location.search).get("clip");
  if (clip) {
    video.src = `/lab/clip/${clip}`;
    video.loop = true;
    video.muted = true;
  } else {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });
  }
  await new Promise<void>((resolve) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
  await video.play();
}

// the video fills the canvas without stretching, so the overlay sits on the pixels the model saw
function videoBox(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
): { x: number; y: number; w: number; h: number } {
  const scale = Math.min(
    canvas.width / video.videoWidth,
    canvas.height / video.videoHeight,
  );
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  return { x: (canvas.width - w) / 2, y: (canvas.height - h) / 2, w, h };
}

// what the HUD's "keynet" status line says: hunting for a keyboard, or the board it tracks and
// whether the black keys' tops are used
function keynetStatusText(fit: KeyNetFit | null): string {
  if (fit === null) {
    return "searching";
  }
  const tops = fit.lift !== null ? "black tops used" : "no black tops seen";
  return `tracking ${fit.whiteKeys} white keys from ${fit.phase}, ${tops}`;
}

function startLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  hud: Hud,
  handTracker: HandTracker | null,
  occlusionMask: OcclusionMask | null,
  keyNet: KeyNetRunner | null,
): () => Corners {
  let hands: HandLandmarkerResult | null = null;
  let lastVideoTime = -1;
  let segmented: ImageSegmenterResult | null = null;
  let lastSegmentVideoTime = -1;
  const effects = createEffects();
  const keyNetSession = keyNet ? createKeyNetSession(keyNet) : null;

  // the corners are often placed perfectly but a half turn out, which reads as front and back
  // swapped; rolling by two relabels the same rectangle rather than making it be dragged again
  hud.onFlip(() => {
    const c = calibration.getCorners();
    calibration.setCorners([c[2], c[3], c[0], c[1]]);
  });

  // the dragged corners mean what was dragged: handle 1 to 2 runs along the black keys,
  // and the flip button is how the back is swapped
  const orientedManual = (): Corners =>
    canonicalQuad(calibration.getCorners()) as Corners;

  const frame = (now: number): void => {
    if (handTracker && hud.state.hands && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = handTracker.detect(video, now);
    }
    if (
      occlusionMask &&
      hud.state.glow &&
      video.currentTime !== lastSegmentVideoTime
    ) {
      lastSegmentVideoTime = video.currentTime;
      // segmentForVideo copies its masks per call, so we close the previous result by hand or
      // its GPU texture and wasm buffer are never freed
      segmented?.close();
      segmented = occlusionMask.segment(video, now);
    }
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const box = videoBox(video, canvas);
    ctx.drawImage(video, box.x, box.y, box.w, box.h);

    ctx.save();
    ctx.translate(box.x, box.y);
    const size = { width: video.videoWidth, height: video.videoHeight };
    if (keyNetSession && size.width > 0 && size.height > 0) {
      void keyNetSession.step(video, size, now).catch(() => null);
    }
    const fit = keyNetSession?.fit() ?? null;
    if (fit) {
      drawQuad(ctx, [...fit.quad], box.w, box.h, AUTO_COLOR, "keybed");
      const keys = keyOutlines(fit);
      window.pianocvDrawnKeys = keys;
      if (hud.state.keys) {
        drawKeys(ctx, keys, box.w, box.h);
      }
      if (hud.state.glow) {
        effects.draw(
          ctx,
          keys,
          lowestPitchFor(fit.whiteKeys, fit.phase),
          box.w,
          box.h,
          now,
        );
        if (segmented) {
          drawOcclusion(ctx, video, segmented, box.w, box.h);
        }
      }
    } else {
      window.pianocvDrawnKeys = undefined;
    }
    hud.status(
      "keynet",
      keyNet ? `${keyNet.backend}, ${keynetStatusText(fit)}` : "unavailable",
    );
    if (hud.state.corners) {
      drawQuad(ctx, orientedManual(), box.w, box.h, MANUAL_COLOR, "manual");
      calibration.draw(ctx, box.w, box.h);
    }
    if (hands && hud.state.hands) {
      drawHands(ctx, hands, box.w, box.h);
    }
    ctx.restore();
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  return orientedManual;
}

async function boot(): Promise<void> {
  const video = createVideo();
  const canvas = createCanvas();
  let errorText: string | null = null;

  const resize = (): void => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    if (errorText) {
      renderError(canvas, errorText);
    }
  };
  resize();
  window.addEventListener("resize", resize);

  try {
    const hud = createHud();
    const occlusionLoading = createOcclusionMask(viteAssets).then(
      (mask) => {
        hud.status("glow", "play a key to light it");
        return mask;
      },
      () => {
        hud.status("glow", "occlusion unavailable, glow draws over hands");
        return null;
      },
    );
    const keyNetLoading = createKeyNet(viteAssets, modelUrl(KEYNET_URL)).then(
      (runner) => {
        hud.status("keynet", `keynet ready on ${runner.backend}`);
        return runner;
      },
      () => {
        hud.status("keynet", "keynet unavailable");
        return null;
      },
    );
    await startCamera(video);
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("2d canvas context unavailable");
    }
    const calibration = createCalibration(
      canvas,
      () => hud.state.corners,
      () => videoBox(video, canvas),
    );
    const handTracker = await createHandTracker(viteAssets).then(
      (tracker) => {
        hud.status("hands", "hand tracker ready");
        return tracker;
      },
      (err) => {
        hud.status("hands", `hand tracker unavailable (${errorMessage(err)})`);
        return null;
      },
    );
    const occlusionMask = await occlusionLoading;
    const labelCorners = startLoop(
      video,
      canvas,
      ctx,
      calibration,
      hud,
      handTracker,
      occlusionMask,
      await keyNetLoading,
    );
    const stream = video.srcObject;
    if (stream instanceof MediaStream) {
      createLab({
        video,
        stream,
        getCorners: labelCorners,
        mount: hud.capture,
      });
    }
  } catch (err) {
    errorText = errorMessage(err);
    renderError(canvas, errorText);
  }
}

void boot();
