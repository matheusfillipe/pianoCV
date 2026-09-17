import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import { calibrateBoard } from "./boardgeometry";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import { createDetector, type Detector, INPUT_SIZE } from "./detector";
import { drawHands, drawModelInput, drawQuad } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import type { Point } from "./homography";
import { createHud, type Hud } from "./hud";
import { type Board, drawKeyMasks } from "./keypolygons";
import { createLab } from "./lab";
import { lockKeybed } from "./lock";
import { depthInKeyWidths, measureCorners } from "./measure";
import { canonicalQuad, setCameraFocal, setKeybedDepth } from "./pose";
import { createSteady } from "./steady";
import { viteAssets } from "./viteassets";

const DETECT_INTERVAL_MS = 250;
const MANUAL_COLOR = "rgba(56,189,248,0.9)";
const AUTO_COLOR = "#4ade80";
const WEAK_COLOR = "#f87171";
const MODEL_VIEW_PX = 216;
// a couple of rejected frames is noise; a run of them means the keybed really is gone
const MISSES_BEFORE_CLEAR = 4;
const DEPTH_KEY = "kvt.keybedDepthUnits.v2";
const FOCAL_KEY = "kvt.cameraFocalFraction.v2";
const SHOW_KEY_MASKS = new URLSearchParams(location.search).has("keymask");

// when a recording plays in place of the camera, every detection is kept on the window so
// a lab session can read the pipeline's behaviour over time
interface LabRecord {
  t: number;
  ms: number;
  still: boolean;
  motion: number;
  raw: { x: number; y: number }[] | null;
  drawn: { x: number; y: number }[] | null;
  note: string;
  // the snapped boundary points the rectangle fit was given, in frame pixels
  points?: { x: number; y: number }[];
}
declare global {
  interface Window {
    kvtLab?: LabRecord[];
  }
}
function labLog(record: LabRecord): void {
  window.kvtLab?.push(record);
}

interface Lock {
  quad: Point[];
  inputQuad: Point[];
  latencyMs: number;
  onKeybed: boolean;
  margin: number;
  gray: Float32Array;
}

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
    window.kvtLab = [];
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

function startLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  hud: Hud,
  detector: Detector | null,
  tracker: HandTracker | null,
): () => Corners {
  let lock: Lock | null = null;
  let misses = 0;
  const steady = createSteady();
  let hands: HandLandmarkerResult | null = null;
  let lastVideoTime = -1;
  let inFlight = false;
  let lastDetectAt = -Infinity;
  let board: Board | null = null;
  let boardQuad: Point[] | null = null;
  let boardProposal: Point[] | null = null;
  let lastBoardReadAt = -Infinity;
  let lastDetectionStill = false;

  const clearBoard = (): void => {
    board = null;
    boardQuad = null;
    boardProposal = null;
    lastBoardReadAt = -Infinity;
  };

  const detect = (now: number): void => {
    if (!detector || inFlight || video.videoWidth === 0) {
      return;
    }
    inFlight = true;
    lastDetectAt = now;
    detector
      .detect(video)
      .then((detection) => {
        if (!detection.quad) {
          labLog({
            t: performance.now(),
            ms: detection.latencyMs,
            still: detection.still,
            motion: detection.motion,
            raw: null,
            drawn: null,
            note: "no keybed",
          });
          lock = null;
          clearBoard();
          lastDetectionStill = false;
          steady.reset();
          hud.status("detect", `${detection.latencyMs.toFixed(1)} ms`);
          hud.status(
            "keybed",
            `no keybed (mask ${(detection.coverage * 100).toFixed(1)}%)`,
          );
          return;
        }
        const held = lockKeybed(detection, {
          width: video.videoWidth,
          height: video.videoHeight,
        });
        if (!held.held) {
          lastDetectionStill = false;
          labLog({
            t: performance.now(),
            ms: detection.latencyMs,
            still: detection.still,
            motion: detection.motion,
            raw: detection.quad,
            drawn: null,
            note: held.reason,
          });
          misses += 1;
          if (misses >= MISSES_BEFORE_CLEAR) {
            lock = null;
            clearBoard();
            lastDetectionStill = false;
            steady.reset();
          }
          hud.status("detect", `${detection.latencyMs.toFixed(1)} ms`);
          hud.status("keybed", `${held.reason} (${misses} missed)`);
          return;
        }
        const framed = held.quad;
        const proposal = boardProposal;
        if (
          proposal &&
          framed.some(
            (p, i) =>
              Math.hypot(p.x - proposal[i].x, p.y - proposal[i].y) > 0.04,
          )
        ) {
          clearBoard();
        }
        const facts = {
          quad: held.inputQuad,
          margin: held.margin,
          onKeybed: true,
        };
        misses = 0;
        lastDetectionStill = detection.still;
        lock = {
          quad: steady.accept(framed, detection.still),
          inputQuad: facts.quad,
          latencyMs: detection.latencyMs,
          onKeybed: facts.onKeybed,
          margin: facts.margin,
          gray: detection.gray,
        };
        labLog({
          t: performance.now(),
          ms: detection.latencyMs,
          still: detection.still,
          motion: detection.motion,
          raw: framed,
          drawn: lock.quad,
          note: "",
          points: window.kvtPoints,
        });
        hud.status(
          "detect",
          `${detection.latencyMs.toFixed(1)} ms  ${detection.still ? "still" : "moving"} ${
            Number.isFinite(detection.motion)
              ? detection.motion.toFixed(4)
              : "-"
          }`,
        );
        hud.status(
          "keybed",
          `mask ${(detection.coverage * 100).toFixed(1)}%  ` +
            `${facts.onKeybed ? "keys found" : "no key pattern"} (${facts.margin.toFixed(2)})`,
        );
      })
      .catch((err: unknown) => {
        lastDetectionStill = false;
        hud.status("model", errorMessage(err));
      })
      .finally(() => {
        inFlight = false;
      });
  };

  hud.onRedetect(() => {
    clearBoard();
    lastDetectionStill = false;
    detect(performance.now());
  });
  hud.onAdopt(() => {
    if (lock) {
      calibration.setCorners(lock.quad);
    }
  });
  // the corners are often placed perfectly but a half turn out, which reads as front and back
  // swapped; rolling by two relabels the same rectangle rather than making it be dragged again
  hud.onFlip(() => {
    const c = calibration.getCorners();
    calibration.setCorners([c[2], c[3], c[0], c[1]]);
  });
  // the keybed's shape and the camera's lens, each measured once from the dragged corners
  // and kept in this browser: the shape from a view from above, where the aspect is plain
  // to see, the lens from an oblique view, where the convergence fixes it. Which one a
  // press measures is decided by how unequal the two ends are.
  const remember = (key: string, value: number): void => {
    try {
      localStorage.setItem(key, String(value));
    } catch {
      // storage can be blocked; the measurement holds for this session
    }
  };
  try {
    const depth = localStorage.getItem(DEPTH_KEY);
    if (depth) {
      setKeybedDepth(Number(depth));
    }
    const focal = localStorage.getItem(FOCAL_KEY);
    if (focal) {
      setCameraFocal(Number(focal));
    }
  } catch {
    // storage can be blocked; the defaults stand
  }
  hud.onMeasure(() => {
    const measured = measureCorners(orientedManual(), {
      width: video.videoWidth,
      height: video.videoHeight,
    });
    if (measured.kind === "refused") {
      hud.status("keybed", `${measured.reason}, nothing measured`);
      return;
    }
    if (measured.kind === "depth") {
      setKeybedDepth(measured.units);
      remember(DEPTH_KEY, measured.units);
      clearBoard();
      lastDetectionStill = false;
      hud.status(
        "keybed",
        `shape measured: ${depthInKeyWidths(measured.units).toFixed(2)} key widths per depth`,
      );
      return;
    }
    setCameraFocal(measured.fraction);
    remember(FOCAL_KEY, measured.fraction);
    clearBoard();
    lastDetectionStill = false;
    hud.status(
      "keybed",
      `lens measured: focal ${measured.fraction.toFixed(2)} of the frame width`,
    );
  });

  // the dragged corners mean what was dragged: handle 1 to 2 runs along the black keys,
  // and the flip button is how the back is swapped. Deciding the back from the picture
  // every frame made the handles jump between the two orders on their own.
  const orientedManual = (): Corners =>
    canonicalQuad(calibration.getCorners()) as Corners;

  const frame = (now: number): void => {
    if (hud.state.live && now - lastDetectAt > DETECT_INTERVAL_MS) {
      detect(now);
    }
    if (tracker && hud.state.hands && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = tracker.detect(video, now);
    }
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const box = videoBox(video, canvas);
    ctx.drawImage(video, box.x, box.y, box.w, box.h);

    if (
      lock &&
      board === null &&
      lastDetectionStill &&
      now - lastBoardReadAt > 1500
    ) {
      lastBoardReadAt = now;
      const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const fitted = calibrateBoard(
        image,
        lock.quad.map((point) => ({
          x: box.x + point.x * box.w,
          y: box.y + point.y * box.h,
        })),
      );
      board = fitted?.board ?? null;
      boardProposal = fitted ? lock.quad.map((point) => ({ ...point })) : null;
      boardQuad =
        fitted?.quad.map((point) => ({
          x: (point.x - box.x) / box.w,
          y: (point.y - box.y) / box.h,
        })) ?? null;
      hud.status(
        "keys",
        board
          ? `${board.lowest} to ${board.highest}, black depth ${board.blackDepth.toFixed(2)}`
          : "reading black keys",
      );
    }

    ctx.save();
    ctx.translate(box.x, box.y);
    if (lock) {
      drawQuad(
        ctx,
        board && boardQuad ? boardQuad : lock.quad,
        box.w,
        box.h,
        lock.onKeybed ? AUTO_COLOR : WEAK_COLOR,
        lock.onKeybed ? "keybed" : "no key pattern",
      );
      if (SHOW_KEY_MASKS && board && boardQuad) {
        drawKeyMasks(ctx, boardQuad, board, box.w, box.h);
      }
    }
    if (hud.state.corners) {
      drawQuad(ctx, orientedManual(), box.w, box.h, MANUAL_COLOR, "manual");
      calibration.draw(ctx, box.w, box.h);
    }
    if (hands && hud.state.hands) {
      drawHands(ctx, hands, box.w, box.h);
    }
    ctx.restore();

    if (lock && hud.state.input) {
      drawModelInput(ctx, lock.gray, INPUT_SIZE, lock.inputQuad, MODEL_VIEW_PX);
    }
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
    hud.status("model", "loading");
    const loading = createDetector(viteAssets).then(
      (detector) => {
        hud.status("model", "ready");
        return detector;
      },
      (err: unknown) => {
        hud.status("model", `unavailable (${errorMessage(err)})`);
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
    const tracker = await createHandTracker(viteAssets).catch(() => null);
    const labelCorners = startLoop(
      video,
      canvas,
      ctx,
      calibration,
      hud,
      await loading,
      tracker,
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
