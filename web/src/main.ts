import type { HandLandmarkerResult } from "@mediapipe/tasks-vision";
import type { BoardRead } from "./board";
import { createBoardReader } from "./boardreader";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import {
  createDetector,
  type Detection,
  type Detector,
  INPUT_SIZE,
} from "./detector";
import { drawHands, drawModelInput, drawQuad } from "./draw";
import { createHandTracker, type HandTracker } from "./hands";
import { createHud, type Hud } from "./hud";
import { keybedSpace } from "./keyspace";
import { createLab } from "./lab";
import { createLabeller } from "./labeller";
import { depthInKeyWidths } from "./measure";
import { canonicalQuad, setCameraFocal, setKeybedDepth } from "./pose";
import {
  createTracker,
  type Measured,
  readsToHold,
  type TrackerState,
} from "./tracker";
import { viteAssets } from "./viteassets";

const MANUAL_COLOR = "rgba(56,189,248,0.9)";
const AUTO_COLOR = "#4ade80";
const MODEL_VIEW_PX = 216;
const DEPTH_KEY = "kvt.keybedDepthUnits.v2";
const FOCAL_KEY = "kvt.cameraFocalFraction.v2";

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

// what the HUD's status line says about the tracker: reading progress while it hunts, held
// once corners are fixed, or the reason it gave up
function trackerStatusText(state: TrackerState): string {
  if (state.kind === "hunting") {
    return `reading ${state.progress.agreed}/${readsToHold}`;
  }
  if (state.kind === "held") {
    return state.byHand ? "held (hand-placed)" : "held";
  }
  return `lost: ${state.reason}`;
}

// what the HUD's status line says about the board reader: how it found, or why it is still
// unsure, once it has had a look at the held keybed
function boardStatusText(read: BoardRead | null): string {
  if (read === null) {
    return "reading";
  }
  if (read.kind === "unsure") {
    return read.reason;
  }
  const keys = read.board.highest - read.board.lowest + 1;
  return `${keys} keys, agreement ${(read.agreement * 100).toFixed(0)}%`;
}

function startLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  hud: Hud,
  detector: Detector | null,
  handTracker: HandTracker | null,
): () => Corners {
  let hands: HandLandmarkerResult | null = null;
  let lastVideoTime = -1;
  // kept only so the "input" HUD toggle can still show the model's own view; the tracker
  // itself never exposes a detection's raw gray image
  let lastDetection: Detection | null = null;

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

  const onMeasured = (measured: Measured): void => {
    if (measured.kind === "depth") {
      remember(DEPTH_KEY, measured.units);
      hud.status(
        "keybed",
        `shape measured: ${depthInKeyWidths(measured.units).toFixed(2)} key widths per depth`,
      );
      return;
    }
    remember(FOCAL_KEY, measured.fraction);
    hud.status(
      "keybed",
      `lens measured: focal ${measured.fraction.toFixed(2)} of the frame width`,
    );
  };
  const capturing: Detector | null = detector && {
    detect: async (frame) => {
      const detection = await detector.detect(frame);
      lastDetection = detection;
      return detection;
    },
    reset: () => detector.reset(),
  };
  const tracker = capturing ? createTracker(capturing, { onMeasured }) : null;
  const boardReader = createBoardReader();
  const labeller = createLabeller();
  let boardHeld: TrackerState | null = null;

  hud.onRedetect(() => tracker?.release());
  hud.onAdopt(() => {
    const state = tracker?.state();
    if (state?.kind === "held") {
      calibration.setCorners(state.quad);
    }
  });
  // the corners are often placed perfectly but a half turn out, which reads as front and back
  // swapped; rolling by two relabels the same rectangle rather than making it be dragged again
  hud.onFlip(() => {
    const c = calibration.getCorners();
    calibration.setCorners([c[2], c[3], c[0], c[1]]);
  });

  // the dragged corners mean what was dragged: handle 1 to 2 runs along the black keys,
  // and the flip button is how the back is swapped. Deciding the back from the picture
  // every frame made the handles jump between the two orders on their own.
  const orientedManual = (): Corners =>
    canonicalQuad(calibration.getCorners()) as Corners;

  hud.onMeasure(() => tracker?.hold(orientedManual()));

  const frame = (now: number): void => {
    if (hud.state.live) {
      void tracker?.look(video, now);
    }
    if (handTracker && hud.state.hands && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hands = handTracker.detect(video, now);
    }
    ctx.fillStyle = "#050505";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const box = videoBox(video, canvas);
    ctx.drawImage(video, box.x, box.y, box.w, box.h);

    ctx.save();
    ctx.translate(box.x, box.y);
    const state = tracker?.state();
    if (state?.kind === "held") {
      drawQuad(ctx, state.quad, box.w, box.h, AUTO_COLOR, "keybed");
      if (state !== boardHeld) {
        boardHeld = state;
        boardReader.moved(now);
      }
      const size = { width: video.videoWidth, height: video.videoHeight };
      const space = keybedSpace({ quad: state.quad }, size);
      if (space) {
        boardReader.look(space, video, size, now);
      }
    } else {
      boardHeld = null;
    }
    if (hud.state.corners) {
      drawQuad(ctx, orientedManual(), box.w, box.h, MANUAL_COLOR, "manual");
      calibration.draw(ctx, box.w, box.h);
    }
    if (hands && hud.state.hands) {
      drawHands(ctx, hands, box.w, box.h);
    }
    ctx.restore();

    if (hud.state.input && lastDetection?.quad && lastDetection.inputQuad) {
      drawModelInput(
        ctx,
        lastDetection.gray,
        INPUT_SIZE,
        lastDetection.inputQuad,
        MODEL_VIEW_PX,
      );
    }

    if (state) {
      const reading = tracker?.reading() ?? null;
      hud.status(
        "detect",
        reading ? `${reading.latencyMs.toFixed(1)} ms` : "waiting",
      );
      hud.status("keybed", trackerStatusText(state));
      if (state.kind === "held") {
        hud.status("board", boardStatusText(boardReader.last()));
      }
      hud.status("label", `${labeller.saved()} saved`);
      labeller.look(video, state, boardReader.last(), hud.state.label, now);
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
    const handTracker = await createHandTracker(viteAssets).catch(() => null);
    const labelCorners = startLoop(
      video,
      canvas,
      ctx,
      calibration,
      hud,
      await loading,
      handTracker,
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
