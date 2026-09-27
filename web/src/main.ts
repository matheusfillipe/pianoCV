import type {
  HandLandmarkerResult,
  ImageSegmenterResult,
} from "@mediapipe/tasks-vision";
import { createBoardReader } from "./boardreader";
import { type Calibration, type Corners, createCalibration } from "./calibrate";
import {
  createDetector,
  type Detection,
  type Detector,
  INPUT_SIZE,
  MODEL_URL,
} from "./detector";
import { drawHands, drawKeys, drawModelInput, drawQuad } from "./draw";
import { createEffects, lowestPitchFor } from "./effects";
import { createFollower } from "./follow";
import { createHandTracker, type HandTracker } from "./hands";
import type { Point } from "./homography";
import { createHud, type Hud } from "./hud";
import { createKeyMatcher, KEYMATCH_URL, type KeyMatcher } from "./keymatch";
import {
  convexHull,
  createKeySegmenter,
  KEYSEG_URL,
  type KeyRegion,
  type KeySegmenter,
  snapKeys,
} from "./keyseg";
import { keybedSpace } from "./keyspace";
import {
  createKeyReader,
  type DetectedKey,
  type FarEdge,
  type KeyRead,
  outlineQuad,
  PLAIN_OUTLINE,
  projectKeyFaces,
  trimFarEdge,
} from "./keystrip";
import { createLab } from "./lab";
import { createLabeller } from "./labeller";
import { depthInKeyWidths } from "./measure";
import {
  createOcclusionMask,
  drawOcclusion,
  type OcclusionMask,
} from "./occlusion";
import { canonicalQuad, setCameraFocal, setKeybedDepth } from "./pose";
import {
  createTracker,
  type Measured,
  type Reading,
  readsToHold,
  type TrackerState,
} from "./tracker";
import { viteAssets } from "./viteassets";

declare global {
  interface Window {
    // the tracker's own frozen corners next to what the follower drew from them, for the lab
    // to read off how far one has drifted from the other
    pianocvHeldQuad?: Point[];
    pianocvFollowedQuad?: Point[];
    // what the live segmenter last found, for the lab to read off a live page
    pianocvLiveKeys?: {
      readonly outline: Point[] | null;
      readonly regions: number;
      readonly current: boolean;
    };
  }
}

// the site can be served under a path, as on a project page, and the models sit beside it
function modelUrl(file: string): string {
  return `${import.meta.env.BASE_URL}${file.replace(/^\//, "")}`;
}

const MANUAL_COLOR = "rgba(56,189,248,0.9)";
const AUTO_COLOR = "#4ade80";
const MODEL_VIEW_PX = 216;
const TRUSTED_BOARD_CONFIDENCE = 0.5;
const DEPTH_KEY = "pianocv.keybedDepthUnits.v2";
const FOCAL_KEY = "pianocv.cameraFocalFraction.v2";

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

// what the HUD's status line says about the keys found in the rectified strip: how many white
// keys and how sure of it, or why it is still unsure, once it has had a look at the held keybed
function keyStatusText(read: KeyRead | null): string {
  if (read === null) {
    return "reading";
  }
  if (read.kind === "unsure") {
    return read.reason;
  }
  return `${read.whiteKeys} white keys from ${read.phase}, confidence ${(read.confidence * 100).toFixed(0)}%, black keys ${read.blackRaiseMm} mm up, strip ${read.stripKeys.toFixed(1)} keys wide`;
}

// the template draws a black key as its raised top and its front face; to match it against one
// segmented key we take the outline of both
type DrawnKey = {
  readonly black: boolean;
  readonly semitone: number;
  readonly bar: readonly Point[];
};

function oneFacePerKey(faces: readonly DetectedKey[]): DrawnKey[] {
  const keys: DrawnKey[] = [];
  for (let i = 0; i < faces.length; i += 1) {
    const face = faces[i];
    const front = faces[i + 1];
    if (face.black && front?.black && front.semitone === face.semitone) {
      keys.push({ ...face, bar: convexHull([...face.bar, ...front.bar]) });
      i += 1;
    } else {
      keys.push(face);
    }
  }
  return keys;
}

function startLoop(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  calibration: Calibration,
  hud: Hud,
  detector: Detector | null,
  handTracker: HandTracker | null,
  occlusionMask: OcclusionMask | null,
  keyMatcher: KeyMatcher | null,
  keySegmenter: KeySegmenter | null,
): () => Corners {
  let hands: HandLandmarkerResult | null = null;
  let lastVideoTime = -1;
  let segmented: ImageSegmenterResult | null = null;
  let lastSegmentVideoTime = -1;
  // kept only so the "input" HUD toggle can still show the model's own view; the tracker
  // itself never exposes a detection's raw gray image
  let lastDetection: Detection | null = null;
  const effects = createEffects();

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
  const keyReader = createKeyReader(keyMatcher, keySegmenter);
  const labeller = createLabeller();
  const follower = createFollower();
  let boardHeld: TrackerState | null = null;
  // identity of the held state and the last detection read, so a fresh model confirmation
  // (a new hold, or the same hold re-verified) is what resets the follower, not every frame
  let followedFrom: TrackerState | null = null;
  let followedReading: Reading | null = null;
  type Followed = { kind: "held"; quad: Point[]; byHand: boolean };
  let labelState: Followed | null = null;
  // the keybed outline the segmenter found on the latest frame it finished, which the keys are
  // drawn on so they follow the real keyboard every frame
  let liveOutline: Point[] | null = null;
  let liveRegions: readonly KeyRegion[] = [];
  let segmenting = false;

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
    const state = tracker?.state();
    let trimmed: FarEdge | null = null;
    if (state?.kind === "held") {
      const size = { width: video.videoWidth, height: video.videoHeight };
      const reading = tracker?.reading() ?? null;
      // a fresh labelState object marks a new hold for the labeller's own session
      // bookkeeping; a mere reconfirmation of the same hold must not look like a new one,
      // so it only ever updates the existing object's quad in place
      const isNewHold = state !== followedFrom;
      if (isNewHold) {
        followedFrom = state;
        labelState = {
          kind: "held",
          quad: [...state.quad],
          byHand: state.byHand,
        };
      }
      if (state !== boardHeld) {
        boardHeld = state;
        boardReader.moved(now);
        keyReader.moved(now);
        liveOutline = null;
        liveRegions = [];
      }
      const space = keybedSpace({ quad: state.quad }, size);
      if (space) {
        boardReader.look(space, video, size, now);
      }
      keyReader.look(video, state.quad, size, now);
      if (isNewHold || reading !== followedReading) {
        follower.reset(state.quad, video, size);
        followedReading = reading;
        if (labelState) {
          labelState.quad = [...state.quad];
        }
      } else if (labelState) {
        const moved = follower.update(video, size);
        if (moved) {
          labelState.quad = [...moved];
        }
      }
      // the far edge the model (or a hand) drew reaches into the case; the key detector already
      // measured how far, so drawing, key projection and auto-labelling all trim onto the keys
      trimmed = keyReader.fraction();
      if (labelState && trimmed !== null) {
        labelState.quad = trimFarEdge(labelState.quad, trimmed);
      }
      hud.status(
        "trim",
        trimmed === null
          ? "keep still to measure"
          : `${(trimmed.left * 100).toFixed(0)}% · ${(trimmed.right * 100).toFixed(0)}%`,
      );
      hud.status(
        "follow",
        `${(window.pianocvFollowMs?.at(-1) ?? 0).toFixed(2)} ms`,
      );
      window.pianocvHeldQuad = state.quad;
      window.pianocvFollowedQuad = labelState?.quad ?? state.quad;
      const found = keyReader.last();
      const followed = labelState?.quad ?? state.quad;
      drawQuad(
        ctx,
        liveOutline ??
          (found?.kind === "read"
            ? outlineQuad(followed, found.outline)
            : followed),
        box.w,
        box.h,
        AUTO_COLOR,
        "keybed",
      );
      // a board read this unsure has been wrong about the key count on our recordings, and keys
      // drawn off by one would light the wrong note, so we draw nothing until it is surer
      const trusted =
        found?.kind === "read" && found.confidence >= TRUSTED_BOARD_CONFIDENCE
          ? found
          : null;
      if (keySegmenter && trusted && !segmenting) {
        segmenting = true;
        const heldAt = state;
        void keySegmenter
          .segment(video, size, liveOutline ?? followed)
          .catch(() => null)
          .then((found) => {
            segmenting = false;
            if (boardHeld === heldAt) {
              liveOutline = found?.outline ?? null;
              liveRegions = found?.regions ?? [];
            }
            window.pianocvLiveKeys = {
              outline: liveOutline,
              regions: liveRegions.length,
              current: boardHeld === heldAt,
            };
          });
      }
      const faces =
        trusted && (hud.state.keys || hud.state.glow)
          ? liveOutline
            ? projectKeyFaces(
                { ...trusted, outline: PLAIN_OUTLINE },
                liveOutline,
                size,
              )
            : projectKeyFaces(trusted, followed, size)
          : [];
      const keys = liveOutline
        ? snapKeys(oneFacePerKey(faces), liveRegions)
        : faces;
      if (hud.state.keys) {
        drawKeys(ctx, keys, box.w, box.h);
      }
      if (hud.state.glow) {
        const lowest = trusted
          ? lowestPitchFor(trusted.whiteKeys, trusted.phase)
          : null;
        effects.draw(ctx, keys, lowest, box.w, box.h, now);
        if (segmented) {
          drawOcclusion(ctx, video, segmented, box.w, box.h);
        }
      }
    } else {
      boardHeld = null;
      followedFrom = null;
      followedReading = null;
      labelState = null;
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
        hud.status("board", keyStatusText(keyReader.last()));
      }
      hud.status("label", `${labeller.saved()} saved`);
      // a label is only worth saving once the far edge has been measured onto the keys; until
      // then the corners still reach into the case, and a wrong label outlives the session
      if (state.kind !== "held" || trimmed !== null) {
        labeller.look(
          video,
          state.kind === "held" ? (labelState ?? state) : state,
          boardReader.last(),
          hud.state.label,
          now,
        );
      }
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
    const loading = createDetector(viteAssets, modelUrl(MODEL_URL)).then(
      (detector) => {
        hud.status("model", "ready");
        return detector;
      },
      (err: unknown) => {
        hud.status("model", `unavailable (${errorMessage(err)})`);
        return null;
      },
    );
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
    const matcherLoading = createKeyMatcher(
      viteAssets,
      modelUrl(KEYMATCH_URL),
    ).then(
      (matcher) => {
        hud.status("keys", "matcher ready");
        return matcher;
      },
      () => {
        hud.status("keys", "matcher unavailable, reading keys by brightness");
        return null;
      },
    );
    const segmenterLoading = createKeySegmenter(
      viteAssets,
      modelUrl(KEYSEG_URL),
    ).then(
      (segmenter) => {
        hud.status("outline", `key segmenter ready on ${segmenter.backend}`);
        return segmenter;
      },
      () => {
        hud.status(
          "outline",
          "key segmenter unavailable, keys read on the mask outline",
        );
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
    const occlusionMask = await occlusionLoading;
    const labelCorners = startLoop(
      video,
      canvas,
      ctx,
      calibration,
      hud,
      await loading,
      handTracker,
      occlusionMask,
      await matcherLoading,
      await segmenterLoading,
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
