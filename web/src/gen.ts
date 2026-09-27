import {
  ACESFilmicToneMapping,
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Group,
  Mesh,
  MeshStandardMaterial,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type BackdropKind, makeBackdrop } from "./backdrop";
import { drawQuad } from "./draw";
import { makeEnvironments } from "./environments";
import type { Point } from "./homography";
import { styleButton } from "./hud";
import {
  type CaseBox,
  type CaseLayout,
  caseLayout,
  centreFor,
  cornersFor,
  type KeybedCrop,
  panelDetails,
  projectCorners,
  visibleFraction,
} from "./keybed3d";
import {
  buildKeyboard,
  type Keyboard,
  type KeyGeometry,
  type KeyVariation,
  sampleKeyVariation,
} from "./procedural-keys";
import {
  BOARD_SIZES,
  type BoardSize,
  between,
  CASE_BACK_DEPTH_MM,
  CASE_CHEEK_WIDTH_MM,
  CASE_PRESENCE_PROBABILITY,
  type CaseColorFamily,
  noteName,
  pickBoardSize,
  pickCaseColorFamily,
  SWEEP_FOV_DEG,
  samplePose,
} from "./sweep";

const WIDTH = 640;
const HEIGHT = 480;
const PAGE_PARAMS = new URLSearchParams(location.search);
// ?out=synth-case opts into the case-dressed corpus, ?out=synth-keys into the per-key labelled
// one; every other value keeps the default
const SYNTH_DIR: "synth" | "synth-case" | "synth-keys" =
  PAGE_PARAMS.get("out") === "synth-keys"
    ? "synth-keys"
    : PAGE_PARAMS.get("out") === "synth-case"
      ? "synth-case"
      : "synth";
// ?elevation=min,max and ?azimuth=min,max narrow the sweep, the azimuth as a magnitude either
// side of head on, so a batch can cover the views the rest of the corpus has too few of
function rangeParam(name: string): { min: number; max: number } | null {
  const [min, max] = (PAGE_PARAMS.get(name) ?? "").split(",").map(Number);
  return Number.isFinite(min) && Number.isFinite(max) ? { min, max } : null;
}
const ELEVATION_RANGE = rangeParam("elevation");
const AZIMUTH_RANGE = rangeParam("azimuth");
// ?frames=N starts the auto sweep unattended, for the headless generator runner
const AUTOSTART_FRAMES = Number(PAGE_PARAMS.get("frames"));
const CAPTURE_INTERVAL_MS = 300;
// enough of the keybed to be worth a label; below this there is nothing to learn from
const MIN_VISIBLE_FRACTION = 0.2;
const DEFAULT_SWEEP_TARGET = 500;
const FULL_BOARD: BoardSize =
  BOARD_SIZES[BOARD_SIZES.length - 1] ?? BOARD_SIZES[0];
const MAX_KEYS = Math.max(...BOARD_SIZES.map((board) => board.keys));
// the deterministic reference board the "render test grid" mode holds steady across every pose,
// so a change in the detector is attributable to the pose rather than to a shifting keyboard
const DEFAULT_KEY_VARIATION: KeyVariation = sampleKeyVariation(() => 0.5);
const GRID_KEYBOARD: Keyboard = buildKeyboard(
  { lowestPitch: FULL_BOARD.range.lowest, keys: FULL_BOARD.keys },
  DEFAULT_KEY_VARIATION,
);
const BACKDROPS: BackdropKind[] = ["clutter", "noise", "gradient"];

interface CaseState {
  readonly present: boolean;
  readonly colorFamily: CaseColorFamily | null;
}
// a fixed lattice of poses, rendered under a seeded generator, so a change in the detector is
// attributable to a pose rather than lost in a random sweep
const GRID_ELEVATION = [10, 25, 40, 55, 70];
const GRID_AZIMUTH = [-90, -60, -30, 0, 30, 60, 90];
const GRID_BOARD_SPAN = GRID_KEYBOARD.maxZ - GRID_KEYBOARD.minZ;
const GRID_DISTANCE = [
  GRID_BOARD_SPAN * 0.5,
  GRID_BOARD_SPAN * 0.8,
  GRID_BOARD_SPAN * 1.2,
];
const GRID_FOV = 45;
const GRID_SEED = 7;

interface Pose {
  elevation: number;
  azimuth: number;
  distance: number;
}

function gridPoses(): Pose[] {
  const poses: Pose[] = [];
  for (const elevation of GRID_ELEVATION) {
    for (const azimuth of GRID_AZIMUTH) {
      for (const distance of GRID_DISTANCE) {
        poses.push({ elevation, azimuth, distance });
      }
    }
  }
  return poses;
}

// mulberry32: a tiny seedable generator so a grid render is the same run to run
function seeded(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function stamp(): string {
  const now = new Date();
  const pad = (v: number): string => String(v).padStart(2, "0");
  return (
    `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-` +
    `${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}${pad(now.getMilliseconds() / 10)}`
  );
}

async function post(
  name: string,
  body: Blob | string,
  type?: string,
  directory = "synth",
): Promise<void> {
  const response = await fetch(`/lab/save/${directory}/${name}`, {
    method: "POST",
    body,
    headers: type ? { "Content-Type": type } : undefined,
  });
  if (!response.ok) {
    throw new Error(`save failed: ${response.status}`);
  }
}

export async function boot(): Promise<void> {
  const layer: Partial<CSSStyleDeclaration> = {
    position: "fixed",
    inset: "0",
    margin: "auto",
    width: "min(100vw, 133vh)",
    height: "min(75vw, 100vh)",
  };
  const canvas = document.createElement("canvas");
  canvas.width = WIDTH;
  canvas.height = HEIGHT;
  Object.assign(canvas.style, layer);
  document.body.appendChild(canvas);

  const overlay = document.createElement("canvas");
  overlay.width = WIDTH;
  overlay.height = HEIGHT;
  Object.assign(overlay.style, layer, { pointerEvents: "none" });
  document.body.appendChild(overlay);
  const overlayCtx = overlay.getContext("2d");
  if (!overlayCtx) {
    throw new Error("2d canvas context unavailable");
  }

  const renderer = new WebGLRenderer({
    canvas,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setSize(WIDTH, HEIGHT, false);
  renderer.outputColorSpace = SRGBColorSpace;
  renderer.toneMapping = ACESFilmicToneMapping;

  const scene = new Scene();
  const pmrem = new PMREMGenerator(renderer);
  const environments = makeEnvironments(pmrem);
  scene.environment = environments[0];

  const camera = new PerspectiveCamera(45, WIDTH / HEIGHT, 0.1, 500);
  camera.position.set(6, 14, 14);
  const controls = new OrbitControls(camera, canvas);
  controls.target.copy(centreFor(GRID_KEYBOARD.minZ, GRID_KEYBOARD.maxZ));
  controls.update();

  const key = new DirectionalLight(0xffffff, 2);
  const fill = new DirectionalLight(0xffffff, 1);
  const ambient = new AmbientLight(0xffffff, 0.4);
  scene.add(key, fill, ambient);

  // a real instrument is a case around the keys: a back panel behind the black keys' back end,
  // end cheeks at both ends, a front rail under the white keys' front edge, and a body
  // underneath. A floating slab gives the net no way to learn where the keybed actually ends.
  const body = new Group();
  const caseMaterial = new MeshStandardMaterial();
  const caseUnitBox = new BoxGeometry(1, 1, 1);
  const backPanelMesh = new Mesh(caseUnitBox, caseMaterial);
  const cheekLowMesh = new Mesh(caseUnitBox, caseMaterial);
  const cheekHighMesh = new Mesh(caseUnitBox, caseMaterial);
  const frontRailMesh = new Mesh(caseUnitBox, caseMaterial);
  const caseBodyMesh = new Mesh(caseUnitBox, caseMaterial);
  body.add(
    backPanelMesh,
    cheekLowMesh,
    cheekHighMesh,
    frontRailMesh,
    caseBodyMesh,
  );

  const MAX_PANEL_DETAILS = 8;
  const panelDetailMaterials = Array.from(
    { length: MAX_PANEL_DETAILS },
    () => new MeshStandardMaterial(),
  );
  const panelDetailMeshes = panelDetailMaterials.map(
    (material) => new Mesh(caseUnitBox, material),
  );
  body.add(...panelDetailMeshes);
  scene.add(body);

  const placeBox = (mesh: Mesh, box: CaseBox): void => {
    mesh.scale.set(box.size[0], box.size[1], box.size[2]);
    mesh.position.set(box.center[0], box.center[1], box.center[2]);
  };

  // the keyboard itself: a pool of body+bevel-cap mesh pairs sized for the largest standard
  // board, toggled visible per key so a smaller board simply leaves the tail of the pool hidden
  const whiteKeyMaterial = new MeshStandardMaterial({ color: 0xf1efe6 });
  const blackKeyMaterial = new MeshStandardMaterial({ color: 0x0a0a0a });
  const keyMaterials = [whiteKeyMaterial, blackKeyMaterial];
  const keyBodyMeshes = Array.from(
    { length: MAX_KEYS },
    () => new Mesh(caseUnitBox, whiteKeyMaterial),
  );
  const keyCapMeshes = Array.from(
    { length: MAX_KEYS },
    () => new Mesh(caseUnitBox, whiteKeyMaterial),
  );
  const keysGroup = new Group();
  keysGroup.add(...keyBodyMeshes, ...keyCapMeshes);
  scene.add(keysGroup);

  const applyKeyboard = (keyboard: Keyboard): void => {
    for (let i = 0; i < MAX_KEYS; i += 1) {
      const key: KeyGeometry | undefined = keyboard.keys[i];
      const bodyMesh = keyBodyMeshes[i];
      const capMesh = keyCapMeshes[i];
      bodyMesh.visible = key !== undefined;
      capMesh.visible = key !== undefined;
      if (!key) {
        continue;
      }
      const material = key.black ? blackKeyMaterial : whiteKeyMaterial;
      bodyMesh.material = material;
      capMesh.material = material;
      placeBox(bodyMesh, key.body);
      placeBox(capMesh, key.cap);
    }
  };

  const panel = document.createElement("div");
  Object.assign(panel.style, {
    position: "fixed",
    top: "12px",
    left: "12px",
    zIndex: "10",
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "10px 12px",
    background: "rgba(8,8,10,0.78)",
    border: "1px solid rgba(229,229,229,0.12)",
    borderRadius: "10px",
    color: "#e5e5e5",
    font: "12px/1.5 ui-sans-serif, system-ui, sans-serif",
  });
  const record = document.createElement("button");
  const sweep = document.createElement("button");
  const sweepCount = document.createElement("input");
  const grid = document.createElement("button");
  const shuffle = document.createElement("button");
  const readout = document.createElement("span");
  record.textContent = "record";
  sweep.textContent = "auto sweep";
  sweepCount.type = "number";
  sweepCount.min = "1";
  sweepCount.value = String(DEFAULT_SWEEP_TARGET);
  sweepCount.title = "frames to generate before auto sweep stops itself";
  Object.assign(sweepCount.style, {
    width: "64px",
    background: "rgba(255,255,255,0.05)",
    border: "1px solid rgba(229,229,229,0.18)",
    borderRadius: "6px",
    color: "#d4d4d4",
    font: "inherit",
    padding: "3px 6px",
  });
  grid.textContent = "render test grid";
  shuffle.textContent = "randomise";
  styleButton(record);
  styleButton(sweep);
  styleButton(grid);
  styleButton(shuffle);
  readout.style.color = "#8a8a8a";
  panel.append(record, sweep, sweepCount, grid, shuffle, readout);
  document.body.appendChild(panel);
  let random: () => number = Math.random;
  const status = (text: string): void => {
    readout.textContent = text;
  };

  let currentCaseLayout: CaseLayout = caseLayout(
    { minZ: GRID_KEYBOARD.minZ, maxZ: GRID_KEYBOARD.maxZ },
    CASE_BACK_DEPTH_MM.min,
    CASE_CHEEK_WIDTH_MM.min,
  );
  let currentCase: CaseState = { present: false, colorFamily: null };

  const layoutCase = (crop: KeybedCrop): void => {
    currentCaseLayout = caseLayout(
      crop,
      between(random, CASE_BACK_DEPTH_MM),
      between(random, CASE_CHEEK_WIDTH_MM),
    );
    placeBox(backPanelMesh, currentCaseLayout.backPanel);
    placeBox(cheekLowMesh, currentCaseLayout.cheekLow);
    placeBox(cheekHighMesh, currentCaseLayout.cheekHigh);
    placeBox(frontRailMesh, currentCaseLayout.frontRail);
    placeBox(caseBodyMesh, currentCaseLayout.body);
  };
  layoutCase({ minZ: GRID_KEYBOARD.minZ, maxZ: GRID_KEYBOARD.maxZ });

  const randomiseLens = (): void => {
    camera.fov = between(random, SWEEP_FOV_DEG);
    camera.updateProjectionMatrix();
  };

  const randomise = (): void => {
    scene.background = makeBackdrop(
      BACKDROPS[Math.floor(random() * BACKDROPS.length)],
      random,
    );
    key.position.set(random() * 20 - 10, 5 + random() * 20, random() * 20 - 10);
    key.intensity = 0.8 + random() * 3;
    key.color = new Color().setHSL(
      random(),
      0.15 * random(),
      0.5 + random() * 0.5,
    );
    fill.position.set(
      random() * 20 - 10,
      2 + random() * 10,
      random() * 20 - 10,
    );
    fill.intensity = random() * 1.5;
    ambient.intensity = 0.1 + random() * 0.7;
    scene.environment =
      environments[Math.floor(random() * environments.length)];
    scene.environmentIntensity = 0.15 + random() * 1.9;
    scene.environmentRotation.y = random() * Math.PI * 2;
    renderer.toneMappingExposure = 0.55 + random() * 1.15;
    for (const material of keyMaterials) {
      // key plastic runs from matte to near mirror, and the reflection is what a flat
      // brightness jitter can never fake
      material.roughness = 0.04 + random() * random() * 0.85;
      material.metalness = random() * 0.35;
      material.envMapIntensity = 0.3 + random() * 2.2;
      material.needsUpdate = true;
    }
    // present in most frames so the model mostly sees a case, and sometimes just floating keys,
    // the way it looked before a case existed at all
    const present = random() < CASE_PRESENCE_PROBABILITY;
    body.visible = present;
    if (present) {
      const family = pickCaseColorFamily(random);
      const hue = between(random, family.hue);
      const saturation = between(random, family.saturation);
      const lightness = between(random, family.lightness);
      caseMaterial.color.setHSL(hue, saturation, lightness);
      caseMaterial.roughness = between(random, family.roughness);
      caseMaterial.metalness = between(random, family.metalness);
      caseMaterial.envMapIntensity = 0.3 + random() * 2.0;
      caseMaterial.needsUpdate = true;
      const details = panelDetails(currentCaseLayout, random);
      for (let i = 0; i < panelDetailMeshes.length; i += 1) {
        const detail = details[i];
        panelDetailMeshes[i].visible = detail !== undefined;
        if (!detail) {
          continue;
        }
        placeBox(panelDetailMeshes[i], detail);
        panelDetailMaterials[i].color.setHSL(
          hue,
          saturation * 0.5,
          Math.min(1, lightness + detail.lighter),
        );
        panelDetailMaterials[i].needsUpdate = true;
      }
      currentCase = { present: true, colorFamily: family.family };
    } else {
      currentCase = { present: false, colorFamily: null };
    }
  };
  randomise();
  randomiseLens();

  let recording = false;
  let sweeping = false;
  let saved = 0;
  let sweepTarget = 0;
  let lastCapture = 0;
  const poses = gridPoses();
  let gridIndex = -1;
  let gridSkipped = 0;
  let currentKeyboard: Keyboard = GRID_KEYBOARD;
  applyKeyboard(currentKeyboard);
  // rotateZ never happens outside placeCamera, so this stays an accurate 0 for manual orbiting too
  let currentRollDeg = 0;

  // reads the pose straight back out of the camera, so a manually orbited shot is described as
  // faithfully as a swept one, and an off-centre nudge after aim() is reflected too
  const cameraPoseDeg = (centre: { x: number; y: number; z: number }): Pose => {
    const dx = camera.position.x - centre.x;
    const dy = camera.position.y - centre.y;
    const dz = camera.position.z - centre.z;
    const distance = Math.hypot(dx, dy, dz);
    return {
      elevation: distance > 0 ? (Math.asin(dy / distance) * 180) / Math.PI : 0,
      azimuth: (Math.atan2(dz, dx) * 180) / Math.PI,
      distance,
    };
  };

  const aim = (
    pose: Pose,
    centre = centreFor(GRID_KEYBOARD.minZ, GRID_KEYBOARD.maxZ),
  ): void => {
    const elevation = (pose.elevation * Math.PI) / 180;
    const azimuth = (pose.azimuth * Math.PI) / 180;
    // the keys run along z, so azimuth has to swing around the front (+x) face; putting
    // azimuth 0 on +z aims the camera down the length of the keybed and foreshortens it away
    camera.position.set(
      centre.x + pose.distance * Math.cos(elevation) * Math.cos(azimuth),
      centre.y + pose.distance * Math.sin(elevation),
      centre.z + pose.distance * Math.cos(elevation) * Math.sin(azimuth),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(centre);
  };

  // a scripted camera covers the pose, lens and board-size space evenly and without anybody
  // having to orbit by hand; ranges live in ./sweep so the coverage is legible in one place
  const placeCamera = (): void => {
    const board = pickBoardSize(random);
    const variation = sampleKeyVariation(random);
    currentKeyboard = buildKeyboard(
      { lowestPitch: board.range.lowest, keys: board.keys },
      variation,
    );
    applyKeyboard(currentKeyboard);
    layoutCase({ minZ: currentKeyboard.minZ, maxZ: currentKeyboard.maxZ });
    const sampled = samplePose(random);
    const sample = {
      ...sampled,
      elevationDeg: ELEVATION_RANGE
        ? between(random, ELEVATION_RANGE)
        : sampled.elevationDeg,
      azimuthDeg: AZIMUTH_RANGE
        ? (random() < 0.5 ? -1 : 1) * between(random, AZIMUTH_RANGE)
        : sampled.azimuthDeg,
    };
    const boardSpan = currentKeyboard.maxZ - currentKeyboard.minZ;
    currentRollDeg = sample.rollDeg;
    camera.fov = sample.fovDeg;
    camera.updateProjectionMatrix();
    aim(
      {
        elevation: sample.elevationDeg,
        azimuth: sample.azimuthDeg,
        distance: boardSpan * sample.distanceFactor,
      },
      centreFor(currentKeyboard.minZ, currentKeyboard.maxZ),
    );
    camera.rotateZ((sample.rollDeg * Math.PI) / 180);
    // the board sits off centre and sometimes drifts partly out of frame, the way a hand-held
    // shot does
    camera.translateX(sample.offsetXFactor * boardSpan);
    camera.translateY(sample.offsetYFactor * boardSpan);
  };

  const placeGrid = (): void => {
    aim(poses[gridIndex]);
    camera.fov = GRID_FOV;
    camera.updateProjectionMatrix();
    randomise();
  };

  const stopGrid = (): void => {
    gridIndex = -1;
    random = Math.random;
    controls.enabled = true;
    styleButton(grid, false);
  };

  const stopSweep = (): void => {
    sweeping = false;
    recording = false;
    sweepTarget = 0;
    controls.enabled = true;
    styleButton(sweep, false);
    styleButton(record, false);
  };

  const still = document.createElement("canvas");
  still.width = WIDTH;
  still.height = HEIGHT;
  const stillCtx = still.getContext("2d");
  if (!stillCtx) {
    throw new Error("2d canvas context unavailable");
  }

  interface CapturePose {
    elevation: number;
    azimuth: number;
    distance: number;
    rollDeg: number;
    fovDeg: number;
  }

  interface CaptureBoard {
    keys: number;
    whiteKeys: number;
    lowestPitch: number;
    lowestNote: string;
  }

  interface KeyLabel {
    pitch: number;
    black: boolean;
    top: Point[];
    front: Point[] | null;
  }

  interface CaptureMeta {
    pose: CapturePose;
    board: CaptureBoard;
    keyGeometry: KeyVariation;
    keys: KeyLabel[];
    caseState: CaseState;
  }

  const projectKeyLabels = (keyboard: Keyboard): KeyLabel[] =>
    keyboard.keys.map((k) => ({
      pitch: k.pitch,
      black: k.black,
      top: projectCorners(camera, k.topCorners),
      front: k.frontCorners ? projectCorners(camera, k.frontCorners) : null,
    }));

  // the frame is copied and encoded synchronously: an async toBlob can race the next render
  // and write a png that does not match the corners saved beside it
  const capture = (
    corners: Point[],
    meta: CaptureMeta,
    directory: "synth" | "synth-case" | "synth-keys" | "grid",
  ): void => {
    stillCtx.drawImage(canvas, 0, 0);
    const url = still.toDataURL("image/png");
    const binary = atob(url.slice(url.indexOf(",") + 1));
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    const blob = new Blob([bytes], { type: "image/png" });
    const name =
      directory === "grid"
        ? `grid-e${meta.pose.elevation}-a${meta.pose.azimuth}-d${Math.round(meta.pose.distance)}`
        : `synth-${stamp()}`;
    post(`${name}.png`, blob, undefined, directory)
      .then(() =>
        post(
          `${name}.json`,
          JSON.stringify({
            kind: "synth",
            startedAt: Date.now(),
            durationMs: 0,
            corners,
            imageWidth: WIDTH,
            imageHeight: HEIGHT,
            mimeType: "image/png",
            pose: meta.pose,
            board: meta.board,
            keyGeometry: meta.keyGeometry,
            keys: meta.keys,
            case: meta.caseState.present,
            caseColor: meta.caseState.colorFamily,
          }),
          "application/json",
          directory,
        ),
      )
      .then(() => {
        saved += 1;
        if (sweepTarget > 0 && saved >= sweepTarget) {
          stopSweep();
        }
      })
      .catch((err: unknown) => {
        status(err instanceof Error ? err.message : String(err));
      });
  };

  shuffle.addEventListener("click", () => {
    randomise();
    randomiseLens();
  });
  sweep.addEventListener("click", () => {
    sweeping = !sweeping;
    controls.enabled = !sweeping;
    styleButton(sweep, sweeping);
    if (sweeping) {
      recording = true;
      styleButton(record, true);
      const requested = Math.floor(Number(sweepCount.value));
      sweepTarget =
        saved +
        (Number.isFinite(requested)
          ? Math.max(1, requested)
          : DEFAULT_SWEEP_TARGET);
    } else {
      sweepTarget = 0;
    }
  });
  record.addEventListener("click", () => {
    recording = !recording;
    styleButton(record, recording);
  });
  grid.addEventListener("click", () => {
    if (gridIndex >= 0) {
      stopGrid();
      return;
    }
    stopSweep();
    controls.enabled = false;
    random = seeded(GRID_SEED);
    gridIndex = 0;
    gridSkipped = 0;
    currentKeyboard = GRID_KEYBOARD;
    applyKeyboard(currentKeyboard);
    layoutCase({ minZ: currentKeyboard.minZ, maxZ: currentKeyboard.maxZ });
    styleButton(grid, true);
    placeGrid();
  });
  window.addEventListener("keydown", (event) => {
    if (event.repeat) {
      return;
    }
    if (event.key === "r") {
      recording = !recording;
      styleButton(record, recording);
    }
    if (event.key === " ") {
      randomise();
      randomiseLens();
    }
  });

  if (Number.isFinite(AUTOSTART_FRAMES) && AUTOSTART_FRAMES > 0) {
    sweepCount.value = String(AUTOSTART_FRAMES);
    sweep.click();
  }

  const frame = (now: number): void => {
    if (!sweeping) {
      controls.update();
    }
    renderer.render(scene, camera);
    const activeCorners = cornersFor(
      currentKeyboard.minZ,
      currentKeyboard.maxZ,
    );
    const corners = projectCorners(camera, activeCorners);
    const fraction = visibleFraction(camera, activeCorners);
    const usable = fraction >= MIN_VISIBLE_FRACTION;

    overlayCtx.clearRect(0, 0, WIDTH, HEIGHT);
    drawQuad(
      overlayCtx,
      corners,
      WIDTH,
      HEIGHT,
      usable ? "#4ade80" : "#f87171",
      "keybed",
    );
    status(
      `${Math.round(fraction * 100)}% visible${usable ? "" : ", too little to save"}  saved ${saved}${sweepTarget > 0 ? `/${sweepTarget}` : ""}`,
    );

    if (gridIndex >= 0) {
      // this frame shows the pose placed last tick, so it is captured before moving on
      if (usable) {
        const gridPose = poses[gridIndex];
        capture(
          corners,
          {
            pose: { ...gridPose, rollDeg: 0, fovDeg: GRID_FOV },
            board: {
              keys: currentKeyboard.board.keys,
              whiteKeys: currentKeyboard.whiteKeys,
              lowestPitch: currentKeyboard.board.lowestPitch,
              lowestNote: noteName(currentKeyboard.board.lowestPitch),
            },
            keyGeometry: currentKeyboard.variation,
            keys: projectKeyLabels(currentKeyboard),
            caseState: currentCase,
          },
          "grid",
        );
      } else {
        gridSkipped += 1;
      }
      status(`grid ${gridIndex + 1}/${poses.length}  skipped ${gridSkipped}`);
      gridIndex += 1;
      if (gridIndex >= poses.length) {
        stopGrid();
      } else {
        placeGrid();
      }
      requestAnimationFrame(frame);
      return;
    }
    if (recording && usable && now - lastCapture > CAPTURE_INTERVAL_MS) {
      lastCapture = now;
      const pose = cameraPoseDeg(
        centreFor(currentKeyboard.minZ, currentKeyboard.maxZ),
      );
      capture(
        corners,
        {
          pose: { ...pose, rollDeg: currentRollDeg, fovDeg: camera.fov },
          board: {
            keys: currentKeyboard.board.keys,
            whiteKeys: currentKeyboard.whiteKeys,
            lowestPitch: currentKeyboard.board.lowestPitch,
            lowestNote: noteName(currentKeyboard.board.lowestPitch),
          },
          keyGeometry: currentKeyboard.variation,
          keys: projectKeyLabels(currentKeyboard),
          caseState: currentCase,
        },
        SYNTH_DIR,
      );
      randomise();
      if (sweeping) {
        randomiseLens();
      }
    }
    if (sweeping && (!usable || now - lastCapture < 1)) {
      placeCamera();
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

void boot();
