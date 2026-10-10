import {
  ACESFilmicToneMapping,
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Group,
  Mesh,
  MeshStandardMaterial,
  PCFSoftShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  PMREMGenerator,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { type BackdropKind, makeBackdrop, makeFloorTexture } from "./backdrop";
import { drawQuad } from "./draw";
import { makeEnvironments } from "./environments";
import type { Point } from "./homography";
import { styleButton } from "./hud";
import {
  CASE_BODY_DEPTH_MM,
  type CaseBox,
  type CaseLayout,
  caseLayout,
  centreFor,
  cornersFor,
  cornersVisible,
  KEY_BOTTOM_Y,
  type KeybedCrop,
  mmToUnits,
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
  CASE_TOP_STANDOFF_MM,
  type CaseColorFamily,
  noteName,
  pickBoardSize,
  pickCaseColorFamily,
  type Range,
  SWEEP_AZIMUTH_DEG,
  SWEEP_DISTANCE_FACTOR,
  SWEEP_ELEVATION_DEG,
  SWEEP_FOV_DEG,
  SWEEP_ROLL_DEG,
  type SweepPose,
  samplePose,
} from "./sweep";
import {
  type MotionDir,
  motionDirOf,
  type StillDir,
  stillDirOf,
} from "./synth-dirs";

const WIDTH = 640;
const HEIGHT = 480;
const PAGE_PARAMS = new URLSearchParams(location.search);
const OUT_PARAM = PAGE_PARAMS.get("out") ?? "";
const SYNTH_DIR: StillDir = stillDirOf(OUT_PARAM) ?? "synth";
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
// ?motion=1 switches gen.html to rendering short camera-path sequences with motion blur into
// data/synth-motion, instead of single random-sweep stills
const MOTION_MODE = PAGE_PARAMS.get("motion") === "1";
const MOTION_DIR: MotionDir = motionDirOf(OUT_PARAM) ?? "synth-motion";
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
const FLOOR_PROBABILITY = 0.75;
const BACKDROPS: BackdropKind[] = ["clutter", "noise", "gradient"];

// a motion sequence covers a short hand-held move: pan, tilt, dolly and roll from a start pose
// to a nearby end pose, plus a small sinusoidal shake so fast frames blur convincingly
const MOTION_FRAME_COUNT: Range = { min: 24, max: 49 };
const MOTION_SUBFRAMES = 4;
const MOTION_EXPOSURE_SHARE: Range = { min: 0.2, max: 0.9 };
const MOTION_PAN_DEG: Range = { min: 5, max: 45 };
const MOTION_TILT_DEG: Range = { min: 3, max: 20 };
const MOTION_DOLLY_FACTOR: Range = { min: 0.75, max: 1.3 };
const MOTION_ROLL_DEG: Range = { min: 2, max: 12 };
const MOTION_SHAKE_DEG = 1.2;
const MOTION_SHAKE_CYCLES: Range = { min: 2, max: 6 };
// a sequence's keyboard and case are hidden this often, so the model also learns what a frame
// with no keyboard in it looks like
const MOTION_NEGATIVE_PROBABILITY = 0.1;

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

function clamp(value: number, range: Range): number {
  return Math.min(range.max, Math.max(range.min, value));
}

// the pose sampling placeCamera and motion sequences both start from: samplePose narrowed by
// the ?elevation=/?azimuth= page params, so a batch can cover the views the rest of the corpus
// has too few of
function sampleFramingPose(random: () => number): SweepPose {
  const sampled = samplePose(random);
  return {
    ...sampled,
    elevationDeg: ELEVATION_RANGE
      ? between(random, ELEVATION_RANGE)
      : sampled.elevationDeg,
    azimuthDeg: AZIMUTH_RANGE
      ? (random() < 0.5 ? -1 : 1) * between(random, AZIMUTH_RANGE)
      : sampled.azimuthDeg,
  };
}

function pngBlobFrom(source: HTMLCanvasElement): Blob {
  const url = source.toDataURL("image/png");
  const binary = atob(url.slice(url.indexOf(",") + 1));
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: "image/png" });
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
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = PCFSoftShadowMap;

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
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.camera.left = -20;
  key.shadow.camera.right = 20;
  key.shadow.camera.top = 20;
  key.shadow.camera.bottom = -20;
  key.shadow.camera.near = 1;
  key.shadow.camera.far = 80;
  key.shadow.bias = -0.0005;
  scene.add(key, fill, ambient);

  const floorMaterial = new MeshStandardMaterial({ roughness: 0.8 });
  const floorMesh = new Mesh(new PlaneGeometry(200, 200), floorMaterial);
  floorMesh.rotation.x = -Math.PI / 2;
  floorMesh.position.y = KEY_BOTTOM_Y - mmToUnits(CASE_BODY_DEPTH_MM);
  floorMesh.receiveShadow = true;
  scene.add(floorMesh);

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
  for (const mesh of [
    backPanelMesh,
    cheekLowMesh,
    cheekHighMesh,
    frontRailMesh,
    caseBodyMesh,
  ]) {
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  }
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
  const panelDetailMeshes = panelDetailMaterials.map((material) => {
    const mesh = new Mesh(caseUnitBox, material);
    mesh.receiveShadow = true;
    return mesh;
  });
  body.add(...panelDetailMeshes);
  scene.add(body);

  const caseOccluders = (): Mesh[] =>
    body.visible
      ? [
          backPanelMesh,
          cheekLowMesh,
          cheekHighMesh,
          frontRailMesh,
          caseBodyMesh,
          ...panelDetailMeshes,
        ]
      : [];

  const placeBox = (mesh: Mesh, box: CaseBox): void => {
    mesh.scale.set(box.size[0], box.size[1], box.size[2]);
    mesh.position.set(box.center[0], box.center[1], box.center[2]);
  };

  // the keyboard itself: a pool of body+bevel-cap mesh pairs sized for the largest standard
  // board, toggled visible per key so a smaller board simply leaves the tail of the pool hidden
  const keyMaterials = Array.from(
    { length: MAX_KEYS },
    () => new MeshStandardMaterial(),
  );
  const keyIsBlack: boolean[] = keyMaterials.map(() => false);
  const keyBodyMeshes = keyMaterials.map(
    (material) => new Mesh(caseUnitBox, material),
  );
  const keyCapMeshes = keyMaterials.map(
    (material) => new Mesh(caseUnitBox, material),
  );
  for (const mesh of [...keyBodyMeshes, ...keyCapMeshes]) {
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  }
  const keysGroup = new Group();
  keysGroup.add(...keyBodyMeshes, ...keyCapMeshes);
  scene.add(keysGroup);

  interface KeyLook {
    white: Color;
    black: Color;
    roughness: number;
    metalness: number;
    envMapIntensity: number;
    wear: number;
  }
  let keyLook: KeyLook = {
    white: new Color(0xf1efe6),
    black: new Color(0x0a0a0a),
    roughness: 0.4,
    metalness: 0,
    envMapIntensity: 1,
    wear: 0,
  };

  const paintKeys = (): void => {
    keyMaterials.forEach((material, i) => {
      const jitter = (random() - 0.5) * keyLook.wear;
      material.color
        .copy(keyIsBlack[i] ? keyLook.black : keyLook.white)
        .offsetHSL(0, 0, jitter);
      material.roughness = Math.min(
        1,
        Math.max(0, keyLook.roughness + jitter * 2),
      );
      material.metalness = keyLook.metalness;
      material.envMapIntensity = keyLook.envMapIntensity;
    });
  };

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
      keyIsBlack[i] = key.black;
      placeBox(bodyMesh, key.body);
      placeBox(capMesh, key.cap);
    }
    paintKeys();
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
      between(random, CASE_TOP_STANDOFF_MM),
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
    fill.color.setHSL(
      random(),
      random() < 0.3 ? 0.9 : 0.1,
      0.5 + random() * 0.4,
    );
    ambient.intensity = 0.1 + random() * 0.7;
    scene.environment =
      environments[Math.floor(random() * environments.length)];
    scene.environmentIntensity = 0.15 + random() * 1.9;
    scene.environmentRotation.y = random() * Math.PI * 2;
    renderer.toneMappingExposure = 0.55 + random() * 1.15;
    // ivory, pure white and grey-green whites, black plastic from jet to worn charcoal, and
    // plastic from matte to near mirror: the reflection is what a flat brightness jitter can
    // never fake
    keyLook = {
      white: new Color().setHSL(
        0.08 + random() * 0.12,
        random() * 0.35,
        0.62 + random() * 0.33,
      ),
      black: new Color().setHSL(
        random(),
        random() * 0.3,
        0.01 + random() * 0.14,
      ),
      roughness: 0.04 + random() * random() * 0.85,
      metalness: random() * 0.35,
      envMapIntensity: 0.3 + random() * 2.2,
      wear: random() < 0.5 ? 0 : random() * 0.12,
    };
    paintKeys();
    floorMesh.visible = random() < FLOOR_PROBABILITY;
    floorMaterial.map?.dispose();
    floorMaterial.map = makeFloorTexture(random);
    floorMaterial.roughness = 0.3 + random() * 0.7;
    floorMaterial.needsUpdate = true;
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
        const glowing = random() < 0.35;
        panelDetailMaterials[i].emissive.setHSL(random(), 1, glowing ? 0.5 : 0);
        panelDetailMaterials[i].emissiveIntensity = 0.6 + random() * 1.6;
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
    const sample = sampleFramingPose(random);
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
    cornerVisible: boolean[];
    pose: CapturePose;
    board: CaptureBoard;
    keyGeometry: KeyVariation;
    keys: KeyLabel[];
    caseState: CaseState;
  }

  interface MotionMeta {
    corners: Point[] | null;
    cornerVisible: boolean[];
    pose: CapturePose;
    board: CaptureBoard;
    keyGeometry: KeyVariation;
    keys: KeyLabel[];
    caseState: CaseState;
    piano: boolean;
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
    directory: StillDir | "grid",
  ): void => {
    stillCtx.drawImage(canvas, 0, 0);
    const blob = pngBlobFrom(still);
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
            cornerVisible: meta.cornerVisible,
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

  // one saved frame's png (the accumulated exposure) plus its sidecar, awaited so the next
  // frame's render never starts before this one has left the browser
  const captureMotionFrame = async (
    sequenceId: string,
    frameIndex: number,
    meta: MotionMeta,
  ): Promise<void> => {
    const blob = pngBlobFrom(still);
    const name = `motion-${sequenceId}-${String(frameIndex).padStart(4, "0")}`;
    await post(`${name}.png`, blob, undefined, MOTION_DIR);
    const sidecar: Record<string, unknown> = {
      kind: "synth",
      startedAt: Date.now(),
      durationMs: 0,
      imageWidth: WIDTH,
      imageHeight: HEIGHT,
      mimeType: "image/png",
      pose: meta.pose,
      board: meta.board,
      keyGeometry: meta.keyGeometry,
      keys: meta.keys,
      case: meta.caseState.present,
      caseColor: meta.caseState.colorFamily,
      sequence: sequenceId,
      frameIndex,
      piano: meta.piano,
    };
    if (meta.corners) {
      sidecar.corners = meta.corners;
      sidecar.cornerVisible = meta.cornerVisible;
    }
    await post(
      `${name}.json`,
      JSON.stringify(sidecar),
      "application/json",
      MOTION_DIR,
    );
    saved += 1;
  };

  // sequences run back to back until enough frames are saved; each one picks a keyboard, case,
  // lighting and lens the way the random sweep does, then moves the camera smoothly from a
  // start pose to a nearby end pose, blurring each frame across a few sub-renders
  const runMotionMode = async (target: number): Promise<void> => {
    let sequenceIndex = 0;
    while (saved < target) {
      sequenceIndex += 1;
      const sequenceId = `${stamp()}-${sequenceIndex}`;
      randomise();
      randomiseLens();
      const board = pickBoardSize(random);
      const variation = sampleKeyVariation(random);
      currentKeyboard = buildKeyboard(
        { lowestPitch: board.range.lowest, keys: board.keys },
        variation,
      );
      keysGroup.visible = true;
      applyKeyboard(currentKeyboard);
      layoutCase({ minZ: currentKeyboard.minZ, maxZ: currentKeyboard.maxZ });
      const centre = centreFor(currentKeyboard.minZ, currentKeyboard.maxZ);
      const boardSpan = currentKeyboard.maxZ - currentKeyboard.minZ;

      // negatives: the keyboard and its case are hidden entirely, so the model also sees what
      // no keyboard in view looks like
      const negative = random() < MOTION_NEGATIVE_PROBABILITY;
      if (negative) {
        keysGroup.visible = false;
        body.visible = false;
      }

      const start = sampleFramingPose(random);
      const panSign = random() < 0.5 ? -1 : 1;
      const tiltSign = random() < 0.5 ? -1 : 1;
      const rollSign = random() < 0.5 ? -1 : 1;
      const end: SweepPose = {
        ...start,
        elevationDeg: clamp(
          start.elevationDeg + tiltSign * between(random, MOTION_TILT_DEG),
          SWEEP_ELEVATION_DEG,
        ),
        azimuthDeg: clamp(
          start.azimuthDeg + panSign * between(random, MOTION_PAN_DEG),
          SWEEP_AZIMUTH_DEG,
        ),
        distanceFactor: clamp(
          start.distanceFactor * between(random, MOTION_DOLLY_FACTOR),
          SWEEP_DISTANCE_FACTOR,
        ),
        rollDeg: clamp(
          start.rollDeg + rollSign * between(random, MOTION_ROLL_DEG),
          SWEEP_ROLL_DEG,
        ),
      };
      const shake = {
        elevation: random() * MOTION_SHAKE_DEG,
        azimuth: random() * MOTION_SHAKE_DEG,
        roll: random() * MOTION_SHAKE_DEG * 0.5,
        cycles: between(random, MOTION_SHAKE_CYCLES),
        phaseE: random() * Math.PI * 2,
        phaseA: random() * Math.PI * 2,
        phaseR: random() * Math.PI * 2,
      };

      // t runs 0 to 1 over the whole sequence; the wobble term is a function of t too, so it
      // stays a continuous, smoothly blurrable hand-held shake rather than per-frame noise
      const poseAt = (t: number): SweepPose => {
        const clamped = Math.min(1, Math.max(0, t));
        const wobble = (amp: number, phase: number): number =>
          amp * Math.sin(shake.cycles * Math.PI * 2 * clamped + phase);
        return {
          elevationDeg:
            start.elevationDeg +
            (end.elevationDeg - start.elevationDeg) * clamped +
            wobble(shake.elevation, shake.phaseE),
          azimuthDeg:
            start.azimuthDeg +
            (end.azimuthDeg - start.azimuthDeg) * clamped +
            wobble(shake.azimuth, shake.phaseA),
          distanceFactor:
            start.distanceFactor +
            (end.distanceFactor - start.distanceFactor) * clamped,
          rollDeg:
            start.rollDeg +
            (end.rollDeg - start.rollDeg) * clamped +
            wobble(shake.roll, shake.phaseR),
          fovDeg: start.fovDeg,
          offsetXFactor: start.offsetXFactor,
          offsetYFactor: start.offsetYFactor,
        };
      };

      const applyPathPose = (pose: SweepPose): void => {
        currentRollDeg = pose.rollDeg;
        camera.fov = pose.fovDeg;
        camera.updateProjectionMatrix();
        aim(
          {
            elevation: pose.elevationDeg,
            azimuth: pose.azimuthDeg,
            distance: boardSpan * pose.distanceFactor,
          },
          centre,
        );
        camera.rotateZ((pose.rollDeg * Math.PI) / 180);
        camera.translateX(pose.offsetXFactor * boardSpan);
        camera.translateY(pose.offsetYFactor * boardSpan);
      };

      const frameCount = Math.floor(between(random, MOTION_FRAME_COUNT));
      const frameInterval = frameCount > 1 ? 1 / (frameCount - 1) : 1;

      for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
        const centerT = frameCount > 1 ? frameIndex / (frameCount - 1) : 0;
        // exposure is a random share of the gap between frames, so sub-renders sample a window
        // around the frame's own time rather than the whole path
        const exposure = frameInterval * between(random, MOTION_EXPOSURE_SHARE);
        stillCtx.clearRect(0, 0, WIDTH, HEIGHT);
        for (let sub = 0; sub < MOTION_SUBFRAMES; sub += 1) {
          const subT =
            MOTION_SUBFRAMES > 1
              ? centerT -
                exposure / 2 +
                exposure * (sub / (MOTION_SUBFRAMES - 1))
              : centerT;
          applyPathPose(poseAt(subT));
          renderer.render(scene, camera);
          // cumulative moving average: alpha 1/(sub+1) makes each draw carry equal weight in
          // the final blend, source-over compositing does the rest
          stillCtx.globalAlpha = 1 / (sub + 1);
          stillCtx.drawImage(canvas, 0, 0);
        }
        stillCtx.globalAlpha = 1;

        // the saved corners and keys are the centre-time pose, not any blurred sub-render; only
        // a render refreshes the camera's world matrix, so we refresh it before projecting
        applyPathPose(poseAt(centerT));
        camera.updateMatrixWorld();
        const activeCorners = cornersFor(
          currentKeyboard.minZ,
          currentKeyboard.maxZ,
        );
        const corners = projectCorners(camera, activeCorners);
        const fraction = visibleFraction(camera, activeCorners);
        if (!negative && fraction < MIN_VISIBLE_FRACTION) {
          break;
        }
        const pose = cameraPoseDeg(centre);
        await captureMotionFrame(sequenceId, frameIndex, {
          corners: negative ? null : corners,
          cornerVisible: cornersVisible(camera, activeCorners, caseOccluders()),
          pose: { ...pose, rollDeg: currentRollDeg, fovDeg: camera.fov },
          board: {
            keys: currentKeyboard.board.keys,
            whiteKeys: currentKeyboard.whiteKeys,
            lowestPitch: currentKeyboard.board.lowestPitch,
            lowestNote: noteName(currentKeyboard.board.lowestPitch),
          },
          keyGeometry: currentKeyboard.variation,
          keys: negative ? [] : projectKeyLabels(currentKeyboard),
          caseState: negative
            ? { present: false, colorFamily: null }
            : currentCase,
          piano: !negative,
        });
        status(
          `motion seq ${sequenceIndex} frame ${frameIndex + 1}/${frameCount}  saved ${saved}/${target}`,
        );
        if (saved >= target) {
          return;
        }
      }
    }
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

  if (MOTION_MODE) {
    const target =
      Number.isFinite(AUTOSTART_FRAMES) && AUTOSTART_FRAMES > 0
        ? AUTOSTART_FRAMES
        : DEFAULT_SWEEP_TARGET;
    await runMotionMode(target);
    return;
  }

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
            cornerVisible: cornersVisible(
              camera,
              activeCorners,
              caseOccluders(),
            ),
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
          cornerVisible: cornersVisible(camera, activeCorners, caseOccluders()),
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
