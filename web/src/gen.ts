import {
  ACESFilmicToneMapping,
  AmbientLight,
  BoxGeometry,
  Color,
  DirectionalLight,
  Group,
  LinearSRGBColorSpace,
  Mesh,
  MeshStandardMaterial,
  NoToneMapping,
  PerspectiveCamera,
  PMREMGenerator,
  Scene,
  SRGBColorSpace,
  WebGLRenderer,
} from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { type BackdropKind, makeBackdrop } from "./backdrop";
import { drawQuad } from "./draw";
import { makeEnvironments } from "./environments";
import type { Point } from "./homography";
import { styleButton } from "./hud";
import {
  BACK_X,
  DEPTH,
  FRONT_X,
  KEY_TOP_Y,
  KEYBED_CENTRE,
  projectCorners,
  SPAN,
  SPAN_MAX_Z,
  SPAN_MIN_Z,
  visibleFraction,
} from "./keybed3d";
import {
  createKeyIdOverlay,
  instanceMaskColor,
  type KeyIdSpec,
  keyIdSpecs,
} from "./keylabels";

const MODEL_URL = "/models/piano_keys.glb";
const WIDTH = queryDimension("width", 640);
const HEIGHT = queryDimension("height", 480);
const CAPTURE_INTERVAL_MS = 300;
// enough of the keybed to be worth a label; below this there is nothing to learn from
const MIN_VISIBLE_FRACTION = 0.2;
// the pose ranges a person actually films a keyboard from, swept automatically
const SWEEP_ELEVATION = [8, 78];
// azimuth 0 faces the keys head on and +-90 looks straight down the keybed from one end;
// people film from both, so the sweep has to run past the ends rather than stop short of them
const SWEEP_AZIMUTH = [-105, 105];
const SWEEP_DISTANCE = [SPAN * 0.45, SPAN * 1.4];
const SWEEP_ROLL = [-16, 16];
const BACKDROPS: BackdropKind[] = ["clutter", "noise", "gradient"];
// a fixed lattice of poses, rendered under a seeded generator, so a change in the detector is
// attributable to a pose rather than lost in a random sweep
const GRID_ELEVATION = [10, 25, 40, 55, 70];
const GRID_AZIMUTH = [-90, -60, -30, 0, 30, 60, 90];
const GRID_DISTANCE = [SPAN * 0.5, SPAN * 0.8, SPAN * 1.2];
const GRID_FOV = [25, 45, 70];
const GRID_SEED = 7;
const MAX_BATCH_SAMPLES = 10_000;

interface Pose {
  elevation: number;
  azimuth: number;
  distance: number;
  fov: number;
}

function gridPoses(): Pose[] {
  const poses: Pose[] = [];
  for (const elevation of GRID_ELEVATION) {
    for (const azimuth of GRID_AZIMUTH) {
      for (const distance of GRID_DISTANCE) {
        for (const fov of GRID_FOV) {
          poses.push({ elevation, azimuth, distance, fov });
        }
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

function queryPositiveInteger(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name));
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function queryDimension(name: string, fallback: number): number {
  return Math.min(queryPositiveInteger(name, fallback), fallback);
}

async function post(
  name: string,
  body: Blob | string,
  type?: string,
  directory = "key-instances",
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
  controls.target.copy(KEYBED_CENTRE);
  controls.update();

  const key = new DirectionalLight(0xffffff, 2);
  const fill = new DirectionalLight(0xffffff, 1);
  const ambient = new AmbientLight(0xffffff, 0.4);
  scene.add(key, fill, ambient);

  const gltf = await new GLTFLoader().loadAsync(MODEL_URL);
  scene.add(gltf.scene);
  const keyIds = createKeyIdOverlay();
  scene.add(keyIds.group);
  const keyInstances = keyIdSpecs().map((key, index) => ({
    id: index + 1,
    color: instanceMaskColor(index),
    pitch: key.pitch,
    label: key.label,
    black: key.black,
  }));

  // a real instrument is a case with a panel butting onto the back of the keys; a floating slab
  // gives the net no way to learn where the keybed actually ends
  const body = new Group();
  const caseMaterial = new MeshStandardMaterial({
    color: 0x111114,
    roughness: 0.55,
  });
  const centreZ = (SPAN_MIN_Z + SPAN_MAX_Z) / 2;
  const panelDepth = DEPTH * 1.1;
  const backPanel = new Mesh(
    new BoxGeometry(panelDepth, DEPTH * 0.5, SPAN * 1.02),
    caseMaterial,
  );
  backPanel.position.set(BACK_X - panelDepth / 2, KEY_TOP_Y, centreZ);
  const base = new Mesh(
    new BoxGeometry(panelDepth + DEPTH, DEPTH * 0.3, SPAN * 1.04),
    caseMaterial,
  );
  base.position.set(
    (BACK_X + FRONT_X) / 2 - panelDepth / 2,
    KEY_TOP_Y - DEPTH * 0.28,
    centreZ,
  );
  const cheekWidth = SPAN * 0.02;
  for (const z of [SPAN_MIN_Z - cheekWidth / 2, SPAN_MAX_Z + cheekWidth / 2]) {
    const cheek = new Mesh(
      new BoxGeometry(DEPTH, DEPTH * 0.25, cheekWidth),
      caseMaterial,
    );
    cheek.position.set((BACK_X + FRONT_X) / 2, KEY_TOP_Y - DEPTH * 0.06, z);
    body.add(cheek);
  }
  body.add(backPanel, base);
  scene.add(body);
  const materials: MeshStandardMaterial[] = [];
  gltf.scene.traverse((node) => {
    if (node instanceof Mesh && node.material instanceof MeshStandardMaterial) {
      materials.push(node.material);
    }
  });

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
  const grid = document.createElement("button");
  const shuffle = document.createElement("button");
  const ids = document.createElement("button");
  const mask = document.createElement("button");
  const readout = document.createElement("span");
  record.textContent = "record";
  sweep.textContent = "auto sweep";
  grid.textContent = "render test grid";
  shuffle.textContent = "randomise";
  ids.textContent = "key IDs";
  mask.textContent = "key mask";
  for (const button of [record, sweep, grid, shuffle, ids, mask]) {
    styleButton(button);
  }
  readout.style.color = "#8a8a8a";
  panel.append(record, sweep, grid, shuffle, ids, mask, readout);
  document.body.appendChild(panel);
  let random: () => number = Math.random;
  const status = (text: string): void => {
    readout.textContent = text;
  };

  const randomiseLens = (): void => {
    camera.fov = 25 + random() * 45;
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
    for (const material of materials) {
      // key plastic runs from matte to near mirror, and the reflection is what a flat
      // brightness jitter can never fake
      material.roughness = 0.04 + random() * random() * 0.85;
      material.metalness = random() * 0.35;
      material.envMapIntensity = 0.3 + random() * 2.2;
      material.needsUpdate = true;
    }
    // the back edge is the one boundary the net has to find, so its contrast is randomised
    // both ways; a case that is always dark teaches brightness instead of geometry
    const shade = random();
    caseMaterial.color.setHSL(
      random(),
      0.12 * random(),
      0.02 + shade * shade * 0.72,
    );
    caseMaterial.roughness = 0.08 + random() * 0.8;
    caseMaterial.envMapIntensity = 0.3 + random() * 2.0;
    body.visible = random() > 0.15;
  };
  randomise();
  randomiseLens();

  let recording = false;
  let sweeping = false;
  let saved = 0;
  let lastCapture = 0;
  const poses = gridPoses();
  let gridIndex = -1;
  let gridSkipped = 0;
  let selected: KeyIdSpec | null = null;
  let pointerDown: { x: number; y: number } | null = null;
  let maskPreview = false;
  const batchSamples = Math.min(
    queryPositiveInteger("samples", 0),
    MAX_BATCH_SAMPLES,
  );
  const batchSeed = queryPositiveInteger("seed", 7);
  let queued = 0;

  const between = ([low, high]: number[]): number =>
    low + random() * (high - low);

  const aim = (pose: Pose): void => {
    const elevation = (pose.elevation * Math.PI) / 180;
    const azimuth = (pose.azimuth * Math.PI) / 180;
    // the keys run along z, so azimuth has to swing around the front (+x) face; putting
    // azimuth 0 on +z aims the camera down the length of the keybed and foreshortens it away
    camera.position.set(
      KEYBED_CENTRE.x + pose.distance * Math.cos(elevation) * Math.cos(azimuth),
      KEYBED_CENTRE.y + pose.distance * Math.sin(elevation),
      KEYBED_CENTRE.z + pose.distance * Math.cos(elevation) * Math.sin(azimuth),
    );
    camera.up.set(0, 1, 0);
    camera.lookAt(KEYBED_CENTRE);
  };

  // a scripted camera covers the pose space evenly and without anybody having to orbit by hand
  const placeCamera = (): void => {
    aim({
      elevation: between(SWEEP_ELEVATION),
      azimuth: between(SWEEP_AZIMUTH),
      distance: between(SWEEP_DISTANCE),
      fov: camera.fov,
    });
    camera.rotateZ((between(SWEEP_ROLL) * Math.PI) / 180);
    // aim a little off centre so the keybed is not always dead centre of frame
    camera.translateX((random() - 0.5) * SPAN * 0.25);
    camera.translateY((random() - 0.5) * SPAN * 0.12);
  };

  const placeGrid = (): void => {
    aim(poses[gridIndex]);
    camera.fov = poses[gridIndex].fov;
    camera.updateProjectionMatrix();
    randomise();
  };

  const stopGrid = (): void => {
    gridIndex = -1;
    random = Math.random;
    controls.enabled = true;
    styleButton(grid, false);
  };

  const still = document.createElement("canvas");
  still.width = WIDTH;
  still.height = HEIGHT;
  const stillCtx = still.getContext("2d");
  if (!stillCtx) {
    throw new Error("2d canvas context unavailable");
  }
  const maskStill = document.createElement("canvas");
  maskStill.width = WIDTH;
  maskStill.height = HEIGHT;
  const maskCtx = maskStill.getContext("2d");
  if (!maskCtx) {
    throw new Error("2d mask context unavailable");
  }

  const pngBlob = (source: HTMLCanvasElement): Blob => {
    const url = source.toDataURL("image/png");
    const binary = atob(url.slice(url.indexOf(",") + 1));
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return new Blob([bytes], { type: "image/png" });
  };

  const renderRgb = (): void => {
    const idsVisible = keyIds.group.visible;
    keyIds.group.visible = false;
    renderer.render(scene, camera);
    keyIds.group.visible = idsVisible;
  };

  const renderMask = (): void => {
    const background = scene.background;
    const sourceVisible = gltf.scene.visible;
    const bodyVisible = body.visible;
    const idsVisible = keyIds.group.visible;
    const toneMapping = renderer.toneMapping;
    const outputColorSpace = renderer.outputColorSpace;
    scene.background = new Color(0);
    gltf.scene.visible = false;
    body.visible = false;
    keyIds.group.visible = true;
    keyIds.setMaskMode(true);
    renderer.toneMapping = NoToneMapping;
    renderer.outputColorSpace = LinearSRGBColorSpace;
    renderer.render(scene, camera);
    renderer.outputColorSpace = outputColorSpace;
    renderer.toneMapping = toneMapping;
    keyIds.setMaskMode(false);
    keyIds.group.visible = idsVisible;
    body.visible = bodyVisible;
    gltf.scene.visible = sourceVisible;
    scene.background = background;
  };

  // the frame is copied and encoded synchronously: an async toBlob can race the next render
  // and write a png that does not match the corners saved beside it
  const capture = (corners: Point[], pose?: Pose): void => {
    renderRgb();
    stillCtx.drawImage(canvas, 0, 0);
    const image = pngBlob(still);
    renderMask();
    maskCtx.drawImage(canvas, 0, 0);
    const instanceMask = pngBlob(maskStill);
    {
      const directory = "key-instances";
      const name = pose
        ? `keyinst-grid-e${pose.elevation}-a${pose.azimuth}-d${Math.round(pose.distance)}-f${pose.fov}`
        : batchSamples > 0
          ? `keyinst-batch-${batchSeed}-${stamp()}-${queued}`
          : `keyinst-${stamp()}`;
      queued += 1;
      Promise.all([
        post(`${name}.png`, image, undefined, directory),
        post(`${name}-keys.png`, instanceMask, undefined, directory),
      ])
        .then(() =>
          post(
            `${name}.json`,
            JSON.stringify({
              kind: "key-instance-synth",
              startedAt: Date.now(),
              durationMs: 0,
              corners,
              camera: {
                fov: camera.fov,
                distance: camera.position.distanceTo(KEYBED_CENTRE),
              },
              imageWidth: WIDTH,
              imageHeight: HEIGHT,
              mimeType: "image/png",
              instanceMask: `${name}-keys.png`,
              instances: keyInstances,
              ...(pose ? { pose } : {}),
            }),
            "application/json",
            directory,
          ),
        )
        .then(() => {
          saved += 1;
        })
        .catch((err: unknown) => {
          status(err instanceof Error ? err.message : String(err));
        });
    }
  };

  shuffle.addEventListener("click", () => {
    randomise();
    randomiseLens();
  });
  ids.addEventListener("click", () => {
    maskPreview = false;
    styleButton(mask, false);
    keyIds.group.visible = !keyIds.group.visible;
    styleButton(ids, keyIds.group.visible);
    selected = null;
  });
  mask.addEventListener("click", () => {
    maskPreview = !maskPreview;
    styleButton(mask, maskPreview);
    if (maskPreview) {
      keyIds.group.visible = false;
      styleButton(ids, false);
      selected = null;
    }
  });
  canvas.addEventListener("pointerdown", (event) => {
    pointerDown = { x: event.clientX, y: event.clientY };
  });
  canvas.addEventListener("pointerup", (event) => {
    if (!pointerDown || !keyIds.group.visible) {
      pointerDown = null;
      return;
    }
    const distance = Math.hypot(
      event.clientX - pointerDown.x,
      event.clientY - pointerDown.y,
    );
    pointerDown = null;
    if (distance > 4) {
      return;
    }
    const box = canvas.getBoundingClientRect();
    selected = keyIds.pick(
      camera,
      ((event.clientX - box.left) / box.width) * 2 - 1,
      -((event.clientY - box.top) / box.height) * 2 + 1,
    );
  });
  sweep.addEventListener("click", () => {
    sweeping = !sweeping;
    controls.enabled = !sweeping;
    styleButton(sweep, sweeping);
    if (sweeping) {
      recording = true;
      styleButton(record, true);
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
    sweeping = false;
    styleButton(sweep, false);
    recording = false;
    styleButton(record, false);
    controls.enabled = false;
    random = seeded(GRID_SEED);
    gridIndex = 0;
    gridSkipped = 0;
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

  if (batchSamples > 0) {
    random = seeded(batchSeed);
    randomise();
    randomiseLens();
    sweeping = true;
    recording = true;
    controls.enabled = false;
    styleButton(sweep, true);
    styleButton(record, true);
  }

  const frame = (now: number): void => {
    if (!sweeping) {
      controls.update();
    }
    if (maskPreview) {
      renderMask();
    } else {
      renderer.render(scene, camera);
    }
    const corners = projectCorners(camera);
    const fraction = visibleFraction(camera);
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
      `${Math.round(fraction * 100)}% visible${usable ? "" : ", too little to save"}  saved ${saved}${
        selected ? `  ${selected.label} (${selected.pitch})` : ""
      }`,
    );

    if (gridIndex >= 0) {
      // this frame shows the pose placed last tick, so it is captured before moving on
      if (usable) {
        capture(corners, poses[gridIndex]);
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
    if (
      recording &&
      !maskPreview &&
      usable &&
      (batchSamples === 0 || queued < batchSamples) &&
      now - lastCapture > CAPTURE_INTERVAL_MS
    ) {
      lastCapture = now;
      capture(corners);
      randomise();
      if (sweeping) {
        randomiseLens();
      }
    }
    if (batchSamples > 0 && queued >= batchSamples) {
      recording = false;
      sweeping = false;
      controls.enabled = true;
      styleButton(sweep, false);
      styleButton(record, false);
    }
    if (sweeping && (!usable || now - lastCapture < 1)) {
      placeCamera();
    }
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

void boot();
