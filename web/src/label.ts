import { fitStill } from "./fitcheck";
import { createKeyNet } from "./keynetrunner";
import {
  emptyLabels,
  type FixedLabels,
  type LabelPoint,
  lacksRear,
  POINT_KINDS,
  type PointKind,
  parseLabels,
  prefill,
  prefillRear,
  type RearLabels,
  reproject,
  serialise,
} from "./labelfix";
import { viteAssets } from "./viteassets";

const COLORS: Record<PointKind, string> = {
  corners: "#ffffff",
  gaps: "#ffd800",
  blackLow: "#00e5ff",
  blackHigh: "#ff40ff",
  blackTopLow: "#6bff6b",
  blackTopHigh: "#ff9a30",
  backGaps: "#b08cff",
  blackBackLow: "#ff6b6b",
  blackBackHigh: "#8cb4ff",
};
const HINTS =
  "n next frame | p previous frame | s save | x hide or show point | a move points low | d move points high | r refit points | wheel zoom | drag move point";
const FILL_REAR = new URLSearchParams(location.search).has("fillRear");
const GRAB_PX = 9;
const MOST_ZOOM = 24;
const REAL_URL = "/lab/data/real-keys";
const FIXED_URL = "/lab/data/real-keys-fixed";

type Ref = { readonly kind: PointKind; readonly index: number };

function required<T>(value: T | null | undefined): T {
  if (value === null || value === undefined) {
    throw new Error("label page is missing its elements");
  }
  return value;
}

const canvas = required(document.querySelector<HTMLCanvasElement>("#view"));
const statusLine = required(document.querySelector<HTMLDivElement>("#status"));
const legend = required(document.querySelector<HTMLDivElement>("#legend"));
const context = required(canvas.getContext("2d"));

legend.innerHTML = POINT_KINDS.map(
  (kind) => `<span style="color:${COLORS[kind]}">&#9679; ${kind}</span>`,
).join("<br>");

const keyNetReady = createKeyNet(viteAssets);

let stems: string[] = [];
let current = 0;
let image: ImageBitmap | null = null;
let labels: FixedLabels = emptyLabels({ width: 1, height: 1 });
let ghosts: FixedLabels = labels;
let origin: "saved" | "model fit" | "loading" = "loading";
let dirty = false;
let message = "";
let zoom = 1;
let panX = 0;
let panY = 0;
let cursor = { x: 0, y: 0 };
let grabbed: Ref | null = null;
let panning = false;

function setLabels(next: FixedLabels): void {
  labels = next;
  ghosts = reproject(next, 0, true);
}

function fitView(): void {
  if (!image) {
    return;
  }
  zoom = Math.min(innerWidth / image.width, innerHeight / image.height);
  panX = (innerWidth - image.width * zoom) / 2;
  panY = (innerHeight - image.height * zoom) / 2;
}

function toScreen(p: [number, number]): [number, number] {
  return [p[0] * zoom + panX, p[1] * zoom + panY];
}

function describe(): string {
  const state = dirty ? "unsaved edits" : origin;
  return `${stems[current] ?? "no frames"} ${current + 1}/${stems.length} | ${state}${message ? ` | ${message}` : ""} | ${HINTS}`;
}

function draw(): void {
  canvas.width = innerWidth;
  canvas.height = innerHeight;
  context.fillStyle = "#050505";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (image) {
    context.imageSmoothingEnabled = zoom < 4;
    context.drawImage(
      image,
      panX,
      panY,
      image.width * zoom,
      image.height * zoom,
    );
  }
  for (const kind of POINT_KINDS) {
    context.strokeStyle = COLORS[kind];
    context.fillStyle = COLORS[kind];
    labels[kind].forEach((point, index) => {
      const ghost = ghosts[kind][index];
      const shown = point ?? ghost;
      if (!shown) {
        return;
      }
      const [x, y] = toScreen(shown);
      context.globalAlpha = point ? 1 : 0.35;
      context.beginPath();
      context.arc(x, y, point ? 3 : 4, 0, Math.PI * 2);
      if (point) {
        context.fill();
      } else {
        context.stroke();
      }
    });
  }
  context.globalAlpha = 1;
  statusLine.textContent = describe();
}

function pointAt(x: number, y: number, hidden: boolean): Ref | null {
  let best: Ref | null = null;
  let bestDistance = GRAB_PX;
  for (const kind of POINT_KINDS) {
    labels[kind].forEach((point, index) => {
      const shown: LabelPoint = hidden
        ? point
          ? null
          : ghosts[kind][index]
        : point;
      if (!shown) {
        return;
      }
      const [sx, sy] = toScreen(shown);
      const distance = Math.hypot(sx - x, sy - y);
      if (distance <= bestDistance) {
        best = { kind, index };
        bestDistance = distance;
      }
    });
  }
  return best;
}

function visiblePointAt(x: number, y: number): Ref | null {
  return pointAt(x, y, false) ?? pointAt(x, y, true);
}

async function modelRear(frame: ImageBitmap): Promise<RearLabels | null> {
  const { fit, peaks } = await fitStill(await keyNetReady, frame);
  return fit === null || peaks === null
    ? null
    : prefillRear(fit, peaks, { width: frame.width, height: frame.height });
}

async function modelLabels(frame: ImageBitmap): Promise<FixedLabels> {
  const size = { width: frame.width, height: frame.height };
  const keyNet = await keyNetReady;
  const { fit, peaks } = await fitStill(keyNet, frame);
  if (fit === null || peaks === null) {
    return emptyLabels(size);
  }
  return prefill(fit, peaks, size);
}

async function load(index: number): Promise<void> {
  current = index;
  origin = "loading";
  dirty = false;
  message = "";
  image = null;
  draw();
  const stem = stems[index];
  const frame = await createImageBitmap(
    await (await fetch(`${REAL_URL}/${stem}.png`)).blob(),
  );
  const saved = await fetch(`${FIXED_URL}/${stem}.json`);
  if (current !== index) {
    return;
  }
  image = frame;
  fitView();
  if (saved.ok) {
    const text = await saved.text();
    setLabels(parseLabels(text));
    origin = "saved";
    draw();
    if (FILL_REAR && lacksRear(text)) {
      const rear = await modelRear(frame);
      if (current === index && rear !== null) {
        setLabels({ ...labels, ...rear });
        dirty = true;
        message = "rear points filled from the model";
        draw();
      }
    }
    return;
  }
  setLabels(emptyLabels(frame));
  draw();
  const fitted = await modelLabels(frame);
  if (current !== index) {
    return;
  }
  setLabels(fitted);
  origin = "model fit";
  message = fitted.whiteKeys === 0 ? "no keyboard found" : "";
  draw();
}

async function save(): Promise<void> {
  if (labels.whiteKeys === 0) {
    message = "nothing to save";
    draw();
    return;
  }
  const response = await fetch(
    `/lab/save/real-keys-fixed/${stems[current]}.json`,
    { method: "POST", body: serialise(labels) },
  );
  dirty = !response.ok;
  origin = response.ok ? "saved" : origin;
  message = response.ok ? "" : "save failed";
  draw();
}

function go(step: number): void {
  const next = current + step;
  if (next < 0 || next >= stems.length) {
    return;
  }
  if (dirty) {
    message = "s save before leaving this frame";
    draw();
    return;
  }
  void load(next);
}

function toggleHidden(): void {
  const ref = visiblePointAt(cursor.x, cursor.y);
  if (!ref) {
    return;
  }
  const point = labels[ref.kind][ref.index];
  if (point) {
    ghosts[ref.kind][ref.index] = point;
    labels[ref.kind][ref.index] = null;
  } else {
    labels[ref.kind][ref.index] = ghosts[ref.kind][ref.index];
  }
  dirty = true;
  draw();
}

function shiftPoints(keys: number): void {
  setLabels(reproject(labels, keys));
  dirty = true;
  draw();
}

addEventListener("keydown", (event) => {
  const actions: Record<string, () => void> = {
    n: () => go(1),
    p: () => go(-1),
    s: () => void save(),
    x: toggleHidden,
    a: () => shiftPoints(-1),
    d: () => shiftPoints(1),
    r: () => shiftPoints(0),
  };
  if (!event.metaKey && !event.ctrlKey) {
    actions[event.key]?.();
  }
});

canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (!image) {
    return;
  }
  const next = Math.min(
    MOST_ZOOM,
    Math.max(0.1, zoom * Math.exp(-event.deltaY * 0.0015)),
  );
  panX = event.clientX - (event.clientX - panX) * (next / zoom);
  panY = event.clientY - (event.clientY - panY) * (next / zoom);
  zoom = next;
  draw();
});

canvas.addEventListener("pointerdown", (event) => {
  const ref = pointAt(event.clientX, event.clientY, false);
  grabbed = ref;
  panning = ref === null;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  const dx = event.clientX - cursor.x;
  const dy = event.clientY - cursor.y;
  cursor = { x: event.clientX, y: event.clientY };
  if (grabbed) {
    labels[grabbed.kind][grabbed.index] = [
      (event.clientX - panX) / zoom,
      (event.clientY - panY) / zoom,
    ];
    dirty = true;
    draw();
  } else if (panning) {
    panX += dx;
    panY += dy;
    draw();
  }
});

canvas.addEventListener("pointerup", () => {
  if (grabbed) {
    ghosts = reproject(labels, 0, true);
  }
  grabbed = null;
  panning = false;
});

addEventListener("resize", draw);

async function start(): Promise<void> {
  const listing = await fetch("/lab/list/real-keys");
  stems = ((await listing.json()) as string[])
    .map((name) => name.replace(/\.json$/, ""))
    .sort();
  if (stems.length > 0) {
    await load(0);
  }
}

void start();
