import { createDetector } from "./detector";
import type { Point } from "./homography";
import { keybedDepth, setKeybedDepth, WHITE_KEY_COUNT } from "./pose";
import { viteAssets } from "./viteassets";

const DEFAULT_SAMPLES = 105;

// the grid renders an 88-key keybed, 52 white keys at 23.5 mm deep by 150 mm, while
// WHITE_KEY_COUNT stays fixed to the real 61-key instrument; only the aspect ratio the
// rectangle fit derives from these matters, so we reproduce it by scaling the depth alone
const RENDER_WHITE_KEY_COUNT = 52;
const RENDER_WHITE_KEY_MM = 23.5;
const RENDER_DEPTH_MM = 150;
const RENDER_DEPTH_UNITS = RENDER_DEPTH_MM / RENDER_WHITE_KEY_MM;
const RENDER_KEYBED_DEPTH_UNITS =
  (WHITE_KEY_COUNT * RENDER_DEPTH_UNITS) / RENDER_WHITE_KEY_COUNT;

interface Sidecar {
  corners: Point[];
  imageWidth: number;
  imageHeight: number;
  pose: { elevation: number; azimuth: number; distance: number };
}

interface ErrorPx {
  mean: number;
  max: number;
}

interface Stage {
  found: number;
  error: ErrorPx | null;
}

interface Row {
  name: string;
  pose: Sidecar["pose"];
  fullyVisible: boolean;
  raw: Stage;
  rectangle: Stage;
}

interface Summary {
  eligible: number;
  found: number;
  medianErrorPx: number | null;
  p95ErrorPx: number | null;
  worstErrorPx: number | null;
}

type StageName = "raw" | "rectangle";
type StageSummary = Record<StageName, Summary>;

function count(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name));
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function pointError(
  actual: readonly Point[],
  expected: readonly Point[],
  width: number,
  height: number,
): ErrorPx {
  let best: number[] | null = null;
  for (const reverse of [false, true]) {
    for (let shift = 0; shift < 4; shift += 1) {
      const errors = actual.map((point, index) => {
        const raw = reverse ? 3 - index : index;
        const target = expected[(raw + shift) % 4] ?? expected[0];
        return Math.hypot(
          (point.x - target.x) * width,
          (point.y - target.y) * height,
        );
      });
      if (
        best === null ||
        errors.reduce((sum, value) => sum + value, 0) <
          best.reduce((sum, value) => sum + value, 0)
      )
        best = errors;
    }
  }
  const values = best ?? [];
  return {
    mean: values.reduce((sum, value) => sum + value, 0) / values.length,
    max: Math.max(...values),
  };
}

async function bitmap(name: string): Promise<ImageBitmap> {
  const response = await fetch(`/lab/data/grid/${encodeURIComponent(name)}`);
  if (!response.ok) throw new Error(`cannot read ${name}: ${response.status}`);
  return createImageBitmap(await response.blob());
}

function stage(
  quad: readonly Point[] | null,
  truth: readonly Point[],
  width: number,
  height: number,
): Stage {
  return {
    found: quad ? 1 : 0,
    error: quad ? pointError(quad, truth, width, height) : null,
  };
}

function fullyVisible(corners: readonly Point[]): boolean {
  return corners.every(
    (corner) =>
      corner.x >= 0 && corner.x <= 1 && corner.y >= 0 && corner.y <= 1,
  );
}

// only a fully visible keybed has a well-defined ground truth for every corner, so a
// pose the camera clips is excluded from the scored summaries rather than scored on
// whatever corners happen to land in frame
function summary(rows: readonly Row[], name: StageName): Summary {
  const observable = rows.filter((row) => row.fullyVisible);
  const errors = observable
    .map((row) => row[name].error?.mean)
    .filter((value): value is number => Number.isFinite(value))
    .sort((a, b) => a - b);
  return {
    eligible: observable.length,
    found: errors.length,
    medianErrorPx: errors.length
      ? (errors[Math.floor(errors.length / 2)] ?? null)
      : null,
    p95ErrorPx: errors.length
      ? (errors[
          Math.min(errors.length - 1, Math.floor(errors.length * 0.95))
        ] ?? null)
      : null,
    worstErrorPx: errors.at(-1) ?? null,
  };
}

function stageSummary(rows: readonly Row[]): StageSummary {
  return { raw: summary(rows, "raw"), rectangle: summary(rows, "rectangle") };
}

function groupedBy(
  rows: readonly Row[],
  key: (row: Row) => number,
): Record<string, StageSummary> {
  const groups: Record<string, StageSummary> = {};
  for (const value of [...new Set(rows.map(key))].sort((a, b) => a - b)) {
    groups[String(value)] = stageSummary(
      rows.filter((row) => key(row) === value),
    );
  }
  return groups;
}

async function evaluateGrid(): Promise<{
  kind: string;
  summary: StageSummary;
  byElevation: Record<string, StageSummary>;
  byAbsAzimuth: Record<string, StageSummary>;
  rows: Row[];
}> {
  const response = await fetch("/lab/list/grid");
  if (!response.ok) throw new Error(`cannot list grid: ${response.status}`);
  const names = ((await response.json()) as unknown[])
    .filter(
      (name): name is string =>
        typeof name === "string" && name.endsWith(".json"),
    )
    .sort()
    .slice(0, count("samples", DEFAULT_SAMPLES));
  const detector = await createDetector(viteAssets);
  const rows: Row[] = [];
  const previousDepth = keybedDepth();
  setKeybedDepth(RENDER_KEYBED_DEPTH_UNITS);
  try {
    for (const name of names) {
      const sidecarResponse = await fetch(
        `/lab/data/grid/${encodeURIComponent(name)}`,
      );
      if (!sidecarResponse.ok) continue;
      const sidecar = (await sidecarResponse.json()) as Sidecar;
      if (sidecar.corners.length !== 4 || !sidecar.pose) continue;
      const image = await bitmap(`${name.slice(0, -5)}.png`);
      // each render is an independent pose, so the detector's temporal smoothing (motion,
      // still-frame averaging, the held rectangle) must not carry over from the last one
      detector.reset();
      const detection = await detector.detect(image);
      image.close();
      rows.push({
        name,
        pose: sidecar.pose,
        fullyVisible: fullyVisible(sidecar.corners),
        raw: stage(
          detection.proposalQuad,
          sidecar.corners,
          sidecar.imageWidth,
          sidecar.imageHeight,
        ),
        rectangle: stage(
          detection.quad,
          sidecar.corners,
          sidecar.imageWidth,
          sidecar.imageHeight,
        ),
      });
    }
  } finally {
    setKeybedDepth(previousDepth);
  }
  return {
    kind: "pianocv-browser-grid-evaluation",
    summary: stageSummary(rows),
    byElevation: groupedBy(rows, (row) => row.pose.elevation),
    byAbsAzimuth: groupedBy(rows, (row) => Math.abs(row.pose.azimuth)),
    rows,
  };
}

async function main(): Promise<void> {
  const element = document.querySelector<HTMLPreElement>("#report");
  try {
    const report = await evaluateGrid();
    const body = JSON.stringify(report, null, 2);
    if (element) element.textContent = body;
    const run =
      new URLSearchParams(window.location.search).get("run") ?? "browser-grid";
    const response = await fetch(
      `/lab/save/evaluations/${encodeURIComponent(run)}.json`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      },
    );
    if (!response.ok) throw new Error(`cannot save report: ${response.status}`);
    document.title = "evaluation complete";
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : String(error);
    if (element) element.textContent = message;
    document.title = "evaluation failed";
    throw error;
  }
}

void main();
