import { calibrateBoard } from "./boardgeometry";
import { createDetector } from "./detector";
import type { Point } from "./homography";
import { viteAssets } from "./viteassets";

const DEFAULT_SAMPLES = 105;

interface Sidecar {
  corners: Point[];
  imageWidth: number;
  imageHeight: number;
  pose: { elevation: number; azimuth: number; distance: number; fov: number };
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
  raw: Stage;
  rectangle: Stage;
  keyLayout: Stage;
}

interface Summary {
  found: number;
  medianErrorPx: number | null;
  p95ErrorPx: number | null;
  worstErrorPx: number | null;
}

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

function pixels(image: ImageBitmap): ImageData {
  const canvas = document.createElement("canvas");
  canvas.width = image.width;
  canvas.height = image.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("2d canvas context unavailable");
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, image.width, image.height);
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

function summary(
  rows: readonly Row[],
  name: keyof Pick<Row, "raw" | "rectangle" | "keyLayout">,
): Summary {
  const errors = rows
    .map((row) => row[name].error?.mean)
    .filter((value): value is number => value !== null)
    .sort((a, b) => a - b);
  return {
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

async function evaluateGrid(): Promise<{
  kind: string;
  summary: Record<"raw" | "rectangle" | "keyLayout", Summary>;
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
  for (const name of names) {
    const sidecarResponse = await fetch(
      `/lab/data/grid/${encodeURIComponent(name)}`,
    );
    if (!sidecarResponse.ok) continue;
    const sidecar = (await sidecarResponse.json()) as Sidecar;
    if (sidecar.corners.length !== 4 || !sidecar.pose) continue;
    const image = await bitmap(`${name.slice(0, -5)}.png`);
    const detection = await detector.detect(image);
    const frame = pixels(image);
    const layout = detection.quad
      ? calibrateBoard(
          frame,
          detection.quad.map((point) => ({
            x: point.x * image.width,
            y: point.y * image.height,
          })),
        )
      : null;
    image.close();
    rows.push({
      name,
      pose: sidecar.pose,
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
      keyLayout: stage(
        layout?.quad.map((point) => ({
          x: point.x / sidecar.imageWidth,
          y: point.y / sidecar.imageHeight,
        })) ?? null,
        sidecar.corners,
        sidecar.imageWidth,
        sidecar.imageHeight,
      ),
    });
  }
  return {
    kind: "kvt-browser-grid-evaluation",
    summary: {
      raw: summary(rows, "raw"),
      rectangle: summary(rows, "rectangle"),
      keyLayout: summary(rows, "keyLayout"),
    },
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
