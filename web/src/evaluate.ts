import { calibrateBoard } from "./boardgeometry";
import { applyHomography, findHomography, type Point } from "./homography";
import { type Board, keyPolygons } from "./keypolygons";

const UNIT: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const DEFAULT_SAMPLES = 12;
const MAX_SAMPLES = 100;
const REAR_PERTURBATION = { left: 0.18, right: 0.05 };
const DATA_ROOT = "/lab/data/key-instances/";

interface Sidecar {
  kind?: string;
  corners: Point[];
  imageWidth: number;
  imageHeight: number;
  instanceMask?: string;
  instances?: { id: number; color: number; pitch: number }[];
  pose?: { elevation: number; azimuth: number; distance: number; fov: number };
  camera?: { fov: number; distance: number };
}

interface Sample {
  name: string;
  sidecar: Sidecar;
  image: ImageData;
  target: Int16Array;
}

interface TrackResult {
  status: "evaluated" | "no-valid-results";
  attempted: number;
  accepted: number;
  meanCornerErrorPx: number | null;
  maxCornerErrorPx: number | null;
  meanKeyIoU: number | null;
  minKeyIoU: number | null;
  visibleKeyIoU: number | null;
  samples: SampleResult[];
}

interface SampleResult {
  name: string;
  camera: { fov: number | null; distance: number | null };
  proposalCornerErrorPx: { mean: number; max: number };
  calibratedCornerErrorPx: { mean: number | null; max: number | null };
  calibrated: boolean;
  board: Board | null;
  octaveOffset: number | null;
  meanKeyIoU: number | null;
  minKeyIoU: number | null;
  visibleKeyIoU: number | null;
}

interface EvaluationReport {
  kind: "kvt-calibration-evaluation";
  generatedAt: string;
  source: {
    directory: string;
    requestedSamples: number;
    availableSamples: number;
  };
  perturbation: {
    rearEdgeExtensionInUnitDepth: { left: number; right: number };
  };
  maskTarget: "rendered per-key instance IDs";
  tracks: {
    oracle: TrackResult;
    perturbed: TrackResult;
    detector: DetectorStatus;
  };
}

interface DetectorStatus {
  status: "not-evaluated";
  reason: string;
}

function queryPositive(name: string, fallback: number): number {
  const value = Number(new URLSearchParams(window.location.search).get(name));
  return Number.isSafeInteger(value) && value > 0 ? value : fallback;
}

function dataUrl(name: string): string {
  return `${DATA_ROOT}${encodeURIComponent(name)}`;
}

async function imageData(
  name: string,
  width: number,
  height: number,
): Promise<ImageData> {
  const response = await fetch(dataUrl(name));
  if (!response.ok) throw new Error(`cannot read ${name}: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) throw new Error("2d canvas context unavailable");
  context.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return context.getImageData(0, 0, width, height);
}

function instanceLabels(
  image: ImageData,
  instances: readonly { id: number; color: number; pitch: number }[],
): Int16Array {
  const byColor = new Map(
    instances.map((instance) => [instance.color, instance.pitch]),
  );
  const palette = instances.map((instance) => ({
    pitch: instance.pitch,
    red: instance.color >> 16,
    green: (instance.color >> 8) & 0xff,
    blue: instance.color & 0xff,
  }));
  const labels = new Int16Array(image.width * image.height).fill(-1);
  for (let at = 0, pixel = 0; pixel < labels.length; pixel += 1, at += 4) {
    const red = image.data[at];
    const green = image.data[at + 1];
    const blue = image.data[at + 2];
    const color = (red << 16) | (green << 8) | blue;
    const pitch = byColor.get(color);
    if (pitch !== undefined) {
      labels[pixel] = pitch;
      continue;
    }
    let nearestPitch = -1;
    let nearestDistance = 9;
    for (const candidate of palette) {
      const distance = Math.max(
        Math.abs(red - candidate.red),
        Math.abs(green - candidate.green),
        Math.abs(blue - candidate.blue),
      );
      if (distance < nearestDistance) {
        nearestPitch = candidate.pitch;
        nearestDistance = distance;
      }
    }
    labels[pixel] = nearestPitch;
  }
  return labels;
}

async function loadSamples(
  limit: number,
): Promise<{ samples: Sample[]; available: number }> {
  const response = await fetch("/lab/list/key-instances");
  if (!response.ok)
    throw new Error(`cannot list key-instances: ${response.status}`);
  const names = ((await response.json()) as unknown[])
    .filter(
      (name): name is string =>
        typeof name === "string" &&
        name.startsWith("keyinst-") &&
        name.endsWith(".json"),
    )
    .sort();
  const selected: Sample[] = [];
  for (const name of names) {
    if (selected.length >= limit) break;
    const sidecarResponse = await fetch(dataUrl(name));
    if (!sidecarResponse.ok) continue;
    const sidecar = (await sidecarResponse.json()) as Sidecar;
    if (
      sidecar.kind !== "key-instance-synth" ||
      sidecar.corners.length !== 4 ||
      !sidecar.instanceMask ||
      !sidecar.instances
    )
      continue;
    const imageName = `${name.slice(0, -5)}.png`;
    try {
      const image = await imageData(
        imageName,
        sidecar.imageWidth,
        sidecar.imageHeight,
      );
      const mask = await imageData(
        sidecar.instanceMask,
        sidecar.imageWidth,
        sidecar.imageHeight,
      );
      selected.push({
        name: imageName,
        sidecar,
        image,
        target: instanceLabels(mask, sidecar.instances),
      });
    } catch {
      // A partially written capture is ignored. The next run can consume it safely.
    }
  }
  return { samples: selected, available: names.length };
}

function perturbRearEdges(quad: readonly Point[]): Point[] {
  const h = findHomography(UNIT, quad);
  return [
    applyHomography(h, 0, -REAR_PERTURBATION.left),
    applyHomography(h, 1, -REAR_PERTURBATION.right),
    quad[2],
    quad[3],
  ];
}

function pixelQuad(
  quad: readonly Point[],
  width: number,
  height: number,
): Point[] {
  return quad.map((point) => ({ x: point.x * width, y: point.y * height }));
}

function cornerError(
  actual: readonly Point[],
  expected: readonly Point[],
): { mean: number; max: number } {
  const errors = actual.map((point, index) =>
    Math.hypot(point.x - expected[index].x, point.y - expected[index].y),
  );
  return {
    mean: errors.reduce((sum, value) => sum + value, 0) / errors.length,
    max: Math.max(...errors),
  };
}

function contains(point: Point, quad: readonly Point[]): boolean {
  let sign = 0;
  for (let index = 0; index < 4; index += 1) {
    const a = quad[index];
    const b = quad[(index + 1) % 4];
    const cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
    if (Math.abs(cross) < 0.0001) continue;
    const next = Math.sign(cross);
    if (sign !== 0 && next !== sign) return false;
    sign = next;
  }
  return sign !== 0;
}

function predictedLabels(
  quad: readonly Point[],
  board: Parameters<typeof keyPolygons>[1],
  width: number,
  height: number,
): Int16Array {
  const labels = new Int16Array(width * height).fill(-1);
  const polygons = keyPolygons(quad, board).sort(
    (a, b) => Number(a.black) - Number(b.black),
  );
  for (const key of polygons) {
    const xs = key.points.map((point) => point.x),
      ys = key.points.map((point) => point.y);
    const left = Math.max(0, Math.floor(Math.min(...xs))),
      right = Math.min(width - 1, Math.ceil(Math.max(...xs)));
    const top = Math.max(0, Math.floor(Math.min(...ys))),
      bottom = Math.min(height - 1, Math.ceil(Math.max(...ys)));
    for (let y = top; y <= bottom; y += 1)
      for (let x = left; x <= right; x += 1) {
        if (contains({ x: x + 0.5, y: y + 0.5 }, key.points))
          labels[y * width + x] = key.pitch;
      }
  }
  return labels;
}

function iou(
  predicted: Int16Array,
  target: Int16Array,
  predictedPitch?: number,
  targetPitch?: number,
): number {
  let intersection = 0;
  let union = 0;
  for (let index = 0; index < predicted.length; index += 1) {
    const predictedPixel =
      predictedPitch === undefined
        ? predicted[index] >= 0
        : predicted[index] === predictedPitch;
    const targetPixel =
      targetPitch === undefined
        ? target[index] >= 0
        : target[index] === targetPitch;
    if (predictedPixel || targetPixel) union += 1;
    if (predictedPixel && targetPixel) intersection += 1;
  }
  return union === 0 ? 1 : intersection / union;
}

function bestOctaveOffset(predicted: Int16Array, target: Int16Array): number {
  let offset = 0;
  let score = -1;
  for (let candidate = -120; candidate <= 120; candidate += 12) {
    let matched = 0;
    for (let index = 0; index < predicted.length; index += 1) {
      if (
        predicted[index] >= 0 &&
        predicted[index] + candidate === target[index]
      )
        matched += 1;
    }
    if (matched > score) {
      offset = candidate;
      score = matched;
    }
  }
  return offset;
}

function keyIou(
  predicted: Int16Array,
  target: Int16Array,
  predictedPitch: number,
  targetPitch: number,
): number | null {
  let occupied = false;
  for (let index = 0; index < predicted.length; index += 1) {
    if (predicted[index] === predictedPitch || target[index] === targetPitch) {
      occupied = true;
      break;
    }
  }
  return occupied ? iou(predicted, target, predictedPitch, targetPitch) : null;
}

function aggregate(results: SampleResult[], attempted: number): TrackResult {
  const accepted = results.filter((result) => result.calibrated);
  const values = <T extends number | null>(
    get: (result: SampleResult) => T,
  ): number[] =>
    accepted.flatMap((result) => {
      const value = get(result);
      return value === null || Number.isNaN(value) ? [] : [value];
    });
  const cornerMean = values((result) => result.calibratedCornerErrorPx.mean);
  const cornerMax = values((result) => result.calibratedCornerErrorPx.max);
  const meanIous = values((result) => result.meanKeyIoU);
  const minIous = values((result) => result.minKeyIoU);
  const visibleIous = values((result) => result.visibleKeyIoU);
  return {
    status: accepted.length === 0 ? "no-valid-results" : "evaluated",
    attempted,
    accepted: accepted.length,
    meanCornerErrorPx: cornerMean.length
      ? cornerMean.reduce((a, b) => a + b, 0) / cornerMean.length
      : null,
    maxCornerErrorPx: cornerMax.length ? Math.max(...cornerMax) : null,
    meanKeyIoU: meanIous.length
      ? meanIous.reduce((a, b) => a + b, 0) / meanIous.length
      : null,
    minKeyIoU: minIous.length ? Math.min(...minIous) : null,
    visibleKeyIoU: visibleIous.length
      ? visibleIous.reduce((a, b) => a + b, 0) / visibleIous.length
      : null,
    samples: results,
  };
}

function runTrack(
  samples: Sample[],
  proposalFor: (sample: Sample) => Point[],
): TrackResult {
  const results: SampleResult[] = [];
  for (const sample of samples) {
    const truth = pixelQuad(
      sample.sidecar.corners,
      sample.image.width,
      sample.image.height,
    );
    const proposal = pixelQuad(
      proposalFor(sample),
      sample.image.width,
      sample.image.height,
    );
    const proposalError = cornerError(proposal, truth);
    const fit = calibrateBoard(sample.image, proposal);
    const calibrated = fit !== null;
    const calibratedError = fit
      ? cornerError(fit.quad, truth)
      : { mean: null, max: null };
    const prediction = fit
      ? predictedLabels(
          fit.quad,
          fit.board,
          sample.image.width,
          sample.image.height,
        )
      : null;
    const pitches =
      sample.sidecar.instances?.map((instance) => instance.pitch) ?? [];
    const octaveOffset = prediction
      ? bestOctaveOffset(prediction, sample.target)
      : null;
    const keyIous = prediction
      ? pitches.flatMap((pitch) => {
          const value = keyIou(
            prediction,
            sample.target,
            pitch - (octaveOffset ?? 0),
            pitch,
          );
          return value === null ? [] : [value];
        })
      : [];
    results.push({
      name: sample.name,
      camera: {
        fov: sample.sidecar.camera?.fov ?? sample.sidecar.pose?.fov ?? null,
        distance:
          sample.sidecar.camera?.distance ??
          sample.sidecar.pose?.distance ??
          null,
      },
      proposalCornerErrorPx: proposalError,
      calibratedCornerErrorPx: calibratedError,
      calibrated,
      board: fit?.board ?? null,
      octaveOffset,
      meanKeyIoU: keyIous.length
        ? keyIous.reduce((sum, value) => sum + value, 0) / keyIous.length
        : null,
      minKeyIoU: keyIous.length ? Math.min(...keyIous) : null,
      visibleKeyIoU: prediction
        ? iou(prediction, sample.target, undefined, undefined)
        : null,
    });
  }
  return aggregate(results, samples.length);
}

async function run(): Promise<EvaluationReport> {
  const requestedSamples = Math.min(
    queryPositive("samples", DEFAULT_SAMPLES),
    MAX_SAMPLES,
  );
  const loaded = await loadSamples(requestedSamples);
  const oracle = runTrack(loaded.samples, (sample) => sample.sidecar.corners);
  const perturbed = runTrack(loaded.samples, (sample) =>
    perturbRearEdges(sample.sidecar.corners),
  );
  return {
    kind: "kvt-calibration-evaluation",
    generatedAt: new Date().toISOString(),
    source: {
      directory: "data/key-instances",
      requestedSamples,
      availableSamples: loaded.available,
    },
    perturbation: { rearEdgeExtensionInUnitDepth: REAR_PERTURBATION },
    maskTarget: "rendered per-key instance IDs",
    tracks: {
      oracle,
      perturbed,
      detector: {
        status: "not-evaluated",
        reason:
          "No detector proposal was supplied; this run evaluates geometry calibration only.",
      },
    },
  };
}

async function main(): Promise<void> {
  const reportElement = document.querySelector<HTMLPreElement>("#report");
  try {
    const report = await run();
    const body = JSON.stringify(report, null, 2);
    if (reportElement) reportElement.textContent = body;
    const runName =
      new URLSearchParams(window.location.search).get("run") ??
      "calibration-evaluation";
    const response = await fetch(
      `/lab/save/evaluations/${encodeURIComponent(runName)}.json`,
      { method: "POST", headers: { "Content-Type": "application/json" }, body },
    );
    if (!response.ok) throw new Error(`cannot save report: ${response.status}`);
    document.title = "evaluation complete";
  } catch (error: unknown) {
    const body = JSON.stringify(
      {
        kind: "kvt-calibration-evaluation",
        error: error instanceof Error ? error.message : String(error),
      },
      null,
      2,
    );
    if (reportElement) reportElement.textContent = body;
    document.title = "evaluation failed";
    throw error;
  }
}

void main();
