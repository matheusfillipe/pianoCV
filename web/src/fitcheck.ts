import { applyHomography, type Point } from "./homography";
import { type KeyNetFit, type KeyNetPeaks, keyboardTemplate } from "./keycore";
import { createKeyNet, KEYNET_URL, type KeyNetRunner } from "./keynetrunner";
import { createKeyNetSession } from "./keynetsession";
import type { Size } from "./keyspace";
import { viteAssets } from "./viteassets";

/** A labelled frame's keypoints in frame pixels, low key to high, as `make lab-keynet-eval
 * ARGS="--points-out ..."` writes them. */
type LabelPoints = {
  readonly width: number;
  readonly height: number;
  readonly whiteKeys: number;
  readonly corners: readonly (readonly number[])[];
  readonly gaps: readonly (readonly number[])[];
  readonly blackLow: readonly (readonly number[])[];
  readonly blackHigh: readonly (readonly number[])[];
};

export type FitCheck = {
  readonly frame: string;
  readonly acquired: boolean;
  readonly board: string | null;
  readonly rightBoard: boolean;
  /** How many white keys the fitted numbering sits off the labels, 0 when it is right. */
  readonly shift: number | null;
  /** Median errors in white-key widths with the numbering as fitted; corners are the front two. */
  readonly gapKeys: number | null;
  readonly blackKeys: number | null;
  readonly cornerKeys: number | null;
};

const POINTS_URL = "/lab/data/evaluations/real-keys-points.json";
const MOST_STEPS = 80;
// steps of tracking before a fit is scored, so the smoothing has settled on the still
const SETTLE_STEPS = 12;
const FRAME_MS = 33;
const SHIFTS = [-2, -1, 0, 1, 2];

function median(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function errorKeys(
  fitted: readonly Point[],
  labels: readonly (readonly number[])[],
  shift: number,
  keyPx: number,
): number | null {
  const errors: number[] = [];
  fitted.forEach((p, i) => {
    const label = labels[i + shift];
    if (label !== undefined) {
      errors.push(Math.hypot(p.x - label[0], p.y - label[1]) / keyPx);
    }
  });
  return median(errors);
}

/** How the fitted board lands on one frame's labels. */
export function checkFit(
  frame: string,
  fit: KeyNetFit | null,
  labels: LabelPoints,
): FitCheck {
  if (fit === null) {
    return {
      frame,
      acquired: false,
      board: null,
      rightBoard: false,
      shift: null,
      gapKeys: null,
      blackKeys: null,
      cornerKeys: null,
    };
  }
  const toPx = (p: Point): Point => {
    const q = applyHomography(fit.homography, p.x, p.y);
    return { x: q.x * labels.width, y: q.y * labels.height };
  };
  const [, , frontHigh, frontLow] = labels.corners;
  const keyPx =
    Math.hypot(frontHigh[0] - frontLow[0], frontHigh[1] - frontLow[1]) /
    labels.whiteKeys;
  const template = keyboardTemplate(fit.whiteKeys, fit.phase);
  const gaps = template.gaps.map(toPx);
  const rightBoard = fit.whiteKeys === labels.whiteKeys;
  const shift = rightBoard
    ? SHIFTS.reduce((best, s) =>
        (errorKeys(gaps, labels.gaps, s, keyPx) ?? Number.POSITIVE_INFINITY) <
        (errorKeys(gaps, labels.gaps, best, keyPx) ?? Number.POSITIVE_INFINITY)
          ? s
          : best,
      )
    : null;
  const blacks = [...template.blackLow, ...template.blackHigh].map(toPx);
  const blackLabels = [...labels.blackLow, ...labels.blackHigh];
  return {
    frame,
    acquired: true,
    board: `${fit.whiteKeys}${fit.phase}`,
    rightBoard,
    shift,
    gapKeys: rightBoard ? errorKeys(gaps, labels.gaps, 0, keyPx) : null,
    blackKeys:
      blacks.length === blackLabels.length
        ? errorKeys(blacks, blackLabels, 0, keyPx)
        : null,
    // the back corners hide under the case on a real keyboard, so the fit only extrapolates
    // them and only the front ones are scored
    cornerKeys: errorKeys(
      template.corners.slice(2).map(toPx),
      labels.corners.slice(2),
      0,
      keyPx,
    ),
  };
}

async function loadFrame(name: string): Promise<ImageBitmap> {
  const response = await fetch(`/lab/data/real-keys/${name}.png`);
  return createImageBitmap(await response.blob());
}

/** Runs the app's own KeyNet loop on a still from a cold start as a camera would begin, and
 * returns the board it settles on with the peaks of the last run. */
export async function fitStill(
  keyNet: KeyNetRunner,
  image: ImageBitmap,
): Promise<{ fit: KeyNetFit | null; peaks: KeyNetPeaks | null }> {
  const size: Size = { width: image.width, height: image.height };
  const session = createKeyNetSession(keyNet);
  let tracked = 0;
  for (let step = 0; step < MOST_STEPS && tracked < SETTLE_STEPS; step += 1) {
    await session.step(image, size, step * FRAME_MS);
    tracked = session.fit() === null ? 0 : tracked + 1;
  }
  return { fit: session.fit(), peaks: session.peaks() };
}

/** Runs the app's own KeyNet loop on every labelled frame of the given recordings, each from a
 * cold start as a camera would begin, and scores the board it settles on. */
export async function checkFits(
  clips: readonly string[],
  modelUrl: string = KEYNET_URL,
): Promise<FitCheck[]> {
  const labels = (await (await fetch(POINTS_URL)).json()) as Record<
    string,
    LabelPoints
  >;
  const keyNet = await createKeyNet(viteAssets, modelUrl);
  const checks: FitCheck[] = [];
  for (const [frame, points] of Object.entries(labels)) {
    if (!clips.some((clip) => frame.startsWith(clip))) {
      continue;
    }
    const image = await loadFrame(frame);
    checks.push(checkFit(frame, (await fitStill(keyNet, image)).fit, points));
    image.close();
  }
  return checks;
}

export type FitSummary = {
  readonly frames: number;
  readonly acquired: number;
  readonly rightBoard: number;
  readonly rightNumbering: number;
  readonly gapKeys: number | null;
  readonly blackKeys: number | null;
  readonly cornerKeys: number | null;
};

export function summarise(checks: readonly FitCheck[]): FitSummary {
  const numbered = checks.filter((c) => c.shift === 0);
  const values = (pick: (c: FitCheck) => number | null): number[] =>
    numbered.map(pick).filter((v): v is number => v !== null);
  const share = (n: number): number =>
    checks.length ? Math.round((n / checks.length) * 100) / 100 : 0;
  const round = (v: number | null): number | null =>
    v === null ? null : Math.round(v * 1000) / 1000;
  return {
    frames: checks.length,
    acquired: share(checks.filter((c) => c.acquired).length),
    rightBoard: share(checks.filter((c) => c.rightBoard).length),
    rightNumbering: share(numbered.length),
    gapKeys: round(median(values((c) => c.gapKeys))),
    blackKeys: round(median(values((c) => c.blackKeys))),
    cornerKeys: round(median(values((c) => c.cornerKeys))),
  };
}
