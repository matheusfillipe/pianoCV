import { applyHomography, type Point } from "./homography";
import {
  type KeyNetFit,
  type KeyNetPeaks,
  keyboardTemplate,
  liftPoints,
  refineHomography,
} from "./keycore";
import type { Size } from "./keyspace";

/** A point in frame pixels, or null where it is hidden by a hand, the case, or off frame. */
export type LabelPoint = [number, number] | null;

export const POINT_KINDS = [
  "corners",
  "gaps",
  "blackLow",
  "blackHigh",
  "blackTopLow",
  "blackTopHigh",
  "backGaps",
  "blackBackLow",
  "blackBackHigh",
] as const;
export type PointKind = (typeof POINT_KINDS)[number];

export type FixedLabels = Record<PointKind, LabelPoint[]> & {
  width: number;
  height: number;
  whiteKeys: number;
  phase: string;
};

type Locator = Pick<KeyNetFit, "homography" | "whiteKeys" | "phase" | "lift">;
type FrontTops = Pick<KeyNetPeaks, "blackTopLow" | "blackTopHigh">;
type BackPeaks = Pick<KeyNetPeaks, "blackBackLow" | "blackBackHigh">;
type RearKind = "backGaps" | "blackBackLow" | "blackBackHigh";
export type RearLabels = Pick<FixedLabels, RearKind>;

export function emptyLabels(size: Size): FixedLabels {
  return {
    width: size.width,
    height: size.height,
    whiteKeys: 0,
    phase: "C",
    corners: [null, null, null, null],
    gaps: [],
    blackLow: [],
    blackHigh: [],
    blackTopLow: [],
    blackTopHigh: [],
    backGaps: [],
    blackBackLow: [],
    blackBackHigh: [],
  };
}

export function serialise(labels: FixedLabels): string {
  return JSON.stringify(labels);
}

const nulls = (length: number): LabelPoint[] =>
  Array.from({ length }, () => null);

/** Files saved before the rear kinds existed parse with those kinds all hidden. */
export function parseLabels(text: string): FixedLabels {
  const labels = JSON.parse(text) as FixedLabels;
  return {
    ...labels,
    backGaps: labels.backGaps ?? nulls(labels.gaps.length),
    blackBackLow: labels.blackBackLow ?? nulls(labels.blackLow.length),
    blackBackHigh: labels.blackBackHigh ?? nulls(labels.blackHigh.length),
  };
}

export function lacksRear(text: string): boolean {
  return (JSON.parse(text) as Partial<FixedLabels>).backGaps === undefined;
}

function nearestPeak(
  bottom: [number, number],
  peaks: readonly Point[],
  size: Size,
  within: number,
): LabelPoint {
  let best: LabelPoint = null;
  let bestDistance = within;
  for (const peak of peaks) {
    const x = peak.x * size.width;
    const y = peak.y * size.height;
    const distance = Math.hypot(x - bottom[0], y - bottom[1]);
    if (distance <= bestDistance) {
      best = [x, y];
      bestDistance = distance;
    }
  }
  return best;
}

function toPixels(
  fit: Pick<Locator, "homography">,
  size: Size,
  p: Point,
): [number, number] {
  const q = applyHomography(fit.homography, p.x, p.y);
  return [q.x * size.width, q.y * size.height];
}

function keyWidthPx(fit: Locator, size: Size): number {
  const [low, high] = [
    toPixels(fit, size, { x: 0, y: 1 }),
    toPixels(fit, size, { x: 1, y: 1 }),
  ];
  return Math.hypot(high[0] - low[0], high[1] - low[1]);
}

/** The rear labels the fit implies: back gaps on the template through the homography, and for each
 * black key the back-top peak nearest its lifted template point within one white key, else that
 * lifted point itself when the fit has a lift. */
export function prefillRear(
  fit: Locator,
  peaks: BackPeaks,
  size: Size,
): RearLabels {
  const template = keyboardTemplate(fit.whiteKeys, fit.phase);
  const keyPx = keyWidthPx(fit, size);
  const backTops = (
    bottoms: readonly Point[],
    found: readonly Point[],
  ): LabelPoint[] => {
    const lift = fit.lift;
    const expected =
      lift === null
        ? null
        : liftPoints(
            fit.homography,
            lift,
            bottoms.map((p) => ({ x: p.x, y: 0 })),
          );
    return bottoms.map((_, i) => {
      const at = expected?.[i];
      if (at === undefined) {
        return null;
      }
      const lifted: [number, number] = [at.x * size.width, at.y * size.height];
      return nearestPeak(lifted, found, size, keyPx) ?? lifted;
    });
  };
  return {
    backGaps: template.backGaps.map((p) => toPixels(fit, size, p)),
    blackBackLow: backTops(template.blackLow, peaks.blackBackLow),
    blackBackHigh: backTops(template.blackHigh, peaks.blackBackHigh),
  };
}

/** The labels the app's own fit implies: the template through the fit's homography, and for each
 * black key the nearest top peak within one white key of its bottom. */
export function prefill(
  fit: Locator,
  tops: FrontTops & BackPeaks,
  size: Size,
): FixedLabels {
  const template = keyboardTemplate(fit.whiteKeys, fit.phase);
  const keyPx = keyWidthPx(fit, size);
  const blackLow = template.blackLow.map((p) => toPixels(fit, size, p));
  const blackHigh = template.blackHigh.map((p) => toPixels(fit, size, p));
  const topsOf = (bottoms: [number, number][], peaks: readonly Point[]) =>
    bottoms.map((bottom) => nearestPeak(bottom, peaks, size, keyPx));
  return {
    width: size.width,
    height: size.height,
    whiteKeys: fit.whiteKeys,
    phase: fit.phase,
    corners: template.corners.map((p) => toPixels(fit, size, p)),
    gaps: template.gaps.map((p) => toPixels(fit, size, p)),
    blackLow,
    blackHigh,
    blackTopLow: topsOf(blackLow, tops.blackTopLow),
    blackTopHigh: topsOf(blackHigh, tops.blackTopHigh),
    ...prefillRear(fit, tops, size),
  };
}

function followTops(
  tops: LabelPoint[],
  before: LabelPoint[],
  after: LabelPoint[],
  fillHidden: boolean,
): LabelPoint[] {
  return tops.map((top, i) => {
    const [from, to] = [before[i], after[i]];
    if (top === null) {
      return fillHidden ? to : null;
    }
    if (from === null) {
      return top;
    }
    return to === null
      ? null
      : [top[0] + to[0] - from[0], top[1] + to[1] - from[1]];
  });
}

/** Moves every gap and black point `keys` template indices along, through a least-squares
 * homography from the template onto the points that are visible now; 0 refits them in place.
 * Hidden points stay null unless `fillHidden`, which gives them a projected position instead. */
export function reproject(
  labels: FixedLabels,
  keys: number,
  fillHidden = false,
): FixedLabels {
  const template = keyboardTemplate(labels.whiteKeys, labels.phase);
  const targets: Record<
    "gaps" | "backGaps" | "blackLow" | "blackHigh",
    readonly Point[]
  > = {
    gaps: template.gaps,
    backGaps: template.backGaps,
    blackLow: template.blackLow,
    blackHigh: template.blackHigh,
  };
  const pairs = (Object.keys(targets) as (keyof typeof targets)[]).flatMap(
    (kind) =>
      labels[kind].flatMap((p, i) =>
        p === null
          ? []
          : [{ src: targets[kind][i], dst: { x: p[0], y: p[1] } }],
      ),
  );
  const homography = pairs.length < 4 ? null : refineHomography(pairs);
  if (homography === null) {
    return structuredClone(labels);
  }
  const project = (src: Point | undefined): LabelPoint => {
    if (src === undefined) {
      return null;
    }
    const q = applyHomography(homography, src.x, src.y);
    return [q.x, q.y];
  };
  const moved = (kind: keyof typeof targets): LabelPoint[] =>
    labels[kind].map((p, i) => {
      if (p === null && !fillHidden) {
        return null;
      }
      return kind === "gaps" || kind === "backGaps"
        ? project({ x: targets[kind][i].x + keys, y: targets[kind][i].y })
        : project(targets[kind][i + keys]);
    });
  const blackLow = moved("blackLow");
  const blackHigh = moved("blackHigh");
  return {
    ...labels,
    gaps: moved("gaps"),
    backGaps: moved("backGaps"),
    blackLow,
    blackHigh,
    blackTopLow: followTops(
      labels.blackTopLow,
      labels.blackLow,
      blackLow,
      fillHidden,
    ),
    blackTopHigh: followTops(
      labels.blackTopHigh,
      labels.blackHigh,
      blackHigh,
      fillHidden,
    ),
    blackBackLow: followTops(
      labels.blackBackLow,
      labels.blackLow,
      blackLow,
      fillHidden,
    ),
    blackBackHigh: followTops(
      labels.blackBackHigh,
      labels.blackHigh,
      blackHigh,
      fillHidden,
    ),
  };
}
