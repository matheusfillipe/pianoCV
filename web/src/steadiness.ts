import type { Point } from "./homography";
import { nearestOnOutline, outlinePoints } from "./keyseg";
import { otsu } from "./keystrip";

/** How still and how well placed the drawn keys are on a live page, for the lab: the camera of
 * a recording does not move, so any motion of a drawn key is the pipeline's own. */
export type Steadiness = {
  readonly frames: number;
  readonly keys: number;
  /** How far a drawn key's outline moves from one frame to the next, in video pixels: each point
   * of it against the nearest point of the same key's outline the frame before. */
  readonly movePxMedian: number | null;
  readonly movePxP95: number | null;
  readonly movePxP999: number | null;
  /** The share of frames in which some key corner moved more than a pixel. */
  readonly jumpyFrames: number | null;
  /** How far each key's centre wanders around its own mean, in video pixels. */
  readonly swingPxMedian: number | null;
  readonly swingPxP95: number | null;
  /** How well each drawn black key overlaps the black key the segmenter found under it, over
   * the keys the segmenter found on their own. */
  readonly blackIoUMedian: number | null;
  readonly blackIoUP10: number | null;
  /** The same overlap for the template keys before they were snapped. */
  readonly templateIoUMedian: number | null;
  readonly templateIoUP10: number | null;
  /** The share of black keys the segmenter found on their own, unmerged. */
  readonly blackSingleShare: number | null;
  /** For the drawn keys and the template they were snapped from, the share of each black key
   * that is dark in the video and of the white keys' visible surface that is bright, split at
   * the keybed's own Otsu threshold. */
  readonly fit: Readonly<Record<string, FitSummary>>;
};

type FitSummary = {
  readonly blackDarkMedian: number | null;
  readonly blackDarkP10: number | null;
  readonly whiteBright: number | null;
};

type Drawn = {
  readonly black: boolean;
  readonly semitone: number;
  readonly bar: readonly Point[];
};

const RASTER_SCALE = 2;
const IOU_EVERY = 20;
const EVEN_POINTS = 40;
// a region this much bigger than the key holds several keys, as snapKeys judges it
const ONE_KEY_MOST_AREA = 1.6;

const wait = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));
const nextFrame = (): Promise<number> =>
  new Promise((resolve) => requestAnimationFrame(resolve));

function quantile(values: readonly number[], share: number): number | null {
  if (values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  return (
    Math.round(sorted[Math.floor(share * (sorted.length - 1))] * 100) / 100
  );
}

function rasterise(
  ctx: OffscreenCanvasRenderingContext2D,
  bar: readonly Point[],
): Set<number> {
  const { width, height } = ctx.canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.beginPath();
  bar.forEach((p, i) => {
    if (i === 0) {
      ctx.moveTo(p.x * width, p.y * height);
    } else {
      ctx.lineTo(p.x * width, p.y * height);
    }
  });
  ctx.closePath();
  ctx.fill();
  const xs = bar.map((p) => p.x * width);
  const ys = bar.map((p) => p.y * height);
  const left = Math.max(0, Math.floor(Math.min(...xs)));
  const top = Math.max(0, Math.floor(Math.min(...ys)));
  const w = Math.min(width, Math.ceil(Math.max(...xs))) - left;
  const h = Math.min(height, Math.ceil(Math.max(...ys))) - top;
  const inside = new Set<number>();
  if (w <= 0 || h <= 0) {
    return inside;
  }
  const alpha = ctx.getImageData(left, top, w, h).data;
  for (let i = 0; i < w * h; i += 1) {
    if (alpha[i * 4 + 3] > 127) {
      inside.add((top + Math.floor(i / w)) * width + left + (i % w));
    }
  }
  return inside;
}

/** The overlap with the region sharing most pixels with the key, or null when that region is
 * missing or holds several keys the segmenter merged. */
function singleOverlap(
  drawn: Set<number>,
  found: readonly Set<number>[],
): number | null {
  let most = 0;
  let best: Set<number> | null = null;
  for (const region of found) {
    let shared = 0;
    for (const i of drawn) {
      if (region.has(i)) {
        shared += 1;
      }
    }
    if (shared > most) {
      most = shared;
      best = region;
    }
  }
  if (best === null || best.size > drawn.size * ONE_KEY_MOST_AREA) {
    return null;
  }
  return most / (drawn.size + best.size - most);
}

type PixelFit = {
  readonly blackDark: number[];
  readonly whiteBright: number;
};

function pixelFit(
  ctx: OffscreenCanvasRenderingContext2D,
  luminance: Float32Array,
  keys: readonly Drawn[],
): PixelFit {
  const whites = keys.filter((k) => !k.black).map((k) => rasterise(ctx, k.bar));
  const blacks = keys.filter((k) => k.black).map((k) => rasterise(ctx, k.bar));
  const keybed = new Set<number>();
  for (const set of [...whites, ...blacks]) {
    for (const i of set) {
      keybed.add(i);
    }
  }
  const threshold = otsu([...keybed].map((i) => luminance[i]));
  const covered = new Set<number>();
  for (const set of blacks) {
    for (const i of set) {
      covered.add(i);
    }
  }
  let visible = 0;
  let bright = 0;
  for (const set of whites) {
    for (const i of set) {
      if (!covered.has(i)) {
        visible += 1;
        bright += luminance[i] >= threshold ? 1 : 0;
      }
    }
  }
  return {
    blackDark: blacks
      .filter((set) => set.size > 0)
      .map(
        (set) =>
          [...set].filter((i) => luminance[i] < threshold).length / set.size,
      ),
    whiteBright: visible ? bright / visible : 0,
  };
}

const id = (key: Drawn): string => `${key.black}${key.semitone}`;

export async function measureSteadiness(
  durationMs = 5000,
  settleMs = 3000,
): Promise<Steadiness> {
  const toggle = [...document.querySelectorAll("button")].find(
    (b) => b.textContent?.trim() === "keys",
  );
  for (let i = 0; i < 300 && !window.pianocvDrawnKeys?.length; i += 1) {
    if (i === 0) {
      toggle?.click();
    }
    await wait(100);
  }
  await wait(settleMs);
  const video = document.querySelector("video");
  const width = video?.videoWidth ?? 0;
  const height = video?.videoHeight ?? 0;
  const canvas = new OffscreenCanvas(
    Math.max(1, Math.round(width * RASTER_SCALE)),
    Math.max(1, Math.round(height * RASTER_SCALE)),
  );
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const picture = new OffscreenCanvas(canvas.width, canvas.height);
  const pictureCtx = picture.getContext("2d", { willReadFrequently: true });
  const luminance = new Float32Array(canvas.width * canvas.height);
  const fits = new Map<string, { dark: number[]; bright: number[] }>();
  const moves: number[] = [];
  const centres = new Map<string, Point[]>();
  const overlaps: number[] = [];
  const templateOverlaps: number[] = [];
  let blackSeen = 0;
  let blackSingle = 0;
  let previous: readonly Drawn[] | null = null;
  let frames = 0;
  let compared = 0;
  let jumpy = 0;
  const started = performance.now();
  while (performance.now() - started < durationMs) {
    await nextFrame();
    const keys = window.pianocvDrawnKeys;
    if (!keys || keys === previous) {
      continue;
    }
    // the drawn outlines are simplified to however many corners each frame needs, so we spread
    // points evenly over each and measure them against the outline drawn the frame before
    const even = keys.map((key) => ({
      ...key,
      bar: outlinePoints(key.bar, EVEN_POINTS),
    }));
    const movesBefore = moves.length;
    const pixels = (bar: readonly Point[]): Point[] =>
      bar.map((p) => ({ x: p.x * width, y: p.y * height }));
    const before = new Map(
      (previous ?? []).map((key) => [id(key), pixels(key.bar)]),
    );
    for (const key of even) {
      const outline = before.get(id(key));
      if (outline) {
        for (const p of pixels(key.bar)) {
          const q = nearestOnOutline(p, outline);
          moves.push(Math.hypot(q.x - p.x, q.y - p.y));
        }
      }
    }
    if (previous !== null) {
      compared += 1;
      jumpy += moves.slice(movesBefore).some((move) => move > 1) ? 1 : 0;
    }
    for (const key of even) {
      const centre = {
        x: (key.bar.reduce((s, p) => s + p.x, 0) / key.bar.length) * width,
        y: (key.bar.reduce((s, p) => s + p.y, 0) / key.bar.length) * height,
      };
      centres.set(id(key), [...(centres.get(id(key)) ?? []), centre]);
    }
    if (ctx && pictureCtx && video && frames % IOU_EVERY === 0) {
      pictureCtx.drawImage(video, 0, 0, picture.width, picture.height);
      const rgb = pictureCtx.getImageData(
        0,
        0,
        picture.width,
        picture.height,
      ).data;
      for (let i = 0; i < luminance.length; i += 1) {
        luminance[i] =
          0.299 * rgb[i * 4] + 0.587 * rgb[i * 4 + 1] + 0.114 * rgb[i * 4 + 2];
      }
      const sets = { drawn: keys, template: window.pianocvTemplateKeys ?? [] };
      for (const [name, set] of Object.entries(sets)) {
        if (set.length === 0) {
          continue;
        }
        const fit = pixelFit(ctx, luminance, set);
        const into = fits.get(name) ?? { dark: [], bright: [] };
        into.dark.push(...fit.blackDark);
        into.bright.push(fit.whiteBright);
        fits.set(name, into);
      }
      const found = (window.pianocvLiveKeys?.regions ?? [])
        .filter((region) => region.black)
        .map((region) => rasterise(ctx, region.bar));
      for (const key of keys.filter((k) => k.black)) {
        const overlap = singleOverlap(rasterise(ctx, key.bar), found);
        blackSeen += 1;
        if (overlap !== null) {
          blackSingle += 1;
          overlaps.push(overlap);
        }
      }
      for (const key of (window.pianocvTemplateKeys ?? []).filter(
        (k) => k.black,
      )) {
        const overlap = singleOverlap(rasterise(ctx, key.bar), found);
        if (overlap !== null) {
          templateOverlaps.push(overlap);
        }
      }
    }
    previous = keys;
    frames += 1;
  }
  const swing = [...centres.values()].map((points) => {
    const mx = points.reduce((s, p) => s + p.x, 0) / points.length;
    const my = points.reduce((s, p) => s + p.y, 0) / points.length;
    return Math.sqrt(
      points.reduce((s, p) => s + (p.x - mx) ** 2 + (p.y - my) ** 2, 0) /
        points.length,
    );
  });
  return {
    frames,
    keys: previous?.length ?? 0,
    movePxMedian: quantile(moves, 0.5),
    movePxP95: quantile(moves, 0.95),
    movePxP999: quantile(moves, 0.999),
    jumpyFrames: compared ? Math.round((jumpy / compared) * 1000) / 1000 : null,
    swingPxMedian: quantile(swing, 0.5),
    swingPxP95: quantile(swing, 0.95),
    blackIoUMedian: quantile(overlaps, 0.5),
    blackIoUP10: quantile(overlaps, 0.1),
    templateIoUMedian: quantile(templateOverlaps, 0.5),
    templateIoUP10: quantile(templateOverlaps, 0.1),
    fit: Object.fromEntries(
      [...fits].map(([name, { dark, bright }]) => [
        name,
        {
          blackDarkMedian: quantile(dark, 0.5),
          blackDarkP10: quantile(dark, 0.1),
          whiteBright: quantile(bright, 0.5),
        },
      ]),
    ),
    blackSingleShare: blackSeen
      ? Math.round((blackSingle / blackSeen) * 100) / 100
      : null,
  };
}
