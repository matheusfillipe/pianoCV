import { applyHomography, findHomography, type Point } from "./homography";
import type { Board } from "./keypolygons";
import { isBlack, keyUnits } from "./keys";

const SAMPLES = 360;
const DEPTHS = [0.18, 0.26, 0.34, 0.42, 0.5];
const BLEEDS = [0, 0.1, 0.2, 0.3, 0.4];
const MIN_AGREEMENT = 0.62;
const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];
const WHITE_PITCHES = Array.from(
  { length: 88 },
  (_, index) => index + 21,
).filter((pitch) => !isBlack(pitch));

function brightness(pixels: ImageData, point: Point): number | null {
  const x = Math.round(point.x);
  const y = Math.round(point.y);
  if (x < 1 || y < 1 || x >= pixels.width - 1 || y >= pixels.height - 1) {
    return null;
  }
  const at = (y * pixels.width + x) * 4;
  return (
    0.299 * (pixels.data[at] ?? 0) +
    0.587 * (pixels.data[at + 1] ?? 0) +
    0.114 * (pixels.data[at + 2] ?? 0)
  );
}

function threshold(values: readonly number[]): number | null {
  if (values.length < 40) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
}

function expected(unit: number, bleed: number): boolean {
  for (let pitch = 60; pitch < 72; pitch += 1) {
    if (!isBlack(pitch)) continue;
    const key = keyUnits(pitch);
    const from = (((key.from - bleed) % 7) + 7) % 7;
    const to = (((key.to + bleed) % 7) + 7) % 7;
    const at = ((unit % 7) + 7) % 7;
    if (from < to ? at >= from && at <= to : at >= from || at <= to)
      return true;
  }
  return false;
}

function rangeFor(whites: number, phase: number): Board | null {
  let best: Board | null = null;
  for (const [index, lowest] of WHITE_PITCHES.entries()) {
    const highest = WHITE_PITCHES[index + whites - 1];
    if (
      lowest === undefined ||
      highest === undefined ||
      keyUnits(lowest).from % 7 !== phase
    )
      continue;
    const candidate = {
      lowest,
      highest,
      origin: keyUnits(lowest).from,
      span: whites,
    };
    if (
      best === null ||
      Math.abs((lowest + highest) / 2 - 64.5) <
        Math.abs((best.lowest + best.highest) / 2 - 64.5)
    )
      best = candidate;
  }
  return best;
}

export function fitBoard(
  pixels: ImageData,
  quad: readonly Point[],
): Board | null {
  const homography = findHomography(UNIT_SQUARE, quad);
  let best: { board: Board; agreement: number } | null = null;
  for (const depth of DEPTHS) {
    const levels = Array.from({ length: SAMPLES }, (_, index) =>
      brightness(
        pixels,
        applyHomography(homography, (index + 0.5) / SAMPLES, depth),
      ),
    );
    const cut = threshold(
      levels.flatMap((value) => (value === null ? [] : [value])),
    );
    if (cut === null) continue;
    for (let whites = 12; whites <= 56; whites += 1)
      for (let phase = 0; phase < 7; phase += 1)
        for (const bleed of BLEEDS) {
          const board = rangeFor(whites, phase);
          if (board === null) continue;
          let matched = 0;
          let known = 0;
          for (const [index, level] of levels.entries()) {
            if (level === null) continue;
            known += 1;
            if (
              level < cut ===
              expected(phase + ((index + 0.5) / SAMPLES) * whites, bleed)
            )
              matched += 1;
          }
          const agreement = known === 0 ? 0 : matched / known;
          if (best === null || agreement > best.agreement)
            best = { board, agreement };
        }
  }
  return best !== null && best.agreement >= MIN_AGREEMENT ? best.board : null;
}
