import type { Point } from "./homography";

const STORAGE_KEY = "pianocv-corners";
const HIT_RADIUS_PX = 24;
const HANDLE_RADIUS_PX = 9;
const HINT_TEXT = "drag corners onto the keyboard, press c to hide";

export type Corners = [Point, Point, Point, Point];

function edgeLength(quad: Point[], index: number): number {
  return Math.hypot(quad[index].x - quad[0].x, quad[index].y - quad[0].y);
}

/** Rolls the quad so edge 0 to 1 is its longer side, which runs along the white keys. The
 * remaining half turn is the dragger's choice: handle 1 to 2 runs along the black keys. */
export function canonicalQuad(quad: Point[]): Point[] {
  return edgeLength(quad, 1) >= edgeLength(quad, 3)
    ? quad
    : quad.map((_, i) => quad[(i + 1) % quad.length]);
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Calibration {
  getCorners(): Corners;
  setCorners(corners: readonly Point[]): void;
  draw(ctx: CanvasRenderingContext2D, w: number, h: number): void;
}

function defaultCorners(): Corners {
  return [
    { x: 0.25, y: 0.3 },
    { x: 0.75, y: 0.3 },
    { x: 0.75, y: 0.7 },
    { x: 0.25, y: 0.7 },
  ];
}

function isPoint(value: unknown): value is Point {
  return (
    typeof value === "object" &&
    value !== null &&
    "x" in value &&
    "y" in value &&
    typeof value.x === "number" &&
    Number.isFinite(value.x) &&
    typeof value.y === "number" &&
    Number.isFinite(value.y)
  );
}

function loadCorners(): Corners | null {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw) as unknown;
  } catch {
    return null;
  }
  if (!Array.isArray(parsed) || parsed.length !== 4) {
    return null;
  }
  const points: Point[] = [];
  for (const item of parsed) {
    if (!isPoint(item)) {
      return null;
    }
    points.push(item);
  }
  const corners: Corners = [points[0], points[1], points[2], points[3]];
  return corners;
}

function saveCorners(corners: Corners): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(corners));
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function createCalibration(
  canvas: HTMLCanvasElement,
  isActive: () => boolean,
  // the handles are drawn inside the letterboxed video, so they have to be grabbed there
  // too; hit testing against the whole canvas misses them by the width of the black bars
  getBox: () => Box,
): Calibration {
  const saved = loadCorners();
  const corners: Corners = saved ?? defaultCorners();
  let dragging: number | null = null;

  const hitCorner = (px: number, py: number): number | null => {
    const box = getBox();
    let best: number | null = null;
    let bestDist = HIT_RADIUS_PX;
    for (const [i, corner] of corners.entries()) {
      const dx = px - (box.x + corner.x * box.w);
      const dy = py - (box.y + corner.y * box.h);
      const dist = Math.hypot(dx, dy);
      if (dist <= bestDist) {
        best = i;
        bestDist = dist;
      }
    }
    return best;
  };

  const release = (): void => {
    if (dragging === null) {
      return;
    }
    dragging = null;
    saveCorners(corners);
  };

  canvas.addEventListener("pointerdown", (event) => {
    if (!isActive() || dragging !== null) {
      return;
    }
    const index = hitCorner(event.clientX, event.clientY);
    if (index === null) {
      return;
    }
    dragging = index;
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (dragging === null) {
      return;
    }
    const box = getBox();
    corners[dragging] = {
      x: clamp01((event.clientX - box.x) / box.w),
      y: clamp01((event.clientY - box.y) / box.h),
    };
  });
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  return {
    getCorners: () => corners,
    setCorners: (next) => {
      for (let i = 0; i < corners.length; i += 1) {
        corners[i] = { x: next[i].x, y: next[i].y };
      }
      saveCorners(corners);
    },
    draw: (ctx, w, h) => {
      if (!isActive()) {
        return;
      }
      ctx.strokeStyle = "rgba(56,189,248,0.9)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (const [i, corner] of corners.entries()) {
        const px = corner.x * w;
        const py = corner.y * h;
        if (i === 0) {
          ctx.moveTo(px, py);
        } else {
          ctx.lineTo(px, py);
        }
      }
      ctx.closePath();
      ctx.stroke();
      for (const [i, corner] of corners.entries()) {
        const px = corner.x * w;
        const py = corner.y * h;
        ctx.beginPath();
        ctx.arc(px, py, HANDLE_RADIUS_PX, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(229,229,229,0.95)";
        ctx.fill();
        ctx.fillStyle = "#050505";
        ctx.font = "12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(String(i + 1), px, py);
      }
      if (saved === null) {
        ctx.fillStyle = "#e5e5e5";
        ctx.font = "14px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillText(HINT_TEXT, w / 2, 12);
      }
    },
  };
}
