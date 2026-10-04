import {
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import type { Point } from "./homography";

const HANDEDNESS_COLORS: Record<string, string> = {
  Left: "#38bdf8",
  Right: "#f472b6",
};
const FALLBACK_COLORS = ["#38bdf8", "#f472b6"];

export function drawHands(
  ctx: CanvasRenderingContext2D,
  hands: HandLandmarkerResult,
  w: number,
  h: number,
): void {
  for (const [i, landmarks] of hands.landmarks.entries()) {
    const color =
      HANDEDNESS_COLORS[hands.handedness[i]?.[0]?.categoryName ?? ""] ??
      FALLBACK_COLORS[i % FALLBACK_COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (const connection of HandLandmarker.HAND_CONNECTIONS) {
      const a = landmarks[connection.start];
      const b = landmarks[connection.end];
      if (!a || !b) {
        continue;
      }
      ctx.moveTo(a.x * w, a.y * h);
      ctx.lineTo(b.x * w, b.y * h);
    }
    ctx.stroke();
    ctx.fillStyle = color;
    for (const landmark of landmarks) {
      ctx.beginPath();
      ctx.arc(landmark.x * w, landmark.y * h, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

const BACK_EDGE_COLOR = "#fbbf24";
const ARROW_LENGTH = 0.55;

export function drawQuad(
  ctx: CanvasRenderingContext2D,
  quad: Point[],
  w: number,
  h: number,
  color: string,
  label?: string,
): void {
  const p = quad.map((corner) => ({ x: corner.x * w, y: corner.y * h }));
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (const [i, corner] of p.entries()) {
    if (i === 0) {
      ctx.moveTo(corner.x, corner.y);
    } else {
      ctx.lineTo(corner.x, corner.y);
    }
  }
  ctx.closePath();
  ctx.stroke();

  ctx.strokeStyle = BACK_EDGE_COLOR;
  ctx.lineWidth = 4;
  ctx.beginPath();
  ctx.moveTo(p[0].x, p[0].y);
  ctx.lineTo(p[1].x, p[1].y);
  ctx.stroke();

  const back = { x: (p[0].x + p[1].x) / 2, y: (p[0].y + p[1].y) / 2 };
  const front = { x: (p[2].x + p[3].x) / 2, y: (p[2].y + p[3].y) / 2 };
  const tip = {
    x: back.x + (front.x - back.x) * ARROW_LENGTH,
    y: back.y + (front.y - back.y) * ARROW_LENGTH,
  };
  const angle = Math.atan2(tip.y - back.y, tip.x - back.x);
  ctx.strokeStyle = BACK_EDGE_COLOR;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(back.x, back.y);
  ctx.lineTo(tip.x, tip.y);
  for (const spread of [2.6, -2.6]) {
    ctx.moveTo(tip.x, tip.y);
    ctx.lineTo(
      tip.x + 12 * Math.cos(angle + spread),
      tip.y + 12 * Math.sin(angle + spread),
    );
  }
  ctx.stroke();

  if (label) {
    ctx.fillStyle = color;
    ctx.font = "12px ui-sans-serif, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(label, back.x, back.y - 6);
  }
}

const WHITE_KEY_COLOR = "rgba(226,232,240,0.85)";
const BLACK_KEY_COLOR = "rgba(56,189,248,0.9)";

type Key = { readonly black: boolean; readonly bar: readonly Point[] };

// the clip hides the half of it on the black key and the black key's own 2 px outline covers
// another pixel, which leaves a 1 px white edge around each black key, as wide as a white outline
const NOTCH_WIDTH = 4;

export function tracePolygon(
  ctx: CanvasRenderingContext2D,
  bar: readonly Point[],
  w: number,
  h: number,
): void {
  for (const [i, corner] of bar.entries()) {
    if (i === 0) {
      ctx.moveTo(corner.x * w, corner.y * h);
    } else {
      ctx.lineTo(corner.x * w, corner.y * h);
    }
  }
  ctx.closePath();
}

/** Runs `draw` with every black key cut out of the canvas, so what it draws of the white keys
 * stops at the black keys' edges the way the keys themselves do. */
export function outsideBlackKeys(
  ctx: CanvasRenderingContext2D,
  keys: readonly Key[],
  w: number,
  h: number,
  draw: () => void,
): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(-w, -h, 3 * w, 3 * h);
  for (const key of keys) {
    if (key.black) {
      tracePolygon(ctx, key.bar, w, h);
    }
  }
  ctx.clip("evenodd");
  draw();
  ctx.restore();
}

export function drawKeys(
  ctx: CanvasRenderingContext2D,
  keys: readonly Key[],
  w: number,
  h: number,
): void {
  outsideBlackKeys(ctx, keys, w, h, () => {
    ctx.strokeStyle = WHITE_KEY_COLOR;
    ctx.lineWidth = 1;
    for (const key of keys) {
      if (!key.black) {
        ctx.beginPath();
        tracePolygon(ctx, key.bar, w, h);
        ctx.stroke();
      }
    }
    // the white keys' notches: the clip keeps only the outer half of a black key's outline
    ctx.lineWidth = NOTCH_WIDTH;
    for (const key of keys) {
      if (key.black) {
        ctx.beginPath();
        tracePolygon(ctx, key.bar, w, h);
        ctx.stroke();
      }
    }
  });
  ctx.strokeStyle = BLACK_KEY_COLOR;
  ctx.lineWidth = 2;
  for (const key of keys) {
    if (key.black) {
      ctx.beginPath();
      tracePolygon(ctx, key.bar, w, h);
      ctx.stroke();
    }
  }
}
