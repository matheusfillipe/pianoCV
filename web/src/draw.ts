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

let scratchCanvas: HTMLCanvasElement | null = null;

// allocating a canvas per frame is what made the overlay stutter, so keep one around
function modelInputScratch(size: number): CanvasRenderingContext2D | null {
  if (!scratchCanvas) {
    scratchCanvas = document.createElement("canvas");
  }
  if (scratchCanvas.width !== size) {
    scratchCanvas.width = size;
    scratchCanvas.height = size;
  }
  return scratchCanvas.getContext("2d");
}

export function drawModelInput(
  ctx: CanvasRenderingContext2D,
  gray: Float32Array,
  size: number,
  quad: Point[],
  box: number,
): void {
  const image = ctx.createImageData(size, size);
  for (let i = 0; i < gray.length; i += 1) {
    const v = Math.max(0, Math.min(255, Math.round(gray[i] * 255)));
    image.data[i * 4] = v;
    image.data[i * 4 + 1] = v;
    image.data[i * 4 + 2] = v;
    image.data[i * 4 + 3] = 255;
  }
  const scratchCtx = modelInputScratch(size);
  if (!scratchCtx) {
    return;
  }
  scratchCtx.putImageData(image, 0, 0);
  const scratch = scratchCtx.canvas;

  const x = 12;
  const y = ctx.canvas.height - box - 12;
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, x, y, box, box);
  ctx.strokeStyle = "rgba(229,229,229,0.35)";
  ctx.lineWidth = 1;
  ctx.strokeRect(x + 0.5, y + 0.5, box, box);
  ctx.translate(x, y);
  drawQuad(ctx, quad, box, box, "#4ade80");
  ctx.restore();
  ctx.fillStyle = "#8a8a8a";
  ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(`model input ${size}x${size}`, x, y - 14);
}
