import { applyHomography, findHomography, type Point } from "./homography";
import { isBlack, keyUnits } from "./keys";

export interface KeyPolygon {
  pitch: number;
  black: boolean;
  points: Point[];
}

export interface Board {
  lowest: number;
  highest: number;
  origin: number;
  span: number;
  blackDepth: number;
}

const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

export function keyPolygons(
  quad: readonly Point[],
  board: Board,
): KeyPolygon[] {
  const homography = findHomography(UNIT_SQUARE, quad);
  const polygons: KeyPolygon[] = [];
  for (let pitch = board.lowest; pitch <= board.highest; pitch += 1) {
    const units = keyUnits(pitch);
    const u0 = Math.max(0, (units.from - board.origin) / board.span);
    const u1 = Math.min(1, (units.to - board.origin) / board.span);
    if (u1 <= u0) continue;
    const v1 = isBlack(pitch) ? board.blackDepth : 1;
    polygons.push({
      pitch,
      black: isBlack(pitch),
      points: [
        applyHomography(homography, u0, 0),
        applyHomography(homography, u1, 0),
        applyHomography(homography, u1, v1),
        applyHomography(homography, u0, v1),
      ],
    });
  }
  return polygons;
}

export function drawKeyMasks(
  ctx: CanvasRenderingContext2D,
  quad: readonly Point[],
  board: Board,
  width: number,
  height: number,
): void {
  const scaled = quad.map((point) => ({
    x: point.x * width,
    y: point.y * height,
  }));
  for (const key of keyPolygons(scaled, board)) {
    const [first, ...rest] = key.points;
    if (!first) {
      continue;
    }
    ctx.beginPath();
    ctx.moveTo(first.x, first.y);
    for (const point of rest) {
      ctx.lineTo(point.x, point.y);
    }
    ctx.closePath();
    ctx.fillStyle = key.black
      ? "rgba(236,72,153,0.48)"
      : "rgba(56,189,248,0.22)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.46)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}
