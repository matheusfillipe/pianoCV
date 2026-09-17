import type { Point } from "./homography";
import { isBlack, keyUnits } from "./keys";
import { keybedDepth, projectSpace, solvePose } from "./pose";

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

const BLACK_KEY_HEIGHT = 0.43;

export function keyPolygons(
  quad: readonly Point[],
  board: Board,
  imageWidth = Math.max(...quad.map((point) => point.x)),
  imageHeight = Math.max(...quad.map((point) => point.y)),
): KeyPolygon[] {
  const pose = solvePose([...quad], imageWidth, imageHeight, board.span);
  const polygons: KeyPolygon[] = [];
  for (let pitch = board.lowest; pitch <= board.highest; pitch += 1) {
    const units = keyUnits(pitch);
    const u0 = Math.max(0, units.from - board.origin);
    const u1 = Math.min(board.span, units.to - board.origin);
    if (u1 <= u0) continue;
    const black = isBlack(pitch);
    const v1 = (black ? board.blackDepth : 1) * keybedDepth();
    const elevation = black ? BLACK_KEY_HEIGHT : 0;
    polygons.push({
      pitch,
      black,
      points: [
        projectSpace(pose, u0, 0, elevation, imageWidth, imageHeight),
        projectSpace(pose, u1, 0, elevation, imageWidth, imageHeight),
        projectSpace(pose, u1, v1, elevation, imageWidth, imageHeight),
        projectSpace(pose, u0, v1, elevation, imageWidth, imageHeight),
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
  for (const key of keyPolygons(scaled, board, width, height)) {
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
