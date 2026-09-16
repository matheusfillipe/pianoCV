import { applyHomography, findHomography, type Point } from "./homography";
import { isBlack, keyUnits } from "./keys";

export interface KeyPolygon {
  pitch: number;
  black: boolean;
  points: Point[];
}

const LOW_PITCH = 21;
const HIGH_PITCH = 108;
const BLACK_DEPTH = 0.67;
const LOW_UNITS = keyUnits(LOW_PITCH).from;
const HIGH_UNITS = keyUnits(HIGH_PITCH).to;
const SPAN = HIGH_UNITS - LOW_UNITS;
const UNIT_SQUARE: Point[] = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

export function keyPolygons(quad: readonly Point[]): KeyPolygon[] {
  const homography = findHomography(UNIT_SQUARE, quad);
  const polygons: KeyPolygon[] = [];
  for (let pitch = LOW_PITCH; pitch <= HIGH_PITCH; pitch += 1) {
    const units = keyUnits(pitch);
    const u0 = (units.from - LOW_UNITS) / SPAN;
    const u1 = (units.to - LOW_UNITS) / SPAN;
    const v1 = isBlack(pitch) ? BLACK_DEPTH : 1;
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
  width: number,
  height: number,
): void {
  const scaled = quad.map((point) => ({ x: point.x * width, y: point.y * height }));
  for (const key of keyPolygons(scaled)) {
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
    ctx.fillStyle = key.black ? "rgba(236,72,153,0.48)" : "rgba(56,189,248,0.22)";
    ctx.fill();
    ctx.strokeStyle = "rgba(255,255,255,0.46)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }
}
