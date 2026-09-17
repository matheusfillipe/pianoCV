import { fitBoard } from "./boardfit";
import { applyHomography, findHomography, type Point } from "./homography";
import type { Board } from "./keypolygons";
import { isBlack, keyUnits } from "./keys";

const unit = [
  { x: 0, y: 0 },
  { x: 1, y: 0 },
  { x: 1, y: 1 },
  { x: 0, y: 1 },
];

function fullyInFrame(image: ImageData, quad: readonly Point[]): boolean {
  return (
    quad.length === 4 &&
    quad.every(
      (point) =>
        Number.isFinite(point.x) &&
        Number.isFinite(point.y) &&
        point.x >= 0 &&
        point.y >= 0 &&
        point.x <= image.width - 1 &&
        point.y <= image.height - 1,
    )
  );
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

function level(image: ImageData, p: Point): number | null {
  const x = Math.round(p.x),
    y = Math.round(p.y);
  if (x < 0 || y < 0 || x >= image.width || y >= image.height) return null;
  const at = 4 * (y * image.width + x);
  return (
    0.299 * image.data[at] +
    0.587 * image.data[at + 1] +
    0.114 * image.data[at + 2]
  );
}

interface Line {
  point: Point;
  dx: number;
  dy: number;
}

function lineThrough(a: Point, b: Point): Line | null {
  const length = Math.hypot(b.x - a.x, b.y - a.y);
  return length < 1e-6
    ? null
    : { point: a, dx: (b.x - a.x) / length, dy: (b.y - a.y) / length };
}

function fitLine(points: readonly Point[]): Line | null {
  if (points.length < 8) return null;
  let meanX = 0;
  let meanY = 0;
  for (const point of points) {
    meanX += point.x;
    meanY += point.y;
  }
  meanX /= points.length;
  meanY /= points.length;
  let xx = 0;
  let xy = 0;
  let yy = 0;
  for (const point of points) {
    const x = point.x - meanX;
    const y = point.y - meanY;
    xx += x * x;
    xy += x * y;
    yy += y * y;
  }
  const angle = 0.5 * Math.atan2(2 * xy, xx - yy);
  return {
    point: { x: meanX, y: meanY },
    dx: Math.cos(angle),
    dy: Math.sin(angle),
  };
}

function distanceToLine(point: Point, line: Line): number {
  return Math.abs(
    (point.x - line.point.x) * line.dy - (point.y - line.point.y) * line.dx,
  );
}

function intersect(a: Line, b: Line): Point | null {
  const determinant = a.dx * b.dy - b.dx * a.dy;
  if (Math.abs(determinant) < 1e-6) return null;
  const rx = b.point.x - a.point.x;
  const ry = b.point.y - a.point.y;
  const t = (rx * b.dy - ry * b.dx) / determinant;
  return { x: a.point.x + t * a.dx, y: a.point.y + t * a.dy };
}

function cross(a: Point, b: Point, c: Point): number {
  return (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
}

function validQuad(
  candidate: readonly Point[],
  proposal: readonly Point[],
): boolean {
  if (candidate.length !== 4 || proposal.length !== 4) return false;
  if (
    candidate.some(
      (point) => !Number.isFinite(point.x) || !Number.isFinite(point.y),
    )
  ) {
    return false;
  }
  const turns = candidate.map((point, index) =>
    cross(
      point,
      candidate[(index + 1) % 4] ?? point,
      candidate[(index + 2) % 4] ?? point,
    ),
  );
  if (turns.some((turn) => Math.abs(turn) < 1e-3)) return false;
  if (turns.some((turn) => Math.sign(turn) !== Math.sign(turns[0] ?? turn))) {
    return false;
  }
  return candidate.every((point, index) => {
    const end = index === 0 || index === 3 ? 0 : 1;
    const depth = Math.hypot(
      (proposal[end === 0 ? 3 : 2]?.x ?? 0) - (proposal[end]?.x ?? 0),
      (proposal[end === 0 ? 3 : 2]?.y ?? 0) - (proposal[end]?.y ?? 0),
    );
    return (
      Math.hypot(
        point.x - (proposal[index]?.x ?? point.x),
        point.y - (proposal[index]?.y ?? point.y),
      ) <=
      0.6 * depth + 8
    );
  });
}

function edge(
  points: readonly Point[],
  left: readonly [Point, Point],
  right: readonly [Point, Point],
  longStart: Point,
  longEnd: Point,
): [Point, Point] | null {
  const long = lineThrough(longStart, longEnd);
  const leftSide = lineThrough(left[0], left[1]);
  const rightSide = lineThrough(right[0], right[1]);
  if (!long || !leftSide || !rightSide || points.length < 8) return null;

  // A panel graphic can produce a convincing local gradient. Require edge
  // evidence over most of the keybed before allowing it to move a corner.
  const along = points.map(
    (point) =>
      (point.x - longStart.x) * long.dx + (point.y - longStart.y) * long.dy,
  );
  const extent = Math.max(...along) - Math.min(...along);
  if (
    extent <
    0.65 * Math.hypot(longEnd.x - longStart.x, longEnd.y - longStart.y)
  ) {
    return null;
  }

  let fitted = fitLine(points);
  if (!fitted) return null;
  const initial = fitted;
  const residuals = points.map((point) => distanceToLine(point, initial));
  const typical = median(residuals);
  const inliers = points.filter(
    (point) => distanceToLine(point, initial) <= Math.max(2, 3 * typical),
  );
  if (inliers.length < 8) return null;
  const keptAlong = inliers.map(
    (point) =>
      (point.x - longStart.x) * long.dx + (point.y - longStart.y) * long.dy,
  );
  if (
    Math.max(...keptAlong) - Math.min(...keptAlong) <
    0.65 * Math.hypot(longEnd.x - longStart.x, longEnd.y - longStart.y)
  ) {
    return null;
  }
  if (inliers.length >= 8 && inliers.length < points.length) {
    fitted = fitLine(inliers);
  }
  if (!fitted) return null;
  const finalResidual = median(
    points.map((point) => distanceToLine(point, fitted)),
  );
  if (finalResidual > 4) return null;
  const start = intersect(fitted, leftSide);
  const end = intersect(fitted, rightSide);
  return start && end ? [start, end] : null;
}

export function refineBoardEdges(
  image: ImageData,
  quad: readonly Point[],
  board: Board,
): Point[] {
  const h = findHomography(unit, quad);
  const rear: Point[] = [],
    front: Point[] = [];
  for (let pitch = board.lowest; pitch <= board.highest; pitch += 1) {
    if (isBlack(pitch)) continue;
    const key = keyUnits(pitch);
    const u = ((key.from + key.to) / 2 - board.origin) / board.span;
    if (u < 0.02 || u > 0.98) continue;
    for (const [boundary, output, direction] of [
      [0, rear, 1],
      [1, front, -1],
    ] as const) {
      let best = 20,
        found: number | null = null;
      // Mask quads are often cut a few tenths of the key depth into the case.
      // Search a wider adaptive band, while the distributed line fit below
      // rejects isolated case/panel edges.
      for (let step = -40; step <= 40; step += 1) {
        const v = boundary + step * 0.01;
        const inside = level(
          image,
          applyHomography(h, u, v + direction * 0.025),
        );
        const outside = level(
          image,
          applyHomography(h, u, v - direction * 0.025),
        );
        if (inside === null || outside === null) continue;
        const contrast = inside - outside;
        if (contrast > best) {
          best = contrast;
          found = v;
        }
      }
      if (found !== null) {
        output.push(applyHomography(h, u, found));
      }
    }
  }
  const back = edge(
    rear,
    [quad[0], quad[3]],
    [quad[1], quad[2]],
    quad[0],
    quad[1],
  );
  const near = edge(
    front,
    [quad[3], quad[0]],
    [quad[2], quad[1]],
    quad[3],
    quad[2],
  );
  const candidate = [
    back?.[0] ?? quad[0],
    back?.[1] ?? quad[1],
    near?.[1] ?? quad[2],
    near?.[0] ?? quad[3],
  ];
  return validQuad(candidate, quad) ? candidate : [...quad];
}

export function calibrateBoard(
  image: ImageData,
  proposal: readonly Point[],
): { board: Board; quad: Point[] } | null {
  // The current board model maps one complete physical keyboard into one quad.
  // When an end is outside the frame, fitting only the visible stripe and then
  // spreading it over that quad invents key positions. A later subrange solver
  // can support cropped views; until then, decline rather than draw a false map.
  if (!fullyInFrame(image, proposal)) return null;
  const initial = fitBoard(image, proposal);
  if (initial === null) return null;
  const quad = refineBoardEdges(image, proposal, initial);
  const board = fitBoard(image, quad);
  return board === null ? null : { board, quad };
}
