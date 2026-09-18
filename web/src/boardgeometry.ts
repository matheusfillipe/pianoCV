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

interface EdgeSample {
  u: number;
  point: Point;
}

interface Line {
  point: Point;
  dx: number;
  dy: number;
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

function solve3(matrix: number[][], rhs: number[]): number[] | null {
  const rows = matrix.map((row, index) => [...row, rhs[index] ?? 0]);
  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (
        Math.abs(rows[row][column] ?? 0) > Math.abs(rows[pivot][column] ?? 0)
      ) {
        pivot = row;
      }
    }
    if (Math.abs(rows[pivot][column] ?? 0) < 1e-9) return null;
    [rows[column], rows[pivot]] = [rows[pivot] ?? [], rows[column] ?? []];
    for (let row = column + 1; row < 3; row += 1) {
      const factor = (rows[row][column] ?? 0) / (rows[column][column] ?? 1);
      for (let at = column; at <= 3; at += 1) {
        rows[row][at] = (rows[row][at] ?? 0) - factor * (rows[column][at] ?? 0);
      }
    }
  }
  const solution = [0, 0, 0];
  for (let row = 2; row >= 0; row -= 1) {
    let value = rows[row][3] ?? 0;
    for (let column = row + 1; column < 3; column += 1) {
      value -= (rows[row][column] ?? 0) * (solution[column] ?? 0);
    }
    solution[row] = value / (rows[row][row] ?? 1);
  }
  return solution;
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

function safeSegmentPoint(
  point: Point,
  start: Point,
  end: Point,
): Point | null {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared < 1e-6) return null;
  const length = Math.sqrt(lengthSquared);
  const raw =
    ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared;
  if (raw < -0.08 || raw > 1.08) return null;
  const t = Math.min(1, Math.max(0, raw));
  const perpendicular =
    Math.abs((point.x - start.x) * dy - (point.y - start.y) * dx) / length;
  if (perpendicular > Math.max(8, 0.15 * length)) return null;
  return { x: start.x + t * dx, y: start.y + t * dy };
}

function edge(
  samples: readonly EdgeSample[],
  left: readonly [Point, Point],
  right: readonly [Point, Point],
): [Point, Point] | null {
  if (samples.length < 8) return null;

  // A panel graphic can produce a convincing local gradient. Require edge
  // evidence over most of the canonical keyboard before moving an endpoint.
  const us = samples.map((sample) => sample.u);
  if (Math.min(...us) > 0.15 || Math.max(...us) < 0.85) return null;

  let fitted = fitLine(samples.map((sample) => sample.point));
  if (!fitted) return null;
  const initial = fitted;
  const residuals = samples.map((sample) =>
    distanceToLine(sample.point, initial),
  );
  const typical = median(residuals);
  const inliers = samples.filter(
    (sample) =>
      distanceToLine(sample.point, initial) <= Math.max(2, 3 * typical),
  );
  if (inliers.length < 8) return null;
  fitted = fitLine(inliers.map((sample) => sample.point));
  if (!fitted) return null;
  const finalResidual = median(
    inliers.map((sample) => distanceToLine(sample.point, fitted)),
  );
  if (
    finalResidual > 4 ||
    Math.max(...inliers.map((sample) => distanceToLine(sample.point, fitted))) >
      10
  ) {
    return null;
  }

  const scale = Math.max(
    1,
    ...inliers.map((sample) =>
      Math.abs(
        (sample.point.x - fitted.point.x) * fitted.dx +
          (sample.point.y - fitted.point.y) * fitted.dy,
      ),
    ),
  );
  const normal = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const rhs = [0, 0, 0];
  for (const sample of inliers) {
    const along =
      ((sample.point.x - fitted.point.x) * fitted.dx +
        (sample.point.y - fitted.point.y) * fitted.dy) /
      scale;
    const row = [sample.u, 1, -along * sample.u];
    for (let rowIndex = 0; rowIndex < 3; rowIndex += 1) {
      rhs[rowIndex] = (rhs[rowIndex] ?? 0) + row[rowIndex] * along;
      for (let column = 0; column < 3; column += 1) {
        normal[rowIndex][column] =
          (normal[rowIndex][column] ?? 0) + row[rowIndex] * row[column];
      }
    }
  }
  const coefficients = solve3(normal, rhs);
  if (!coefficients) return null;
  const at = (u: number): number | null => {
    const denominator = 1 + (coefficients[2] ?? 0) * u;
    if (Math.abs(denominator) < 0.1) return null;
    return (
      (((coefficients[0] ?? 0) * u + (coefficients[1] ?? 0)) / denominator) *
      scale
    );
  };
  const startAlong = at(0);
  const endAlong = at(1);
  if (startAlong === null || endAlong === null) return null;
  const start = {
    x: fitted.point.x + fitted.dx * startAlong,
    y: fitted.point.y + fitted.dy * startAlong,
  };
  const end = {
    x: fitted.point.x + fitted.dx * endAlong,
    y: fitted.point.y + fitted.dy * endAlong,
  };
  const projectiveResidual = median(
    inliers.map((sample) => {
      const predicted = at(sample.u);
      const actual =
        (sample.point.x - fitted.point.x) * fitted.dx +
        (sample.point.y - fitted.point.y) * fitted.dy;
      return predicted === null ? Infinity : Math.abs(predicted - actual);
    }),
  );
  if (projectiveResidual > 4) return null;
  const safeStart = safeSegmentPoint(start, left[0], left[1]);
  const safeEnd = safeSegmentPoint(end, right[0], right[1]);
  return safeStart && safeEnd ? [safeStart, safeEnd] : null;
}

export function refineBoardEdges(
  image: ImageData,
  quad: readonly Point[],
  board: Board,
): Point[] {
  const h = findHomography(unit, quad);
  const rear: EdgeSample[] = [],
    front: EdgeSample[] = [];
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
      // The model proposal is the outer safety envelope. Visual refinement may
      // only pull an edge onto the keybed, never grow it into the case or table.
      const firstStep = boundary === 0 ? 0 : -40;
      const lastStep = boundary === 0 ? 40 : 0;
      for (let step = firstStep; step <= lastStep; step += 1) {
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
        output.push({ u, point: applyHomography(h, u, found) });
      }
    }
  }
  const back = edge(rear, [quad[0], quad[3]], [quad[1], quad[2]]);
  const near = edge(front, [quad[3], quad[0]], [quad[2], quad[1]]);
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
