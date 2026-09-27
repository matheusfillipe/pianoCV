export interface Point {
  x: number;
  y: number;
}

export type Homography = number[];

export function solve(matrix: number[][], rhs: number[]): number[] {
  const n = rhs.length;
  const rows = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < n; row += 1) {
      if (Math.abs(rows[row][col]) > Math.abs(rows[pivot][col])) {
        pivot = row;
      }
    }
    const swap = rows[pivot];
    rows[pivot] = rows[col];
    rows[col] = swap;
    for (let row = col + 1; row < n; row += 1) {
      const factor = rows[row][col] / rows[col][col];
      for (let k = col; k <= n; k += 1) {
        rows[row][k] -= factor * rows[col][k];
      }
    }
  }
  const solution = new Array<number>(n).fill(0);
  for (let row = n - 1; row >= 0; row -= 1) {
    let sum = rows[row][n];
    for (let k = row + 1; k < n; k += 1) {
      sum -= rows[row][k] * solution[k];
    }
    solution[row] = sum / rows[row][row];
  }
  return solution;
}

export function findHomography(
  src: readonly Point[],
  dst: readonly Point[],
): Homography {
  if (src.length !== 4 || dst.length !== 4) {
    throw new Error("src and dst must each have 4 points");
  }
  const matrix: number[][] = Array.from({ length: 8 }, () =>
    new Array<number>(8).fill(0),
  );
  const rhs = new Array<number>(8).fill(0);
  for (const [i, s] of src.entries()) {
    const d = dst[i];
    const rowU = matrix[i * 2];
    const rowV = matrix[i * 2 + 1];
    rowU[0] = s.x;
    rowU[1] = s.y;
    rowU[2] = 1;
    rowU[6] = -d.x * s.x;
    rowU[7] = -d.x * s.y;
    rowV[3] = s.x;
    rowV[4] = s.y;
    rowV[5] = 1;
    rowV[6] = -d.y * s.x;
    rowV[7] = -d.y * s.y;
    rhs[i * 2] = d.x;
    rhs[i * 2 + 1] = d.y;
  }
  return [...solve(matrix, rhs), 1];
}

export function applyHomography(h: Homography, x: number, y: number): Point {
  const denom = h[6] * x + h[7] * y + h[8];
  return {
    x: (h[0] * x + h[1] * y + h[2]) / denom,
    y: (h[3] * x + h[4] * y + h[5]) / denom,
  };
}
