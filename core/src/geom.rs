use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScoredPoint {
    pub x: f64,
    pub y: f64,
    pub score: f64,
}

impl ScoredPoint {
    pub fn point(&self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

/// Row-major 3x3, also used for crop pixel to frame fraction maps.
pub type Homography = [f64; 9];

pub fn distance(a: Point, b: Point) -> f64 {
    (b.x - a.x).hypot(b.y - a.y)
}

/// JS `Math.round`: halves go up.
pub fn js_round(x: f64) -> f64 {
    let floor = x.floor();
    if x - floor >= 0.5 {
        floor + 1.0
    } else {
        floor
    }
}

pub fn solve(matrix: &[Vec<f64>], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut rows: Vec<Vec<f64>> = matrix
        .iter()
        .zip(rhs)
        .map(|(row, b)| {
            let mut augmented = row.clone();
            augmented.push(*b);
            augmented
        })
        .collect();
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if rows[row][col].abs() > rows[pivot][col].abs() {
                pivot = row;
            }
        }
        rows.swap(pivot, col);
        for row in col + 1..n {
            let factor = rows[row][col] / rows[col][col];
            for k in col..=n {
                let above = rows[col][k];
                rows[row][k] -= factor * above;
            }
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = rows[row][n];
        for k in row + 1..n {
            sum -= rows[row][k] * solution[k];
        }
        solution[row] = sum / rows[row][row];
    }
    solution
}

pub fn find_homography(src: &[Point; 4], dst: &[Point; 4]) -> Homography {
    let mut matrix = vec![vec![0.0; 8]; 8];
    let mut rhs = vec![0.0; 8];
    for (i, (s, d)) in src.iter().zip(dst).enumerate() {
        matrix[i * 2][0] = s.x;
        matrix[i * 2][1] = s.y;
        matrix[i * 2][2] = 1.0;
        matrix[i * 2][6] = -d.x * s.x;
        matrix[i * 2][7] = -d.x * s.y;
        matrix[i * 2 + 1][3] = s.x;
        matrix[i * 2 + 1][4] = s.y;
        matrix[i * 2 + 1][5] = 1.0;
        matrix[i * 2 + 1][6] = -d.y * s.x;
        matrix[i * 2 + 1][7] = -d.y * s.y;
        rhs[i * 2] = d.x;
        rhs[i * 2 + 1] = d.y;
    }
    let s = solve(&matrix, &rhs);
    [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1.0]
}

pub fn apply_homography(h: &Homography, x: f64, y: f64) -> Point {
    let denom = h[6] * x + h[7] * y + h[8];
    Point {
        x: (h[0] * x + h[1] * y + h[2]) / denom,
        y: (h[3] * x + h[4] * y + h[5]) / denom,
    }
}

pub fn is_finite_homography(h: &Homography) -> bool {
    h.iter().all(|v| v.is_finite())
}

pub fn invert_homography(h: &Homography) -> Option<Homography> {
    let [a, b, c, d, e, f, g, k, i] = *h;
    let c00 = e * i - f * k;
    let c01 = -(d * i - f * g);
    let c02 = d * k - e * g;
    let c10 = -(b * i - c * k);
    let c11 = a * i - c * g;
    let c12 = -(a * k - b * g);
    let c20 = b * f - c * e;
    let c21 = -(a * f - c * d);
    let c22 = a * e - b * d;
    let det = a * c00 + b * c01 + c * c02;
    if det.abs() <= 1e-12 || det.is_nan() {
        return None;
    }
    let inverse = [
        c00 / det,
        c10 / det,
        c20 / det,
        c01 / det,
        c11 / det,
        c21 / det,
        c02 / det,
        c12 / det,
        c22 / det,
    ];
    is_finite_homography(&inverse).then_some(inverse)
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Correspondence {
    pub src: Point,
    pub dst: Point,
}

/// Least-squares homography over every correspondence.
pub fn refine_homography(pairs: &[Correspondence]) -> Option<Homography> {
    let mut ata = vec![vec![0.0; 8]; 8];
    let mut atb = vec![0.0; 8];
    for Correspondence { src, dst } in pairs {
        let row_u = [
            src.x,
            src.y,
            1.0,
            0.0,
            0.0,
            0.0,
            -dst.x * src.x,
            -dst.x * src.y,
        ];
        let row_v = [
            0.0,
            0.0,
            0.0,
            src.x,
            src.y,
            1.0,
            -dst.y * src.x,
            -dst.y * src.y,
        ];
        for (row, b) in [(row_u, dst.x), (row_v, dst.y)] {
            for i in 0..8 {
                atb[i] += row[i] * b;
                for j in 0..8 {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let s = solve(&ata, &atb);
    s.iter()
        .all(|v| v.is_finite())
        .then(|| [s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1.0])
}

const MIN_ASPECT: f64 = 1.8;
const MAX_ASPECT: f64 = 30.0;
const MIN_SHORT_EDGE: f64 = 0.01;
const MAX_END_RATIO: f64 = 3.2;

fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn sign(v: f64) -> f64 {
    if v == 0.0 || v.is_nan() {
        v
    } else {
        v.signum()
    }
}

/// Whether a keybed quad, back-low, back-high, front-high, front-low in image coordinates, shows
/// its board as a mirror image, which no turn of a camera makes.
pub fn is_mirrored_quad(quad: &[Point; 4]) -> bool {
    cross(quad[0], quad[1], quad[2]) < 0.0
}

/// Whether the quad is shaped like a keybed seen by a camera.
pub fn check_quad(quad: &[Point; 4]) -> bool {
    let signs: Vec<f64> = (0..4)
        .map(|i| sign(cross(quad[i], quad[(i + 1) % 4], quad[(i + 2) % 4])))
        .collect();
    let same = signs
        .iter()
        .all(|s| *s == signs[0] || (s.is_nan() && signs[0].is_nan()));
    if signs.contains(&0.0) || !same {
        return false;
    }
    let edges: Vec<f64> = (0..4)
        .map(|i| distance(quad[i], quad[(i + 1) % 4]))
        .collect();
    if edges.iter().copied().fold(f64::INFINITY, f64::min) < MIN_SHORT_EDGE {
        return false;
    }
    let span = (edges[0] + edges[2]) / 2.0;
    let depth = (edges[1] + edges[3]) / 2.0;
    let aspect = span / depth.max(MIN_SHORT_EDGE);
    if !(MIN_ASPECT..=MAX_ASPECT).contains(&aspect) {
        return false;
    }
    let largest = edges[1].max(edges[3]);
    let smallest = edges[1].min(edges[3]);
    largest / smallest.max(MIN_SHORT_EDGE) <= MAX_END_RATIO
}

/// A small deterministic PRNG so RANSAC picks the same samples every run.
pub struct Mulberry32(u32);

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b_79f5);
        let a = self.0;
        let mut t = (a ^ (a >> 15)).wrapping_mul(1 | a);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        f64::from(t ^ (t >> 14)) / 4_294_967_296.0
    }
}
