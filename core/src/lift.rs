use serde::{Deserialize, Serialize};

use crate::decode::Peaks;
use crate::fit::{key_width_px, Fit, Lift, RANSAC_SEED};
use crate::geom::{
    apply_homography, distance, invert_homography, solve, Homography, Mulberry32, Point,
    ScoredPoint, Size,
};
use crate::keys::{board_keys, keyboard_template};

// how far along the board, in white keys, a top corner can sit from the bottom corner under it
// once mapped back through the keybed's homography
const LIFT_SEARCH_KEYS: f64 = 1.5;
const LIFT_TOLERANCE_KEYS: f64 = 0.2;
const LIFT_LEAST_INLIERS: usize = 6;
const LIFT_ITERATIONS: usize = 100;
const LIFT_AGREE_SHARE: f64 = 0.8;
const LIFT_MOST_KEYS: f64 = 1.0;
pub const LIFT_JUMP_KEYS: f64 = 0.25;
pub const LIFT_JUMPS_TO_MOVE: u32 = 10;

pub fn lifted(h: &Homography, lift: &Lift, p: Point) -> Point {
    let u = h[0] * p.x + h[1] * p.y + h[2] + lift[0];
    let v = h[3] * p.x + h[4] * p.y + h[5] + lift[1];
    let s = h[6] * p.x + h[7] * p.y + h[8] + lift[2];
    Point { x: u / s, y: v / s }
}

#[derive(Clone, Copy)]
struct LiftPair {
    top: Point,
    top_id: usize,
    bottom: Point,
}

/// The least-squares lift for these pairs: each gives u + w0 = x (s + w2) and v + w1 = y (s + w2),
/// linear in the lift w.
fn solve_lift(h: &Homography, pairs: &[LiftPair]) -> Option<Lift> {
    let mut normal = vec![vec![0.0; 3]; 3];
    let mut rhs = [0.0; 3];
    for LiftPair { top, bottom, .. } in pairs {
        let u = h[0] * bottom.x + h[1] * bottom.y + h[2];
        let v = h[3] * bottom.x + h[4] * bottom.y + h[5];
        let s = h[6] * bottom.x + h[7] * bottom.y + h[8];
        let rows = [
            ([1.0, 0.0, -top.x], top.x * s - u),
            ([0.0, 1.0, -top.y], top.y * s - v),
        ];
        for (row, value) in rows {
            for i in 0..3 {
                rhs[i] += row[i] * value;
                for j in 0..3 {
                    normal[i][j] += row[i] * row[j];
                }
            }
        }
    }
    let lift = solve(&normal, &rhs);
    lift.iter()
        .all(|v| v.is_finite())
        .then(|| [lift[0], lift[1], lift[2]])
}

/// The lift that puts the most detected top corners on their keys, by RANSAC over which bottom
/// corner each top corner belongs to, or None when too few top corners agree on one.
pub fn estimate_lift(peaks: &Peaks, fit: &Fit, frame: Size) -> Option<Lift> {
    let inverse = invert_homography(&fit.homography)?;
    let template = keyboard_template(fit.white_keys, fit.phase, fit.black_depth);
    let at_back = |points: &[Point]| -> Vec<Point> {
        points.iter().map(|p| Point { x: p.x, y: 0.0 }).collect()
    };
    let tops: Vec<(ScoredPoint, Vec<Point>)> = [
        (&peaks.black_top_low, template.black_low.clone()),
        (&peaks.black_top_high, template.black_high.clone()),
        (&peaks.black_back_low, at_back(&template.black_low)),
        (&peaks.black_back_high, at_back(&template.black_high)),
    ]
    .into_iter()
    .flat_map(|(tops, bottoms)| tops.iter().map(move |top| (*top, bottoms.clone())))
    .map(|(top, bottoms)| {
        let along = apply_homography(&inverse, top.x, top.y).x;
        (
            top,
            bottoms
                .iter()
                .copied()
                .filter(|b| (b.x - along).abs() < LIFT_SEARCH_KEYS)
                .collect(),
        )
    })
    .collect();
    let candidates: Vec<LiftPair> = tops
        .iter()
        .enumerate()
        .flat_map(|(top_id, (top, bottoms))| {
            bottoms.iter().map(move |bottom| LiftPair {
                top: top.point(),
                top_id,
                bottom: *bottom,
            })
        })
        .collect();
    if candidates.len() < 2 {
        return None;
    }
    let tolerance = LIFT_TOLERANCE_KEYS * key_width_px(&fit.quad, fit.white_keys, frame);
    let inliers_of = |lift: &Lift| -> Vec<LiftPair> {
        tops.iter()
            .enumerate()
            .filter_map(|(top_id, (top, bottoms))| {
                let mut best = None;
                let mut best_distance = tolerance;
                for bottom in bottoms {
                    let p = lifted(&fit.homography, lift, *bottom);
                    let distance =
                        ((p.x - top.x) * frame.width).hypot((p.y - top.y) * frame.height);
                    if distance < best_distance {
                        best = Some(LiftPair {
                            top: top.point(),
                            top_id,
                            bottom: *bottom,
                        });
                        best_distance = distance;
                    }
                }
                best
            })
            .collect()
    };
    let mut rng = Mulberry32::new(RANSAC_SEED);
    let mut hypotheses: Vec<(Vec<LiftPair>, f64)> = Vec::new();
    for _ in 0..LIFT_ITERATIONS {
        let a = candidates[(rng.next_f64() * candidates.len() as f64).floor() as usize];
        let b = candidates[(rng.next_f64() * candidates.len() as f64).floor() as usize];
        if a.top_id == b.top_id {
            continue;
        }
        let Some(lift) = solve_lift(&fit.homography, &[a, b]) else {
            continue;
        };
        hypotheses.push((inliers_of(&lift), lift_keys(fit, &lift, frame)));
    }
    let most = hypotheses.iter().map(|h| h.0.len()).fold(0, usize::max);
    // pairing every top with the next key's bottom can agree almost as well on a small frame, and
    // black keys stand about half a white key's width up, so of the answers that agree about as
    // well as the best we take the smallest, and none that moves the tops a whole key
    let chosen = hypotheses
        .iter()
        .filter(|(inliers, keys)| {
            *keys < LIFT_MOST_KEYS && inliers.len() as f64 >= most as f64 * LIFT_AGREE_SHARE
        })
        .fold(None, |best: Option<&(Vec<LiftPair>, f64)>, h| match best {
            Some(b) if h.1 >= b.1 => Some(b),
            _ => Some(h),
        })?;
    if chosen.0.len() < LIFT_LEAST_INLIERS {
        return None;
    }
    solve_lift(&fit.homography, &chosen.0)
}

/// How far the lift moves the middle black key's top from its bottom, in white-key widths.
pub fn lift_keys(fit: &Fit, lift: &Lift, frame: Size) -> f64 {
    let middle = Point {
        x: fit.white_keys as f64 / 2.0,
        y: fit.black_depth,
    };
    let bottom = apply_homography(&fit.homography, middle.x, middle.y);
    let top = lifted(&fit.homography, lift, middle);
    ((top.x - bottom.x) * frame.width).hypot((top.y - bottom.y) * frame.height)
        / key_width_px(&fit.quad, fit.white_keys, frame)
}

/// How far apart two lifts put the middle black key's top, in white-key widths.
pub fn lift_apart_keys(fit: &Fit, a: &Lift, b: &Lift, frame: Size) -> f64 {
    let middle = Point {
        x: fit.white_keys as f64 / 2.0,
        y: fit.black_depth,
    };
    let p = lifted(&fit.homography, a, middle);
    let q = lifted(&fit.homography, b, middle);
    ((p.x - q.x) * frame.width).hypot((p.y - q.y) * frame.height)
        / key_width_px(&fit.quad, fit.white_keys, frame)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Face {
    pub black: bool,
    pub semitone: i32,
    pub bar: [Point; 4],
}

/// The keys' visible faces in frame fractions, drawn through the fit and its lift: each white key
/// flat on the keybed, each black key as its raised top and its footprint on the keybed, whose
/// outline together is the key's silhouette with whichever side and front faces the camera sees.
pub fn key_net_faces(fit: &Fit, lift: &Lift) -> Vec<Face> {
    let plane = |x: f64, y: f64| apply_homography(&fit.homography, x, y);
    let top = |x: f64, y: f64| lifted(&fit.homography, lift, Point { x, y });
    board_keys(fit.white_keys, fit.phase, fit.black_depth)
        .into_iter()
        .flat_map(|key| {
            let (black, semitone, from, to, depth) =
                (key.black, key.semitone, key.from, key.to, key.depth);
            if !black {
                return vec![Face {
                    black,
                    semitone,
                    bar: [
                        plane(from, 0.0),
                        plane(to, 0.0),
                        plane(to, 1.0),
                        plane(from, 1.0),
                    ],
                }];
            }
            vec![
                Face {
                    black,
                    semitone,
                    bar: [
                        top(from, 0.0),
                        top(to, 0.0),
                        top(to, depth),
                        top(from, depth),
                    ],
                },
                Face {
                    black,
                    semitone,
                    bar: [
                        plane(from, 0.0),
                        plane(to, 0.0),
                        plane(to, depth),
                        plane(from, depth),
                    ],
                },
            ]
        })
        .collect()
}

fn signed_area(points: &[Point]) -> f64 {
    points
        .iter()
        .zip(points.iter().cycle().skip(1))
        .map(|(p, q)| p.x * q.y - q.x * p.y)
        .sum::<f64>()
        / 2.0
}

fn convex_hull(points: &[Point]) -> Vec<Point> {
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    let cross =
        |o: Point, a: Point, b: Point| (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
    let half = |list: &mut dyn Iterator<Item = Point>| -> Vec<Point> {
        let mut out: Vec<Point> = Vec::new();
        for p in list {
            while out.len() >= 2 && cross(out[out.len() - 2], out[out.len() - 1], p) <= 0.0 {
                out.pop();
            }
            out.push(p);
        }
        out.pop();
        out
    };
    let mut hull = half(&mut sorted.iter().copied());
    hull.extend(half(&mut sorted.iter().rev().copied()));
    hull
}

/// The outline of a black key's top and footprint together, starting at the top's first corner
/// and wound the same way, so each point of it is the same spot of the key every frame.
pub fn key_hull(top: &[Point; 4], footprint: &[Point; 4]) -> Vec<Point> {
    let mut hull = convex_hull(&[top.as_slice(), footprint.as_slice()].concat());
    if signed_area(&hull) * signed_area(top) < 0.0 {
        hull.reverse();
    }
    let start = hull
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| distance(**a, top[0]).total_cmp(&distance(**b, top[0])))
        .map_or(0, |(i, _)| i);
    hull.rotate_left(start);
    hull
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KeyOutline {
    pub black: bool,
    /// Semitones above the board's first white key.
    pub semitone: i32,
    /// Where the key starts and ends along the board, in white keys from its first key.
    pub from: f64,
    pub to: f64,
    /// The key's outline in frame fractions, a black key's raised top and footprint together.
    pub bar: Vec<Point>,
    /// The face the key shows on top: a white key's own face and a black key's raised top.
    pub top: [Point; 4],
}

/// Every key of the fitted board once, low to high, flat on the keybed until the fit has a lift,
/// where the camera sees it, with its lens's bend.
pub fn key_outlines(fit: &Fit) -> Vec<KeyOutline> {
    let layout = board_keys(fit.white_keys, fit.phase, fit.black_depth);
    let faces = key_net_faces(fit, &fit.lift.unwrap_or([0.0; 3]));
    let mut faces = faces.iter().peekable();
    let mut keys = Vec::new();
    while let Some(face) = faces.next() {
        let bar = match faces.next_if(|f| face.black && f.black && f.semitone == face.semitone) {
            Some(footprint) => key_hull(&face.bar, &footprint.bar),
            None => face.bar.to_vec(),
        };
        let Some(span) = layout.iter().find(|key| key.semitone == face.semitone) else {
            continue;
        };
        let seen = |p: &Point| fit.lens.bend(*p);
        keys.push(KeyOutline {
            black: face.black,
            semitone: face.semitone,
            from: span.from,
            to: span.to,
            bar: bar.iter().map(seen).collect(),
            top: face.bar.each_ref().map(seen),
        });
    }
    keys
}

#[cfg(test)]
mod hull_tests {
    use super::*;

    fn square(x: f64, y: f64, side: f64) -> [Point; 4] {
        [(0.0, 0.0), (side, 0.0), (side, side), (0.0, side)].map(|(dx, dy)| Point {
            x: x + dx,
            y: y + dy,
        })
    }

    #[test]
    fn the_hull_starts_at_the_tops_first_corner_and_winds_with_it() {
        let top = square(0.0, 0.0, 1.0);
        let hull = key_hull(&top, &square(0.5, 1.0, 1.0));
        assert_eq!(hull.len(), 6);
        assert_eq!(hull[0], top[0]);
        assert!(signed_area(&hull) * signed_area(&top) > 0.0);
    }
}
