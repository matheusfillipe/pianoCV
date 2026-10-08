use serde::{Deserialize, Serialize};

use crate::decode::Peaks;
use crate::geom::{
    apply_homography, check_quad, distance, find_homography, invert_homography,
    is_finite_homography, js_round, refine_homography, solve, Correspondence, Homography,
    Mulberry32, Point, ScoredPoint, Size,
};
use crate::keys::{keyboard_template, KeyboardTemplate, Phase, BLACK_KEY_DEPTH, STANDARD_BOARDS};
use crate::lens::Lens;

/// What takes a point on the keybed to the same point on the black keys' tops, in the fit
/// homography's own homogeneous frame.
pub type Lift = [f64; 3];

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lock {
    pub white_keys: usize,
    pub phase: Phase,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fit {
    /// Maps the template plane (white-key units) onto the frame (fractions).
    pub homography: Homography,
    pub quad: [Point; 4],
    pub white_keys: usize,
    pub phase: Phase,
    pub inlier_share: f64,
    /// The share of every detected point the fit explains, which is how boards are compared.
    pub explained: f64,
    pub both_ends: bool,
    pub beyond_ends: usize,
    pub gap_spacing: Option<f64>,
    pub reprojection_error: f64,
    pub lift: Option<Lift>,
    /// How much of the keybed's depth this board's black keys take, which the template was
    /// fitted with.
    #[serde(default = "standard_black_depth")]
    pub black_depth: f64,
    /// The bend the board was fitted under: the homography, quad and lift all live in the
    /// frame with it taken out, and `lens.bend` carries a point of them into the picture.
    #[serde(default)]
    pub lens: Lens,
}

fn standard_black_depth() -> f64 {
    BLACK_KEY_DEPTH
}

impl Fit {
    pub fn lock(&self) -> Lock {
        Lock {
            white_keys: self.white_keys,
            phase: self.phase,
        }
    }
}

type Estimator = Box<dyn Fn(Point) -> f64>;

const GAP_SNAP_KEYS: f64 = 0.3;
const BLACK_SNAP_KEYS: f64 = 0.3;

fn nearest_by_x(points: &[Point], x: f64) -> Option<Point> {
    points.iter().copied().fold(None, |best, p| match best {
        Some(b) if (p.x - x).abs() >= (b.x - x).abs() || (p.x - x).is_nan() => Some(b),
        None => Some(p),
        _ => Some(p),
    })
}

/// Frame-to-template least-squares affine, to place peaks before any homography exists.
fn fit_affine(pairs: &[Correspondence]) -> Option<impl Fn(Point) -> Point> {
    if pairs.len() < 3 {
        return None;
    }
    let mut ata = vec![vec![0.0; 3]; 3];
    let mut atb_x = vec![0.0; 3];
    let mut atb_y = vec![0.0; 3];
    for Correspondence { src, dst } in pairs {
        let row = [src.x, src.y, 1.0];
        for i in 0..3 {
            atb_x[i] += row[i] * dst.x;
            atb_y[i] += row[i] * dst.y;
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let sol_x = solve(&ata, &atb_x);
    let sol_y = solve(&ata, &atb_y);
    if !sol_x.iter().chain(&sol_y).all(|v| v.is_finite()) {
        return None;
    }
    Some(move |p: Point| Point {
        x: sol_x[0] * p.x + sol_x[1] * p.y + sol_x[2],
        y: sol_y[0] * p.x + sol_y[1] * p.y + sol_y[2],
    })
}

/// Where a frame point sits along the template's x axis, coarsely. With all 4 corners it inverts
/// the exact 4-point homography they give; with 3 it falls back to a least-squares affine.
fn template_x_estimator(
    template: &KeyboardTemplate,
    corners: &[Option<Point>; 4],
) -> Option<Estimator> {
    let present: Vec<(usize, Point)> = corners
        .iter()
        .enumerate()
        .filter_map(|(channel, p)| p.map(|p| (channel, p)))
        .collect();
    if present.len() == 4 {
        let dst = [present[0].1, present[1].1, present[2].1, present[3].1];
        let h = find_homography(&template.corners, &dst);
        let inverse = if is_finite_homography(&h) {
            invert_homography(&h)
        } else {
            None
        };
        if let Some(inverse) = inverse {
            return Some(Box::new(move |p| apply_homography(&inverse, p.x, p.y).x));
        }
    }
    let pairs: Vec<Correspondence> = present
        .iter()
        .map(|(channel, point)| Correspondence {
            src: *point,
            dst: template.corners[*channel],
        })
        .collect();
    let affine = fit_affine(&pairs)?;
    Some(Box::new(move |p| affine(p).x))
}

const COMPLETE_LEAST_POINTS: usize = 6;
const COMPLETE_NEIGHBOURS: usize = 4;

/// The corners with the back ones filled in when only the front ones were seen, as on a real
/// keyboard whose case hides the white keys' back ends.
pub fn complete_corners(peaks: &Peaks) -> [Option<Point>; 4] {
    let plain = peaks.corners.map(|c| c.map(|c| c.point()));
    let [back_low, back_high, front_high, front_low] = plain;
    let blacks: Vec<Point> = peaks
        .black_low
        .iter()
        .chain(&peaks.black_high)
        .map(ScoredPoint::point)
        .collect();
    let (Some(front_high), Some(front_low)) = (front_high, front_low) else {
        return plain;
    };
    if (back_low.is_some() && back_high.is_some())
        || peaks.gaps.len() < COMPLETE_LEAST_POINTS
        || blacks.len() < COMPLETE_LEAST_POINTS
    {
        return plain;
    }
    let up_keys_near = |end: Point| -> Point {
        let mut gaps: Vec<Point> = peaks.gaps.iter().map(ScoredPoint::point).collect();
        gaps.sort_by(|a, b| {
            distance(*a, end)
                .partial_cmp(&distance(*b, end))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let offsets: Vec<Point> = gaps
            .iter()
            .take(COMPLETE_NEIGHBOURS)
            .map(|gap| {
                let black = blacks
                    .iter()
                    .copied()
                    .reduce(|best, p| {
                        if distance(p, *gap) < distance(best, *gap) {
                            p
                        } else {
                            best
                        }
                    })
                    .unwrap_or(*gap);
                Point {
                    x: black.x - gap.x,
                    y: black.y - gap.y,
                }
            })
            .collect();
        let share = 1.0 - BLACK_KEY_DEPTH;
        let count = offsets.len() as f64;
        Point {
            x: offsets.iter().fold(0.0, |s, o| s + o.x) / count / share,
            y: offsets.iter().fold(0.0, |s, o| s + o.y) / count / share,
        }
    };
    let behind = |front: Point| -> Point {
        let up = up_keys_near(front);
        Point {
            x: front.x + up.x,
            y: front.y + up.y,
        }
    };
    [
        back_low.or_else(|| Some(behind(front_low))),
        back_high.or_else(|| Some(behind(front_high))),
        Some(front_high),
        Some(front_low),
    ]
}

#[derive(Clone, Debug, Serialize)]
pub struct CountedGap {
    pub gap: ScoredPoint,
    pub index: i32,
}

#[derive(Clone, Copy)]
struct Along {
    gap: ScoredPoint,
    t: f64,
}

#[derive(Clone, Copy)]
struct Stepped {
    t: f64,
    index: f64,
}

const COUNT_EDGE_SLACK: f64 = 0.03;
const COUNT_LEAST_GAPS: usize = 6;
const COUNT_FIRST_STEPS: usize = 5;
const COUNT_SHORTEST_STEP: f64 = 0.6;
const COUNT_PASSES: usize = 2;
const COUNT_SNAP_KEYS: f64 = 0.3;

fn by_value(a: &f64, b: &f64) -> std::cmp::Ordering {
    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
}

/// Each gap on the front edge with the key boundary it is, counted from the low front corner.
/// Null when the front corners or too few gaps were found.
pub fn count_gaps(
    peaks: &Peaks,
    white_keys: Option<usize>,
    guess: Option<&dyn Fn(Point) -> f64>,
) -> Option<Vec<CountedGap>> {
    let front_high = peaks.corners[2]?;
    let front_low = peaks.corners[3]?;
    let dx = front_high.x - front_low.x;
    let dy = front_high.y - front_low.y;
    let length = dx.hypot(dy);
    if length == 0.0 {
        return None;
    }
    let mut along: Vec<(Along, f64)> = peaks
        .gaps
        .iter()
        .map(|gap| {
            (
                Along {
                    gap: *gap,
                    t: ((gap.x - front_low.x) * dx + (gap.y - front_low.y) * dy) / length.powi(2),
                },
                ((gap.x - front_low.x) * dy - (gap.y - front_low.y) * dx).abs() / length,
            )
        })
        .filter(|(a, off)| a.t > 0.0 && a.t < 1.0 && *off < COUNT_EDGE_SLACK * length)
        .collect();
    along.sort_by(|a, b| by_value(&a.0.t, &b.0.t));
    let along: Vec<Along> = along.into_iter().map(|(a, _)| a).collect();
    if along.len() < COUNT_LEAST_GAPS {
        return None;
    }
    let mut first_steps: Vec<f64> = along
        .iter()
        .skip(1)
        .take(COUNT_FIRST_STEPS)
        .enumerate()
        .map(|(i, a)| a.t - along[i].t)
        .collect();
    first_steps.sort_by(by_value);
    let mut spacing = first_steps[first_steps.len() / 2];
    let mut index = js_round(along[0].t / spacing).max(1.0);
    let mut last = along[0];
    let mut stepped = vec![Stepped { t: last.t, index }];
    for next in &along[1..] {
        let step = next.t - last.t;
        if step < COUNT_SHORTEST_STEP * spacing {
            continue;
        }
        let keys = js_round(step / spacing).max(1.0);
        index += keys;
        spacing = (spacing + step / keys) / 2.0;
        last = *next;
        stepped.push(Stepped { t: next.t, index });
    }
    let mut starts = vec![stepped[..stepped.len().div_ceil(2)].to_vec()];
    if let Some(guess) = guess {
        starts.push(
            along
                .iter()
                .map(|a| (a.t, guess(a.gap.point())))
                .filter(|(_, index)| (index - js_round(*index)).abs() < COUNT_SNAP_KEYS)
                .map(|(t, index)| Stepped {
                    t,
                    index: js_round(index),
                })
                .collect(),
        );
    }
    let mut best: Option<(Vec<CountedGap>, f64)> = None;
    for start in &starts {
        if let Some((counted, error)) = number_along_edge(&along, start, white_keys) {
            let better = match &best {
                None => true,
                Some((b, b_error)) => {
                    counted.len() > b.len() || (counted.len() == b.len() && error < *b_error)
                }
            };
            if better {
                best = Some((counted, error));
            }
        }
    }
    best.map(|(counted, _)| counted)
}

/// Every gap numbered by the edge map fitted to `start`, refitted a few times, with how far the
/// numbered gaps sit from whole keys on average.
fn number_along_edge(
    along: &[Along],
    start: &[Stepped],
    white_keys: Option<usize>,
) -> Option<(Vec<CountedGap>, f64)> {
    let mut map = fit_edge_map(start, white_keys);
    let mut result = None;
    for _ in 0..COUNT_PASSES {
        let Some(fitted) = map else { break };
        let numbered: Vec<(Along, f64, f64)> = along
            .iter()
            .map(|a| {
                let exact = fitted.index_at(a.t);
                (*a, exact, js_round(exact))
            })
            .filter(|(_, exact, index)| {
                *index >= 1.0
                    && white_keys.is_none_or(|w| *index < w as f64)
                    && (exact - index).abs() < COUNT_SNAP_KEYS
            })
            .collect();
        let error = numbered
            .iter()
            .fold(0.0, |sum, (_, exact, index)| sum + (exact - index).abs())
            / (numbered.len().max(1) as f64);
        result = Some((
            numbered
                .iter()
                .map(|(a, _, index)| CountedGap {
                    gap: a.gap,
                    index: *index as i32,
                })
                .collect(),
            error,
        ));
        let next: Vec<Stepped> = numbered
            .iter()
            .map(|(a, _, index)| Stepped {
                t: a.t,
                index: *index,
            })
            .collect();
        map = fit_edge_map(&next, white_keys);
    }
    result
}

#[derive(Clone, Copy)]
struct EdgeMap {
    p: f64,
    q: f64,
    r: f64,
}

impl EdgeMap {
    fn index_at(&self, t: f64) -> f64 {
        (t - self.q) / (self.p - self.r * t)
    }
}

/// The 1-D projective map t = (p i + q) / (r i + 1) from a gap's index to where it sits along the
/// front edge, by least squares through the gaps and the front corners.
fn fit_edge_map(points: &[Stepped], white_keys: Option<usize>) -> Option<EdgeMap> {
    if points.len() < COUNT_LEAST_GAPS {
        return None;
    }
    let mut normal = vec![vec![0.0; 3]; 3];
    let mut rhs = vec![0.0; 3];
    let mut weighted: Vec<(f64, f64, f64)> = points.iter().map(|p| (p.t, p.index, 1.0)).collect();
    let anchor_weight = points.len() as f64;
    weighted.push((0.0, 0.0, anchor_weight));
    if let Some(keys) = white_keys {
        weighted.push((1.0, keys as f64, anchor_weight));
    }
    for (t, index, weight) in weighted {
        let row = [index, 1.0, -index * t];
        for a in 0..3 {
            rhs[a] += weight * row[a] * t;
            for b in 0..3 {
                normal[a][b] += weight * row[a] * row[b];
            }
        }
    }
    let s = solve(&normal, &rhs);
    s.iter().all(|v| v.is_finite()).then(|| EdgeMap {
        p: s[0],
        q: s[1],
        r: s[2],
    })
}

fn counted_estimator(peaks: &Peaks, template: &KeyboardTemplate) -> Option<Estimator> {
    let corners = complete_corners(peaks);
    let guess = template_x_estimator(template, &corners);
    let counted = count_gaps(peaks, Some(template.gaps.len() + 1), guess.as_deref())?;
    let mut pairs: Vec<Correspondence> = counted
        .iter()
        .map(|c| Correspondence {
            src: Point {
                x: f64::from(c.index),
                y: 1.0,
            },
            dst: c.gap.point(),
        })
        .collect();
    for (channel, point) in complete_corners(peaks).iter().enumerate() {
        if let Some(point) = point {
            pairs.push(Correspondence {
                src: template.corners[channel],
                dst: *point,
            });
        }
    }
    let inverse = invert_homography(&refine_homography(&pairs)?)?;
    Some(Box::new(move |p| apply_homography(&inverse, p.x, p.y).x))
}

/// How a peak is placed on the template before any fit exists: through the previous frame's fit
/// while tracking, or from the corners, moved by a whole number of keys, when the board is first
/// found.
pub enum Indexer {
    Prior(Homography),
    Corners(f64),
}

pub fn prior_indexer(prior: &Fit) -> Option<Indexer> {
    invert_homography(&prior.homography).map(Indexer::Prior)
}

fn build_pool(
    peaks: &Peaks,
    template: &KeyboardTemplate,
    white_keys: usize,
    indexer: &Indexer,
) -> Vec<Correspondence> {
    let mut pool = Vec::new();
    for (channel, point) in peaks.corners.iter().enumerate() {
        if let Some(point) = point {
            pool.push(Correspondence {
                src: template.corners[channel],
                dst: point.point(),
            });
        }
    }
    let template_x_of: Option<Estimator> = match indexer {
        Indexer::Prior(inverse) => {
            let inverse = *inverse;
            Some(Box::new(move |p| apply_homography(&inverse, p.x, p.y).x))
        }
        Indexer::Corners(shift) => {
            let shift = *shift;
            counted_estimator(peaks, template)
                .or_else(|| template_x_estimator(template, &complete_corners(peaks)))
                .map(|estimate| Box::new(move |p| estimate(p) + shift) as Estimator)
        }
    };
    let Some(template_x_of) = template_x_of else {
        return pool;
    };
    pool.extend(snap_gaps(&peaks.gaps, &template_x_of, white_keys, 1.0));
    pool.extend(snap_gaps(&peaks.back_gaps, &template_x_of, white_keys, 0.0));
    pool.extend(snap_blacks(
        &peaks.black_low,
        &template.black_low,
        &template_x_of,
    ));
    pool.extend(snap_blacks(
        &peaks.black_high,
        &template.black_high,
        &template_x_of,
    ));
    pool
}

/// Each detected gap of one edge, `row` deep, taken for the board's gap `template_x_of` puts it
/// nearest, when it sits close enough to one.
pub(crate) fn snap_gaps(
    gaps: &[ScoredPoint],
    template_x_of: &dyn Fn(Point) -> f64,
    white_keys: usize,
    row: f64,
) -> Vec<Correspondence> {
    gaps.iter()
        .filter_map(|gap| {
            let estimate = template_x_of(gap.point());
            let index = js_round(estimate);
            (index >= 1.0
                && index <= white_keys.saturating_sub(1) as f64
                && (estimate - index).abs() < GAP_SNAP_KEYS)
                .then_some(Correspondence {
                    src: Point { x: index, y: row },
                    dst: gap.point(),
                })
        })
        .collect()
}

/// Each detected black-key corner taken for the template corner `template_x_of` puts it nearest,
/// when it sits close enough to one.
pub(crate) fn snap_blacks(
    points: &[ScoredPoint],
    template_points: &[Point],
    template_x_of: &dyn Fn(Point) -> f64,
) -> Vec<Correspondence> {
    points
        .iter()
        .filter_map(|point| {
            let estimate = template_x_of(point.point());
            nearest_by_x(template_points, estimate)
                .filter(|nearest| (nearest.x - estimate).abs() < BLACK_SNAP_KEYS)
                .map(|nearest| Correspondence {
                    src: nearest,
                    dst: point.point(),
                })
        })
        .collect()
}

fn key_width(quad: &[Point; 4], white_keys: usize) -> f64 {
    (distance(quad[0], quad[1]) + distance(quad[3], quad[2])) / 2.0 / white_keys as f64
}

pub fn key_width_px(quad: &[Point; 4], white_keys: usize, frame: Size) -> f64 {
    key_width(&quad.map(|p| to_pixels(p, frame)), white_keys)
}

pub fn to_pixels(p: Point, frame: Size) -> Point {
    Point {
        x: p.x * frame.width,
        y: p.y * frame.height,
    }
}

pub const RANSAC_SEED: u32 = 0x9e37_79b9;
const RANSAC_ITERATIONS: usize = 200;
const RANSAC_VOTE_TOLERANCE: f64 = 0.25;
const MIN_INLIER_SHARE: f64 = 0.6;
const REPROJECTION_TOLERANCE: f64 = 0.15;
const TRIPLET_MIN_SIN: f64 = 1e-6;

fn pick_four_distinct(count: usize, rng: &mut Mulberry32) -> Vec<usize> {
    let mut chosen = Vec::new();
    let mut guard = 0;
    while chosen.len() < 4 && guard < 200 {
        let pick = (rng.next_f64() * count as f64).floor() as usize;
        if !chosen.contains(&pick) {
            chosen.push(pick);
        }
        guard += 1;
    }
    chosen
}

fn triplet_sine(a: Point, b: Point, c: Point) -> f64 {
    let ab = Point {
        x: b.x - a.x,
        y: b.y - a.y,
    };
    let ac = Point {
        x: c.x - a.x,
        y: c.y - a.y,
    };
    let cross = (ab.x * ac.y - ab.y * ac.x).abs();
    let scale = ab.x.hypot(ab.y) * ac.x.hypot(ac.y);
    if scale <= 0.0 {
        0.0
    } else {
        cross / scale
    }
}

/// True when any 3 of the 4 points are (near) collinear.
fn is_degenerate(points: &[Point]) -> bool {
    (0..4).any(|skip| {
        let rest: Vec<Point> = points
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, p)| *p)
            .collect();
        triplet_sine(rest[0], rest[1], rest[2]) < TRIPLET_MIN_SIN
    })
}

/// A white key's width in the frame where the homography puts `at`.
pub fn local_key_width(h: &Homography, at: Point) -> f64 {
    let low = apply_homography(h, at.x - 0.5, at.y);
    let high = apply_homography(h, at.x + 0.5, at.y);
    (high.x - low.x).hypot(high.y - low.y)
}

fn inliers_of(pool: &[Correspondence], h: &Homography) -> Vec<Correspondence> {
    pool.iter()
        .filter(|pair| {
            let tolerance = RANSAC_VOTE_TOLERANCE * local_key_width(h, pair.src);
            let proj = apply_homography(h, pair.src.x, pair.src.y);
            tolerance.is_finite() && (proj.x - pair.dst.x).hypot(proj.y - pair.dst.y) < tolerance
        })
        .copied()
        .collect()
}

fn four_points(sample: &[Correspondence]) -> ([Point; 4], [Point; 4]) {
    (
        [sample[0].src, sample[1].src, sample[2].src, sample[3].src],
        [sample[0].dst, sample[1].dst, sample[2].dst, sample[3].dst],
    )
}

fn fit_hypothesis(
    peaks: &Peaks,
    white_keys: usize,
    phase: Phase,
    frame: Size,
    indexer: &Indexer,
    black_depth: f64,
) -> Option<Fit> {
    let template = keyboard_template(white_keys, phase, black_depth);
    let pool = build_pool(peaks, &template, white_keys, indexer);
    if pool.len() < 4 {
        return None;
    }
    let mut rng = Mulberry32::new(RANSAC_SEED);
    let mut best_sample: Option<Vec<Correspondence>> = None;
    let mut best_count: i64 = -1;
    for _ in 0..RANSAC_ITERATIONS {
        let picks = pick_four_distinct(pool.len(), &mut rng);
        if picks.len() < 4 {
            continue;
        }
        let sample: Vec<Correspondence> = picks.iter().map(|i| pool[*i]).collect();
        let (src, dst) = four_points(&sample);
        if is_degenerate(&src) {
            continue;
        }
        let h = find_homography(&src, &dst);
        if !is_finite_homography(&h) {
            continue;
        }
        let count = inliers_of(&pool, &h).len() as i64;
        if count > best_count {
            best_count = count;
            best_sample = Some(sample);
        }
    }
    let seed_sample = best_sample?;
    let (src, dst) = four_points(&seed_sample);
    let seed_h = find_homography(&src, &dst);
    let seed_inliers = inliers_of(&pool, &seed_h);
    if seed_inliers.len() < 4 {
        return None;
    }
    let refined = refine_homography(&seed_inliers).unwrap_or(seed_h);
    let quad = template
        .corners
        .map(|c| apply_homography(&refined, c.x, c.y));
    let final_inliers = inliers_of(&pool, &refined);
    let inlier_share = final_inliers.len() as f64 / pool.len() as f64;
    if inlier_share < MIN_INLIER_SHARE || !check_quad(&quad) {
        return None;
    }
    let width_px = key_width_px(&quad, white_keys, frame);
    let mut errors_px: Vec<f64> = final_inliers
        .iter()
        .map(|pair| {
            let proj = apply_homography(&refined, pair.src.x, pair.src.y);
            ((proj.x - pair.dst.x) * frame.width).hypot((proj.y - pair.dst.y) * frame.height)
        })
        .collect();
    errors_px.sort_by(by_value);
    let median_error = errors_px.get(errors_px.len() / 2).copied()?;
    if width_px.is_nan() || width_px <= 0.0 || median_error >= REPROJECTION_TOLERANCE * width_px {
        return None;
    }
    let detected = peaks.corners.iter().flatten().count()
        + peaks.gaps.len()
        + peaks.back_gaps.len()
        + peaks.black_low.len()
        + peaks.black_high.len();
    Some(Fit {
        homography: refined,
        quad,
        white_keys,
        phase,
        inlier_share,
        explained: final_inliers.len() as f64 / detected.max(1) as f64,
        both_ends: end_seen(peaks, &refined, &template, [0, 3])
            && end_seen(peaks, &refined, &template, [1, 2]),
        beyond_ends: count_beyond_ends(peaks, &refined, white_keys, black_depth),
        gap_spacing: gap_spacing(peaks, &refined),
        reprojection_error: median_error,
        lift: None,
        black_depth,
        lens: Lens::default(),
    })
}

// the model can put the corner nearest the camera on the last gap instead of the board's end
const END_SLACK_KEYS: f64 = 1.25;

fn end_seen(
    peaks: &Peaks,
    h: &Homography,
    template: &KeyboardTemplate,
    channels: [usize; 2],
) -> bool {
    channels.iter().any(|channel| {
        let Some(detected) = peaks.corners[*channel] else {
            return false;
        };
        let end = template.corners[*channel];
        let fitted = apply_homography(h, end.x, end.y);
        (fitted.x - detected.x).hypot(fitted.y - detected.y)
            < END_SLACK_KEYS * local_key_width(h, end)
    })
}

const GAP_SPACING_DEPTH: f64 = 0.3;
const GAP_SPACING_LEAST: f64 = 0.4;
const GAP_SPACING_MOST: f64 = 1.5;
const GAP_SPACING_LEAST_STEPS: usize = 6;
// the closest standard sizes, 45 and 52 keys, space each other's gaps 0.87 or 1.16 apart
const GAP_SPACING_TOLERANCE: f64 = 0.08;

/// The median step between neighbouring detected gaps in template keys.
pub fn gap_spacing(peaks: &Peaks, h: &Homography) -> Option<f64> {
    let inverse = invert_homography(h)?;
    let mut along: Vec<f64> = peaks
        .gaps
        .iter()
        .map(|p| apply_homography(&inverse, p.x, p.y))
        .filter(|t| (t.y - 1.0).abs() < GAP_SPACING_DEPTH)
        .map(|t| t.x)
        .collect();
    along.sort_by(by_value);
    let mut steps: Vec<f64> = along
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|step| *step > GAP_SPACING_LEAST && *step < GAP_SPACING_MOST)
        .collect();
    steps.sort_by(by_value);
    if steps.len() < GAP_SPACING_LEAST_STEPS {
        None
    } else {
        Some(steps[steps.len() / 2])
    }
}

const BEYOND_ENDS_SLACK: f64 = 0.5;
const BEYOND_ENDS_ROW: f64 = 0.15;
// a hand or a reflection past an end can raise a peak or two
const BEYOND_ENDS_ALLOWED: usize = 2;

pub fn count_beyond_ends(
    peaks: &Peaks,
    h: &Homography,
    white_keys: usize,
    black_depth: f64,
) -> usize {
    let Some(inverse) = invert_homography(h) else {
        return 0;
    };
    let beyond = |points: &[ScoredPoint], row: f64| {
        points
            .iter()
            .filter(|p| {
                let t = apply_homography(&inverse, p.x, p.y);
                (t.y - row).abs() < BEYOND_ENDS_ROW
                    && (t.x < -BEYOND_ENDS_SLACK || t.x > white_keys as f64 + BEYOND_ENDS_SLACK)
            })
            .count()
    };
    let blacks: Vec<ScoredPoint> = peaks
        .black_low
        .iter()
        .chain(&peaks.black_high)
        .copied()
        .collect();
    beyond(&peaks.gaps, 1.0) + beyond(&blacks, black_depth)
}

/// Fits the keyboard template, its black keys `black_depth` deep, to the detected peaks and keeps
/// the best-explaining board. With no `lock` every standard board size and phase is tried.
pub fn fit_keyboard(
    peaks: &Peaks,
    lock: Option<Lock>,
    frame: Size,
    prior: Option<&Fit>,
    black_depth: f64,
) -> Option<Fit> {
    fit_candidates(peaks, lock, frame, prior, black_depth)
        .into_iter()
        .fold(None, |best: Option<Fit>, fit| match best {
            Some(b) if fit.explained <= b.explained => Some(b),
            _ => Some(fit),
        })
}

const ACQUIRE_SHIFTS: [f64; 5] = [0.0, -1.0, 1.0, -2.0, 2.0];

/// Every acceptable fit of the template to the peaks, over each candidate board and numbering,
/// for the tracker to vote on across frames when a board is found.
pub fn fit_candidates(
    peaks: &Peaks,
    lock: Option<Lock>,
    frame: Size,
    prior: Option<&Fit>,
    black_depth: f64,
) -> Vec<Fit> {
    let candidates: Vec<Lock> = match lock {
        Some(lock) => vec![lock],
        None => STANDARD_BOARDS
            .iter()
            .map(|b| Lock {
                white_keys: b.white_keys,
                phase: b.phase,
            })
            .collect(),
    };
    let from_prior = match (lock, prior) {
        (Some(lock), Some(prior)) if lock == prior.lock() => prior_indexer(prior),
        _ => None,
    };
    // the corners can sit a key off the board's true end, which numbers every key one off; the
    // black keys' pattern only lines up at the right numbering, so a board found for the first
    // time is tried at a few
    let indexers: Vec<Indexer> = match (from_prior, lock) {
        (Some(prior), _) => vec![prior],
        (None, Some(_)) => vec![Indexer::Corners(0.0)],
        (None, None) => ACQUIRE_SHIFTS
            .iter()
            .map(|s| Indexer::Corners(*s))
            .collect(),
    };
    let mut accepted = Vec::new();
    for candidate in &candidates {
        for indexer in &indexers {
            let Some(fit) = fit_hypothesis(
                peaks,
                candidate.white_keys,
                candidate.phase,
                frame,
                indexer,
                black_depth,
            ) else {
                continue;
            };
            // a board found for the first time must show both its ends, leave no keys past them
            // and space the gaps one key apart, or a crop that cut some keys off, a part of the
            // keyboard, or a board stretched over one of another size would pass for it
            let found_whole = fit.both_ends
                && fit.beyond_ends <= BEYOND_ENDS_ALLOWED
                && fit
                    .gap_spacing
                    .is_some_and(|s| (s - 1.0).abs() < GAP_SPACING_TOLERANCE);
            if lock.is_some() || found_whole {
                accepted.push(fit);
            }
        }
    }
    accepted
}
