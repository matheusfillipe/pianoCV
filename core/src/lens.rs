use serde::{Deserialize, Serialize};

use crate::decode::Peaks;
use crate::fit::Fit;
use crate::geom::{
    apply_homography, invert_homography, refine_homography, Correspondence, Point, ScoredPoint,
    Size,
};
use crate::keys::keyboard_template;

/// How the camera's lens bends straight lines, as one radial term about the middle of the
/// frame. A point at `r` from the middle, measured so the frame's corners sit at 1, is pulled
/// in to `r / (1 + k r²)` once the bend is taken out, so a negative `k` is the barrel of a
/// wide lens.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Lens {
    pub k: f64,
    /// The frame's width over its height, so the bend is round in pixels.
    pub aspect: f64,
}

impl Default for Lens {
    fn default() -> Self {
        Self {
            k: 0.0,
            aspect: 1.0,
        }
    }
}

impl Lens {
    pub fn straight(frame: Size) -> Self {
        Self {
            k: 0.0,
            aspect: frame.width / frame.height,
        }
    }

    fn reach(&self) -> f64 {
        (self.aspect * self.aspect + 1.0).sqrt() / 2.0
    }

    /// Where a point of the frame, in fractions, lies once the bend is taken out.
    pub fn straighten(&self, p: Point) -> Point {
        let (dx, dy) = ((p.x - 0.5) * self.aspect, p.y - 0.5);
        let r2 = (dx * dx + dy * dy) / (self.reach() * self.reach());
        let scale = 1.0 / (1.0 + self.k * r2);
        Point {
            x: 0.5 + dx * scale / self.aspect,
            y: 0.5 + dy * scale,
        }
    }

    /// Where a straightened point lands in the frame the camera films, the inverse of
    /// `straighten`. Straightening takes `r` to `r / (1 + k r²)`, so going back means solving
    /// `k s r² - r + s = 0` for `r`, at the root nearest `s`.
    pub fn bend(&self, p: Point) -> Point {
        let reach = self.reach();
        let (dx, dy) = ((p.x - 0.5) * self.aspect / reach, (p.y - 0.5) / reach);
        let s = dx.hypot(dy);
        let room = 1.0 - 4.0 * self.k * s * s;
        if s < 1e-12 || self.k.abs() < 1e-12 || room < 0.0 {
            return p;
        }
        let r = (1.0 - room.sqrt()) / (2.0 * self.k * s);
        let scale = r / s;
        Point {
            x: 0.5 + dx * scale * reach / self.aspect,
            y: 0.5 + dy * scale * reach,
        }
    }
}

/// How far a detected point may sit from the key it is taken for, in white keys.
const MATCH_KEYS: f64 = 0.3;
/// The fewest matched points a bend is measured from: below this a bend fits noise.
const LEAST_MATCHES: usize = 16;
/// The bends we search between. A phone's main lens or a webcam falls well inside them.
const SEARCH_LIMIT: f64 = 0.5;
const SEARCH_STEPS: usize = 24;

/// Each detected point taken for the template point it belongs to, under `fit`, which was
/// fitted on points straightened by `lens`.
fn matches(peaks: &Peaks, fit: &Fit, lens: &Lens) -> Vec<(Point, Point)> {
    let Some(inverse) = invert_homography(&fit.homography) else {
        return Vec::new();
    };
    let template = keyboard_template(fit.white_keys, fit.phase);
    let rows: [(&[ScoredPoint], &[Point]); 4] = [
        (&peaks.gaps, &template.gaps),
        (&peaks.back_gaps, &template.back_gaps),
        (&peaks.black_low, &template.black_low),
        (&peaks.black_high, &template.black_high),
    ];
    rows.iter()
        .flat_map(|(found, keys)| {
            found.iter().filter_map(|peak| {
                let seen = lens.straighten(peak.point());
                let on_template = apply_homography(&inverse, seen.x, seen.y);
                keys.iter()
                    .copied()
                    .filter(|key| {
                        (key.x - on_template.x).abs() < MATCH_KEYS
                            && (key.y - on_template.y).abs() < MATCH_KEYS
                    })
                    .min_by(|a, b| {
                        (a.x - on_template.x)
                            .abs()
                            .total_cmp(&(b.x - on_template.x).abs())
                    })
                    .map(|key| (key, peak.point()))
            })
        })
        .collect()
}

/// How far the matched points stray from their keys once straightened by a bend of `k` and
/// fitted with one homography, in white keys: straight edges stay straight only under the
/// lens's own bend.
fn stray(pairs: &[(Point, Point)], lens: &Lens) -> Option<f64> {
    let correspondences: Vec<Correspondence> = pairs
        .iter()
        .map(|(key, seen)| Correspondence {
            src: lens.straighten(*seen),
            dst: *key,
        })
        .collect();
    let to_template = refine_homography(&correspondences)?;
    let total: f64 = correspondences
        .iter()
        .map(|Correspondence { src, dst }| {
            let p = apply_homography(&to_template, src.x, src.y);
            (p.x - dst.x).powi(2) + (p.y - dst.y).powi(2)
        })
        .sum();
    Some(total / correspondences.len() as f64)
}

/// The bend that lines this frame's points up best with the board `fit` found, or None when
/// too few of them could be matched to say.
pub fn estimate_bend(peaks: &Peaks, fit: &Fit, lens: &Lens) -> Option<f64> {
    let pairs = matches(peaks, fit, lens);
    if pairs.len() < LEAST_MATCHES {
        return None;
    }
    let at = |k: f64| stray(&pairs, &Lens { k, ..*lens }).unwrap_or(f64::INFINITY);
    // A golden-section search, since the stray has one valley across the bends a lens can have.
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut low, mut high) = (-SEARCH_LIMIT, SEARCH_LIMIT);
    let mut one = high - ratio * (high - low);
    let mut two = low + ratio * (high - low);
    let (mut at_one, mut at_two) = (at(one), at(two));
    for _ in 0..SEARCH_STEPS {
        if at_one < at_two {
            high = two;
            two = one;
            at_two = at_one;
            one = high - ratio * (high - low);
            at_one = at(one);
        } else {
            low = one;
            one = two;
            at_one = at_two;
            two = low + ratio * (high - low);
            at_two = at(two);
        }
    }
    let best = (low + high) / 2.0;
    at(best).is_finite().then_some(best)
}

/// Takes the bend out of every point KeyNet found, so the board can be fitted with straight
/// edges.
pub fn straighten_peaks(peaks: &Peaks, lens: &Lens) -> Peaks {
    moved_peaks(peaks, |p| lens.straighten(p))
}

/// Every point KeyNet found, each moved the same way.
pub(crate) fn moved_peaks(peaks: &Peaks, to: impl Fn(Point) -> Point) -> Peaks {
    let one = |p: &ScoredPoint| {
        let moved = to(p.point());
        ScoredPoint {
            x: moved.x,
            y: moved.y,
            score: p.score,
        }
    };
    let each = |points: &[ScoredPoint]| -> Vec<ScoredPoint> { points.iter().map(one).collect() };
    Peaks {
        corners: peaks.corners.map(|c| c.as_ref().map(one)),
        gaps: each(&peaks.gaps),
        black_low: each(&peaks.black_low),
        black_high: each(&peaks.black_high),
        black_top_low: each(&peaks.black_top_low),
        black_top_high: each(&peaks.black_top_high),
        back_gaps: each(&peaks.back_gaps),
        black_back_low: each(&peaks.black_back_low),
        black_back_high: each(&peaks.black_back_high),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bending_a_straightened_point_brings_it_back() {
        let lens = Lens {
            k: -0.2,
            aspect: 16.0 / 9.0,
        };
        for p in [
            Point { x: 0.1, y: 0.2 },
            Point { x: 0.9, y: 0.85 },
            Point { x: 0.5, y: 0.5 },
        ] {
            let back = lens.bend(lens.straighten(p));
            assert!(
                (back.x - p.x).abs() < 1e-6 && (back.y - p.y).abs() < 1e-6,
                "{back:?}"
            );
        }
    }

    #[test]
    fn a_straight_lens_moves_nothing() {
        let lens = Lens::straight(Size {
            width: 1280.0,
            height: 720.0,
        });
        let p = Point { x: 0.2, y: 0.7 };
        assert_eq!(lens.straighten(p), p);
    }
}
