use crate::decode::Peaks;
use crate::fit::Fit;
use crate::geom::{apply_homography, invert_homography, Point, ScoredPoint};
use crate::input::Pixels;
use crate::keys::board_keys;

/// How far either side of the fitted back edge we look for the real one, as a share of the
/// keys' depth. KeyNet's back points land within this of the edge where the keys meet the case.
const SNAP_REACH: f64 = 0.15;
const SNAP_STEPS: usize = 30;
/// The least a key must outshine the case behind it, in 0 to 255 brightness, for its back
/// edge to be read off the picture: on a pale case there is no edge to find.
const SNAP_CONTRAST: f64 = 40.0;
/// How far from the case's brightness to the key's a key has to climb to have begun. The case's
/// lip shades the keys' back ends, so we take where that shaded part begins, low on the climb,
/// rather than where the shade lifts off the key.
const SNAP_RISE: f64 = 0.3;

/// Where along a white key's back strip, in white keys, we read its edge: the middle of the
/// widest stretch the black keys leave open, so a black key's own end is never taken for it.
fn open_middles(fit: &Fit) -> Vec<f64> {
    let keys = board_keys(fit.white_keys, fit.phase);
    let blacks: Vec<(f64, f64)> = keys
        .iter()
        .filter(|key| key.black)
        .map(|key| (key.from, key.to))
        .collect();
    keys.iter()
        .filter(|key| !key.black)
        .map(|key| {
            let mut edges = vec![key.from, key.to];
            for (from, to) in &blacks {
                edges.extend([from.clamp(key.from, key.to), to.clamp(key.from, key.to)]);
            }
            edges.sort_by(f64::total_cmp);
            edges
                .windows(2)
                .filter(|pair| {
                    let middle = (pair[0] + pair[1]) / 2.0;
                    !blacks
                        .iter()
                        .any(|(from, to)| middle > *from && middle < *to)
                })
                .max_by(|a, b| (a[1] - a[0]).total_cmp(&(b[1] - b[0])))
                .map_or((key.from + key.to) / 2.0, |pair| (pair[0] + pair[1]) / 2.0)
        })
        .collect()
}

fn brightness(pixels: &Pixels, p: Point) -> f64 {
    let [r, g, b] = pixels.bilinear(
        (p.x * pixels.width as f64 - 0.5) as f32,
        (p.y * pixels.height as f64 - 0.5) as f32,
    );
    f64::from(0.299 * r + 0.587 * g + 0.114 * b)
}

/// Where the white key at `along` starts behind its back edge, in the template's depth: we walk
/// back from the brightest of the key in front of `around` to where it has sunk most of the way
/// to the dark case. Walking from the key means a lit button on the case behind is never reached.
fn back_edge_at(fit: &Fit, pixels: &Pixels, along: f64, around: f64) -> Option<f64> {
    let depth_at = |share: f64| around - SNAP_REACH + 2.0 * SNAP_REACH * share;
    let seen: Vec<f64> = (0..=SNAP_STEPS)
        .map(|step| {
            let template = Point {
                x: along,
                y: depth_at(step as f64 / SNAP_STEPS as f64),
            };
            let straight = apply_homography(&fit.homography, template.x, template.y);
            brightness(pixels, fit.lens.bend(straight))
        })
        .collect();
    let (brightest, key) = seen
        .iter()
        .enumerate()
        .skip(SNAP_STEPS / 2)
        .max_by(|a, b| a.1.total_cmp(b.1))?;
    let case = seen.iter().copied().fold(f64::INFINITY, f64::min);
    if key - case < SNAP_CONTRAST {
        return None;
    }
    let begun = case + SNAP_RISE * (key - case);
    let first_dark = (0..brightest).rev().find(|step| seen[*step] < begun)?;
    let (dark, lit) = (seen[first_dark], seen[first_dark + 1]);
    let between = (begun - dark) / (lit - dark);
    Some(depth_at((first_dark as f64 + between) / SNAP_STEPS as f64))
}

/// KeyNet's back points moved along their keys onto the edge where the keys meet the case, read
/// off the picture on the white keys either side of each. KeyNet places them a little in front
/// of that edge where the keys run into shadow, and the board is fitted through them.
pub fn snap_back_gaps(peaks: &Peaks, fit: &Fit, pixels: &Pixels) -> Peaks {
    let Some(inverse) = invert_homography(&fit.homography) else {
        return peaks.clone();
    };
    let middles = open_middles(fit);
    let back_gaps = peaks
        .back_gaps
        .iter()
        .map(|peak| {
            let straight = fit.lens.straighten(peak.point());
            let on_template = apply_homography(&inverse, straight.x, straight.y);
            let gap = on_template.x.round();
            let sides: Vec<f64> = [gap - 1.0, gap]
                .iter()
                .filter_map(|key| middles.get(*key as usize).filter(|_| *key >= 0.0))
                .filter_map(|along| back_edge_at(fit, pixels, *along, on_template.y))
                .collect();
            if sides.is_empty() {
                return *peak;
            }
            let edge = sides.iter().sum::<f64>() / sides.len() as f64;
            let moved = fit
                .lens
                .bend(apply_homography(&fit.homography, on_template.x, edge));
            ScoredPoint {
                x: moved.x,
                y: moved.y,
                score: peak.score,
            }
        })
        .collect();
    Peaks {
        back_gaps,
        ..peaks.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::find_homography;
    use crate::keys::Phase;
    use crate::lens::Lens;

    const WIDTH: usize = 640;
    const HEIGHT: usize = 360;

    fn board() -> Fit {
        let quad = [
            Point { x: 0.1, y: 0.3 },
            Point { x: 0.9, y: 0.3 },
            Point { x: 0.9, y: 0.7 },
            Point { x: 0.1, y: 0.7 },
        ];
        let white_keys = 14;
        let template = [
            Point { x: 0.0, y: 0.0 },
            Point {
                x: white_keys as f64,
                y: 0.0,
            },
            Point {
                x: white_keys as f64,
                y: 1.0,
            },
            Point { x: 0.0, y: 1.0 },
        ];
        Fit {
            homography: find_homography(&template, &quad),
            quad,
            white_keys,
            phase: Phase::C,
            inlier_share: 1.0,
            explained: 1.0,
            both_ends: true,
            beyond_ends: 0,
            gap_spacing: None,
            reprojection_error: 0.0,
            lift: None,
            lens: Lens::default(),
        }
    }

    /// A dark case with the keybed bright from `back` onwards, in frame fractions.
    fn picture(back: f64) -> Vec<u8> {
        (0..WIDTH * HEIGHT)
            .flat_map(|at| {
                let y = (at / WIDTH) as f64 / HEIGHT as f64;
                let level = if y >= back { 230 } else { 20 };
                [level, level, level, 255]
            })
            .collect()
    }

    fn peaks_at(fit: &Fit, depth: f64) -> Peaks {
        Peaks {
            back_gaps: (1..fit.white_keys)
                .map(|gap| {
                    let p = apply_homography(&fit.homography, gap as f64, depth);
                    ScoredPoint {
                        x: p.x,
                        y: p.y,
                        score: 1.0,
                    }
                })
                .collect(),
            ..Peaks::default()
        }
    }

    #[test]
    fn moves_back_points_onto_the_edge_where_the_keys_meet_the_case() {
        let fit = board();
        let rgba = picture(0.3);
        let pixels = Pixels {
            rgba: &rgba,
            width: WIDTH,
            height: HEIGHT,
        };
        let snapped = snap_back_gaps(&peaks_at(&fit, 0.08), &fit, &pixels);
        for peak in &snapped.back_gaps {
            assert!(
                ((peak.y - 0.3) * HEIGHT as f64).abs() < 1.0,
                "{:.2} px off",
                (peak.y - 0.3) * HEIGHT as f64
            );
        }
    }

    #[test]
    fn leaves_back_points_alone_on_a_case_as_pale_as_the_keys() {
        let fit = board();
        let rgba = vec![230; WIDTH * HEIGHT * 4];
        let pixels = Pixels {
            rgba: &rgba,
            width: WIDTH,
            height: HEIGHT,
        };
        let peaks = peaks_at(&fit, 0.08);
        assert_eq!(
            snap_back_gaps(&peaks, &fit, &pixels).back_gaps,
            peaks.back_gaps
        );
    }
}
