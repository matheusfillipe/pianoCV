use crate::decode::Peaks;
use crate::fit::{snap_blacks, snap_gaps, Fit};
use crate::geom::{apply_homography, invert_homography, refine_homography, Point};
use crate::keys::keyboard_template;

/// The fewest gaps on each edge the keybed is placed from, and the fewest black-key fronts read
/// against it: below these a reading follows one bad peak.
const LEAST_EDGE_GAPS: usize = 6;
const LEAST_BLACK_FRONTS: usize = 6;
/// The black-key depths a real board has; a reading outside them is a bad frame.
const LEAST_DEPTH: f64 = 0.45;
const MOST_DEPTH: f64 = 0.85;

/// How much of the keybed's depth this board's black keys take, read off one frame's peaks
/// (straightened like the fit), or None when too few points were seen to say. Boards differ here,
/// and a fit that takes the wrong depth squeezes the keybed to make the black keys' fronts land,
/// dragging the back edge over the keys. So we place the keybed from the gaps on its two edges
/// alone and read where the black keys' fronts fall on it.
pub fn estimate_black_depth(peaks: &Peaks, fit: &Fit) -> Option<f64> {
    let inverse = invert_homography(&fit.homography)?;
    let template_x_of = |p: Point| apply_homography(&inverse, p.x, p.y).x;
    let back = snap_gaps(&peaks.back_gaps, &template_x_of, fit.white_keys, 0.0);
    let front = snap_gaps(&peaks.gaps, &template_x_of, fit.white_keys, 1.0);
    if back.len() < LEAST_EDGE_GAPS || front.len() < LEAST_EDGE_GAPS {
        return None;
    }
    let onto_keybed = invert_homography(&refine_homography(&[back, front].concat())?)?;
    let template = keyboard_template(fit.white_keys, fit.phase, fit.black_depth);
    let mut depths: Vec<f64> = [
        snap_blacks(&peaks.black_low, &template.black_low, &template_x_of),
        snap_blacks(&peaks.black_high, &template.black_high, &template_x_of),
    ]
    .concat()
    .iter()
    .map(|front| apply_homography(&onto_keybed, front.dst.x, front.dst.y).y)
    .filter(|depth| (LEAST_DEPTH..MOST_DEPTH).contains(depth))
    .collect();
    if depths.len() < LEAST_BLACK_FRONTS {
        return None;
    }
    depths.sort_by(f64::total_cmp);
    Some(depths[depths.len() / 2])
}
