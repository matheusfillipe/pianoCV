use serde::{Deserialize, Serialize};

use crate::geom::{apply_homography, Homography, ScoredPoint};

const PEAK_THRESHOLD: f64 = 0.3;
const CORNER_PEAK_THRESHOLD: f64 = 0.15;

/// The model's input size in track mode: an oriented crop around the last fitted quad.
pub const TRACK_WIDTH: usize = 768;
pub const TRACK_HEIGHT: usize = 160;
/// The model's input size in search mode: the whole frame squashed to a square.
pub const SEARCH_SIZE: usize = 256;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Peaks {
    /// Back-low, back-high, front-high, front-low: the best peak per channel.
    pub corners: [Option<ScoredPoint>; 4],
    pub gaps: Vec<ScoredPoint>,
    pub black_low: Vec<ScoredPoint>,
    pub black_high: Vec<ScoredPoint>,
    #[serde(default)]
    pub black_top_low: Vec<ScoredPoint>,
    #[serde(default)]
    pub black_top_high: Vec<ScoredPoint>,
    #[serde(default)]
    pub back_gaps: Vec<ScoredPoint>,
    #[serde(default)]
    pub black_back_low: Vec<ScoredPoint>,
    #[serde(default)]
    pub black_back_high: Vec<ScoredPoint>,
}

struct Heat<'a> {
    values: &'a [f32],
    width: usize,
    height: usize,
}

impl Heat<'_> {
    fn at(&self, channel: usize, i: isize, j: isize) -> f64 {
        if i < 0 || i >= self.height as isize || j < 0 || j >= self.width as isize {
            return f64::NEG_INFINITY;
        }
        f64::from(
            self.values[channel * self.width * self.height + i as usize * self.width + j as usize],
        )
    }

    fn centroid(&self, channel: usize, i: usize, j: usize) -> (f64, f64) {
        let mut weight_sum = 0.0;
        let mut i_sum = 0.0;
        let mut j_sum = 0.0;
        for di in -1..=1isize {
            for dj in -1..=1isize {
                let weight = self.at(channel, i as isize + di, j as isize + dj).max(0.0);
                weight_sum += weight;
                i_sum += weight * (i as isize + di) as f64;
                j_sum += weight * (j as isize + dj) as f64;
            }
        }
        if weight_sum > 0.0 {
            (i_sum / weight_sum, j_sum / weight_sum)
        } else {
            (i as f64, j as f64)
        }
    }

    fn predicted(&self, offsets: &[f32], channel: usize, i: usize, j: usize) -> (f64, f64) {
        let plane = self.width * self.height;
        let at = i * self.width + j;
        (
            i as f64 + f64::from(offsets[(2 * channel + 1) * plane + at]),
            j as f64 + f64::from(offsets[2 * channel * plane + at]),
        )
    }
}

fn decode_channel(
    heat: &Heat,
    channel: usize,
    to_frame: &Homography,
    offsets: Option<&[f32]>,
) -> Vec<ScoredPoint> {
    let threshold = if channel < 4 {
        CORNER_PEAK_THRESHOLD
    } else {
        PEAK_THRESHOLD
    };
    let mut points = Vec::new();
    for i in 0..heat.height {
        for j in 0..heat.width {
            let value = heat.at(channel, i as isize, j as isize);
            if value < threshold {
                continue;
            }
            let is_peak = (-1..=1isize).all(|di| {
                (-1..=1isize).all(|dj| {
                    (di == 0 && dj == 0)
                        || heat.at(channel, i as isize + di, j as isize + dj) <= value
                })
            });
            if !is_peak {
                continue;
            }
            let (ri, rj) = match offsets {
                None => heat.centroid(channel, i, j),
                Some(offsets) => heat.predicted(offsets, channel, i, j),
            };
            let frame = apply_homography(to_frame, 2.0 * rj + 0.5, 2.0 * ri + 0.5);
            points.push(ScoredPoint {
                x: frame.x,
                y: frame.y,
                score: value,
            });
        }
    }
    points
}

fn best_of(points: &[ScoredPoint]) -> Option<ScoredPoint> {
    points.iter().copied().fold(None, |best, point| match best {
        Some(b) if point.score <= b.score => Some(b),
        _ => Some(point),
    })
}

/// Decodes KeyNet's heatmap `[C, H/2, W/2]`: 3x3 non-maximum suppression, then each peak placed by
/// the model's predicted offset, or by a weighted centroid over its 3x3 neighbourhood when the
/// model has no offset head. `to_frame` maps a model input pixel to a frame fraction, so a
/// perspective crop works as well as the affine ones. `width`/`height` are the model input size.
pub fn decode_heatmaps(
    heat: &[f32],
    width: usize,
    height: usize,
    to_frame: &Homography,
    offsets: Option<&[f32]>,
) -> Peaks {
    let grid = Heat {
        values: heat,
        width: width / 2,
        height: height / 2,
    };
    let plane = grid.width * grid.height;
    let channel_count = if plane == 0 {
        0
    } else {
        (heat.len() as f64 / plane as f64).round() as usize
    };
    let channels: Vec<Vec<ScoredPoint>> = (0..channel_count)
        .map(|channel| decode_channel(&grid, channel, to_frame, offsets))
        .collect();
    let channel = |index: usize| channels.get(index).cloned().unwrap_or_default();
    Peaks {
        corners: [0, 1, 2, 3].map(|index| best_of(&channel(index))),
        gaps: channel(4),
        black_low: channel(5),
        black_high: channel(6),
        black_top_low: channel(7),
        black_top_high: channel(8),
        back_gaps: channel(9),
        black_back_low: channel(10),
        black_back_high: channel(11),
    }
}
