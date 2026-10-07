use serde::{Deserialize, Serialize};

use crate::decode::{Peaks, SEARCH_SIZE, TRACK_HEIGHT, TRACK_WIDTH};
use crate::fit::{to_pixels, Fit};
use crate::geom::{apply_homography, find_homography, invert_homography, Homography, Point, Size};
use crate::input::Pixels;
use crate::keys::BLACK_KEY_DEPTH;
use crate::lens::{estimate_bend, straighten_peaks, Lens};
use crate::snap::snap_back_gaps;
use crate::track::{ModelFrame, Tracker};

const END_MARGIN: f64 = 0.02;
const COARSE_PAD_ALONG: f64 = 0.12;
// the black keys' front edge is read coarsely in search mode, and a crop cut short at the back
// loses the keys' back corners
const COARSE_DEPTH_SLACK: f64 = 1.25;
const COARSE_LEAST_GAPS: usize = 4;
const COARSE_LEAST_POINTS: usize = 8;

const SEED_PRESENCE: f64 = 0.6;
const SEED_TRIES: u32 = 6;
const SEED_WIDEN: f64 = 1.3;

// the same margins the model was trained with, as shares of the quad's length
const MARGIN_ALONG: f64 = 0.08;
const MARGIN_ACROSS: f64 = 0.15;

/// Rotates and scales the frame so the keys run left to right with the player's edge at the
/// bottom, around a rough keybed quad; it never bends the frame, so keys keep their own shape.
#[derive(Clone, Copy, Debug)]
pub struct Crop {
    pub centre: Point,
    pub along: Point,
    pub across: Point,
    /// Frame pixels per crop pixel.
    pub scale: f64,
}

fn unit(p: Point) -> Point {
    let length = p.x.hypot(p.y);
    let length = if length == 0.0 { 1.0 } else { length };
    Point {
        x: p.x / length,
        y: p.y / length,
    }
}

fn dot(a: Point, b: Point) -> f64 {
    a.x * b.x + a.y * b.y
}

/// The crop around `quad`, given in frame pixels.
pub fn crop_for(quad: &[Point; 4], width: usize, height: usize) -> Crop {
    let [far_left, far_right, near_right, near_left] = *quad;
    let along = unit(Point {
        x: far_right.x - far_left.x + near_right.x - near_left.x,
        y: far_right.y - far_left.y + near_right.y - near_left.y,
    });
    let depth = Point {
        x: near_left.x - far_left.x + near_right.x - far_right.x,
        y: near_left.y - far_left.y + near_right.y - far_right.y,
    };
    let perpendicular = Point {
        x: -along.y,
        y: along.x,
    };
    let across = if dot(perpendicular, depth) < 0.0 {
        Point {
            x: -perpendicular.x,
            y: -perpendicular.y,
        }
    } else {
        perpendicular
    };
    let centre = Point {
        x: quad.iter().fold(0.0, |s, p| s + p.x) / 4.0,
        y: quad.iter().fold(0.0, |s, p| s + p.y) / 4.0,
    };
    let spread = |axis: Point| -> f64 {
        let values: Vec<f64> = quad
            .iter()
            .map(|p| {
                dot(
                    Point {
                        x: p.x - centre.x,
                        y: p.y - centre.y,
                    },
                    axis,
                )
            })
            .collect();
        values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - values.iter().copied().fold(f64::INFINITY, f64::min)
    };
    let along_extent = spread(along) * (1.0 + 2.0 * MARGIN_ALONG);
    let across_extent =
        spread(across) + (along_extent * MARGIN_ACROSS) / (1.0 + 2.0 * MARGIN_ALONG);
    let scale = (along_extent / width as f64)
        .max(across_extent / height as f64)
        .max(1e-6);
    Crop {
        centre,
        along,
        across,
        scale,
    }
}

/// The map from a crop pixel to a frame fraction, to hand to `decode_heatmaps`.
pub fn crop_matrix(crop: &Crop, width: usize, height: usize, frame: Size) -> Homography {
    let half_w = width as f64 / 2.0;
    let half_h = height as f64 / 2.0;
    let row = |along: f64, across: f64, centre: f64, extent: f64| -> [f64; 3] {
        [
            crop.scale * along / extent,
            crop.scale * across / extent,
            (centre - crop.scale * (half_w * along + half_h * across)) / extent,
        ]
    };
    let x = row(crop.along.x, crop.across.x, crop.centre.x, frame.width);
    let y = row(crop.along.y, crop.across.y, crop.centre.y, frame.height);
    [x[0], x[1], x[2], y[0], y[1], y[2], 0.0, 0.0, 1.0]
}

/// The map from a rectified crop pixel to a frame fraction: the quad (back-low, back-high,
/// front-high, front-low, in frame fractions) goes onto a fixed rectangle, so every key is equally
/// wide in the crop. None when the quad is degenerate.
pub fn rectified_matrix(
    quad: &[Point; 4],
    width: usize,
    height: usize,
    frame: Size,
) -> Option<Homography> {
    let (width, height) = (width as f64, height as f64);
    let left = width * MARGIN_ALONG / (1.0 + 2.0 * MARGIN_ALONG);
    let top = height * MARGIN_ACROSS / (1.0 + 2.0 * MARGIN_ACROSS);
    let rectangle = [
        Point { x: left, y: top },
        Point {
            x: width - left,
            y: top,
        },
        Point {
            x: width - left,
            y: height - top,
        },
        Point {
            x: left,
            y: height - top,
        },
    ];
    let to_crop = find_homography(&quad.map(|p| to_pixels(p, frame)), &rectangle);
    let [a, b, c, d, e, f, g, h, i] = invert_homography(&to_crop)?;
    Some([
        a / frame.width,
        b / frame.width,
        c / frame.width,
        d / frame.height,
        e / frame.height,
        f / frame.height,
        g,
        h,
        i,
    ])
}

/// Whether both ends of a fitted keybed lie inside the track crop taken around `crop_quad`, clear
/// of its edges: a fit whose ends touch the crop's edges may be part of a longer board the crop cut
/// short.
pub fn ends_inside_crop(
    fit_quad: &[Point; 4],
    crop_quad: &[Point; 4],
    frame: Size,
    rectified: bool,
) -> bool {
    let width = TRACK_WIDTH as f64;
    let inside = |x: f64| x >= width * END_MARGIN && x <= width * (1.0 - END_MARGIN);
    if rectified {
        let to_crop = rectified_matrix(crop_quad, TRACK_WIDTH, TRACK_HEIGHT, frame)
            .and_then(|m| invert_homography(&m));
        return to_crop.is_some_and(|m| {
            fit_quad
                .iter()
                .all(|p| inside(apply_homography(&m, p.x, p.y).x))
        });
    }
    let crop = crop_for(
        &crop_quad.map(|p| to_pixels(p, frame)),
        TRACK_WIDTH,
        TRACK_HEIGHT,
    );
    fit_quad.iter().all(|p| {
        let p = to_pixels(*p, frame);
        inside(
            width / 2.0
                + ((p.x - crop.centre.x) * crop.along.x + (p.y - crop.centre.y) * crop.along.y)
                    / crop.scale,
        )
    })
}

/// The quad stretched along the keyboard about its centre by `factor`.
pub fn widen_quad(quad: &[Point; 4], factor: f64) -> [Point; 4] {
    let centre = Point {
        x: quad.iter().fold(0.0, |s, p| s + p.x) / 4.0,
        y: quad.iter().fold(0.0, |s, p| s + p.y) / 4.0,
    };
    let dx = quad[1].x - quad[0].x + quad[2].x - quad[3].x;
    let dy = quad[1].y - quad[0].y + quad[2].y - quad[3].y;
    let length = dx.hypot(dy);
    let length = if length == 0.0 { 1.0 } else { length };
    let along = Point {
        x: dx / length,
        y: dy / length,
    };
    quad.map(|p| {
        let offset = (p.x - centre.x) * along.x + (p.y - centre.y) * along.y;
        Point {
            x: p.x + along.x * offset * (factor - 1.0),
            y: p.y + along.y * offset * (factor - 1.0),
        }
    })
}

fn mean(points: &[Point]) -> Point {
    let count = points.len() as f64;
    Point {
        x: points.iter().fold(0.0, |s, p| s + p.x) / count,
        y: points.iter().fold(0.0, |s, p| s + p.y) / count,
    }
}

/// A rough keybed quad, back-low, back-high, front-high, front-low in frame fractions, around
/// every point search mode found, to take the first track crop from. The gaps mark the front edge
/// and the black keys the back, and the camera's handedness then says which end is low. None when
/// too few points were found.
pub fn coarse_quad(peaks: &Peaks, frame: Size) -> Option<[Point; 4]> {
    let scale = |p: Point| to_pixels(p, frame);
    let gaps: Vec<Point> = peaks.gaps.iter().map(|p| scale(p.point())).collect();
    let blacks: Vec<Point> = peaks
        .black_low
        .iter()
        .chain(&peaks.black_high)
        .map(|p| scale(p.point()))
        .collect();
    let all: Vec<Point> = gaps
        .iter()
        .chain(&blacks)
        .copied()
        .chain(peaks.corners.iter().flatten().map(|p| scale(p.point())))
        .collect();
    if gaps.len() < COARSE_LEAST_GAPS || all.len() < COARSE_LEAST_POINTS || blacks.is_empty() {
        return None;
    }
    let centre = mean(&all);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for p in &all {
        xx += (p.x - centre.x).powi(2);
        xy += (p.x - centre.x) * (p.y - centre.y);
        yy += (p.y - centre.y).powi(2);
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let mut along = Point {
        x: angle.cos(),
        y: angle.sin(),
    };
    let mut across = Point {
        x: -along.y,
        y: along.x,
    };
    let front = mean(&gaps);
    let behind = mean(&blacks);
    let on_across =
        |p: Point, across: Point| (p.x - centre.x) * across.x + (p.y - centre.y) * across.y;
    if on_across(behind, across) < on_across(front, across) {
        across = Point {
            x: -across.x,
            y: -across.y,
        };
    }
    // a camera never mirrors, so with the front edge at the bottom the low keys are on the left,
    // as the player sees them: along runs so that it turns to across the way image x turns to
    // image up
    if along.x * across.y - along.y * across.x > 0.0 {
        along = Point {
            x: -along.x,
            y: -along.y,
        };
    }
    let on_along = |p: Point| (p.x - centre.x) * along.x + (p.y - centre.y) * along.y;
    let first = all
        .iter()
        .map(|p| on_along(*p))
        .fold(f64::INFINITY, f64::min);
    let last = all
        .iter()
        .map(|p| on_along(*p))
        .fold(f64::NEG_INFINITY, f64::max);
    // search mode misses some of the keys, often the ends, so the crop reaches past what it found
    let low = first - (last - first) * COARSE_PAD_ALONG;
    let high = last + (last - first) * COARSE_PAD_ALONG;
    let front_at = on_across(front, across);
    let back_at = front_at
        + ((on_across(behind, across) - front_at) / (1.0 - BLACK_KEY_DEPTH)) * COARSE_DEPTH_SLACK;
    let at = |a: f64, c: f64| Point {
        x: (centre.x + along.x * a + across.x * c) / frame.width,
        y: (centre.y + along.y * a + across.y * c) / frame.height,
    };
    Some([
        at(low, back_at),
        at(high, back_at),
        at(high, front_at),
        at(low, front_at),
    ])
}

/// Which crop the model runs on next, and how to decode what it returns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CropRequest {
    pub mode: Mode,
    pub width: usize,
    pub height: usize,
    /// Maps a model input pixel to a frame fraction.
    pub matrix: Homography,
    /// The quad the track crop is taken around, in frame fractions.
    pub quad: Option<[Point; 4]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Mode {
    Search,
    Track,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    pub fit: Option<Fit>,
    /// Whether this step found the board, after one or more without it.
    pub acquired: bool,
}

/// Search until a keyboard is seen, crop around it, and track it: the whole KeyNet loop from model
/// outputs to a fitted board. The caller runs the model on `next_crop` and hands the outputs to
/// `step`.
pub struct Session {
    tracker: Tracker,
    fit: Option<Fit>,
    // search mode sees where the keyboard is but its keys too small to fit, so the first track
    // crops come from a rough quad around what it found
    seed: Option<[Point; 4]>,
    seed_tries: u32,
    rectified: bool,
    lens: Option<Lens>,
    since_bend: u32,
}

/// How many tracked frames pass between two looks at the lens's bend, and how far each look
/// moves the bend we hold. The camera does not move while someone plays, so a slow average
/// of many frames settles on its lens and shrugs off a frame that read the keys badly.
const BEND_EVERY: u32 = 10;
const BEND_FOLLOW: f64 = 0.2;

impl Session {
    pub fn new(rectified: bool) -> Self {
        Self {
            tracker: Tracker::default(),
            fit: None,
            seed: None,
            seed_tries: 0,
            rectified,
            lens: None,
            since_bend: 0,
        }
    }

    pub fn rectified(&self) -> bool {
        self.rectified
    }

    pub fn fit(&self) -> Option<&Fit> {
        self.fit.as_ref()
    }

    /// Where the board stands in the picture the camera films, with the lens's bend put back.
    fn crop_around(&self) -> Option<[Point; 4]> {
        self.fit
            .as_ref()
            .map(|fit| fit.quad.map(|corner| fit.lens.bend(corner)))
            .or(self.seed)
    }

    pub fn next_crop(&self, frame: Size) -> CropRequest {
        match self.crop_around() {
            None => {
                let scale = 1.0 / SEARCH_SIZE as f64;
                CropRequest {
                    mode: Mode::Search,
                    width: SEARCH_SIZE,
                    height: SEARCH_SIZE,
                    matrix: [scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, 1.0],
                    quad: None,
                }
            }
            Some(quad) => {
                let rectified = self
                    .rectified
                    .then(|| rectified_matrix(&quad, TRACK_WIDTH, TRACK_HEIGHT, frame))
                    .flatten();
                let matrix = rectified.unwrap_or_else(|| {
                    let crop = crop_for(
                        &quad.map(|p| to_pixels(p, frame)),
                        TRACK_WIDTH,
                        TRACK_HEIGHT,
                    );
                    crop_matrix(&crop, TRACK_WIDTH, TRACK_HEIGHT, frame)
                });
                CropRequest {
                    mode: Mode::Track,
                    width: TRACK_WIDTH,
                    height: TRACK_HEIGHT,
                    matrix,
                    quad: Some(quad),
                }
            }
        }
    }

    /// `pixels` is the frame the model ran on, which the board's back edge is read off once a
    /// board is held.
    pub fn step(
        &mut self,
        result: &ModelFrame,
        pixels: Option<&Pixels>,
        size: Size,
        now_ms: f64,
    ) -> Step {
        let snapped = match (&self.fit, pixels) {
            (Some(fit), Some(pixels)) => ModelFrame {
                presence: result.presence,
                peaks: snap_back_gaps(&result.peaks, fit, pixels),
            },
            _ => result.clone(),
        };
        let result = &snapped;
        let Some(crop_around) = self.crop_around() else {
            self.seed = if result.presence >= SEED_PRESENCE {
                coarse_quad(&result.peaks, size)
            } else {
                None
            };
            self.seed_tries = 0;
            return Step {
                fit: self.fit.clone(),
                acquired: false,
            };
        };
        let was_tracking = self.fit.is_some();
        let mut lens = *self.lens.get_or_insert(Lens::straight(size));
        let straight = ModelFrame {
            presence: result.presence,
            peaks: straighten_peaks(&result.peaks, &lens),
        };
        self.fit = self.tracker.update(&straight, size, now_ms).map(|fit| {
            self.since_bend += 1;
            if self.since_bend >= BEND_EVERY {
                self.since_bend = 0;
                if let Some(k) = estimate_bend(&result.peaks, &fit, &lens) {
                    lens.k += BEND_FOLLOW * (k - lens.k);
                    self.lens = Some(lens);
                }
            }
            Fit { lens, ..fit }
        });
        if let Some(fit) = &self.fit {
            let seen = fit.quad.map(|corner| fit.lens.bend(corner));
            if !was_tracking && !ends_inside_crop(&seen, &crop_around, size, self.rectified) {
                self.tracker.reset();
                self.fit = None;
                self.seed = Some(widen_quad(&crop_around, SEED_WIDEN));
                return Step {
                    fit: None,
                    acquired: false,
                };
            }
        }
        if self.fit.is_none() && !was_tracking && self.seed.is_some() {
            // a track crop reads the keys far better than search, so even a failed one gives a
            // truer quad to crop around next
            self.seed = coarse_quad(&result.peaks, size).or(self.seed);
            self.seed_tries += 1;
            if self.seed_tries >= SEED_TRIES {
                self.seed = None;
            }
        }
        if self.fit.is_none() && was_tracking {
            self.seed = None;
        }
        Step {
            fit: self.fit.clone(),
            acquired: self.fit.is_some() && !was_tracking,
        }
    }
}
