use serde::Deserialize;

use crate::decode::{decode_heatmaps, Peaks};
use crate::fit::{count_gaps, fit_keyboard, local_key_width, Fit, Lift, Lock};
use crate::geom::{
    apply_homography, find_homography, invert_homography, Homography, Mulberry32, Point,
    ScoredPoint, Size,
};
use crate::input::{crop_input, Pixels};
use crate::keys::{
    key_units, keyboard_template, white_index, KeyboardTemplate, Phase, BLACK_KEY_DEPTH,
};
use crate::lift::{estimate_lift, key_net_faces, lifted};
use crate::session::{crop_for, crop_matrix, rectified_matrix, Mode, Session};
use crate::track::{ModelFrame, OneEuro, Tracker};

const FRAME: Size = Size {
    width: 3840.0,
    height: 2160.0,
};

fn gaussian_noise(rng: &mut Mulberry32, sigma: f64) -> f64 {
    let u1 = rng.next_f64().max(1e-9);
    let u2 = rng.next_f64();
    sigma * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

const IDENTITY: Homography = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

fn render_gaussian(
    heat: &mut [f32],
    channel: usize,
    (heat_width, heat_height): (usize, usize),
    (centre_i, centre_j): (f64, f64),
    sigma: f64,
    amplitude: f64,
) {
    let offset = channel * heat_width * heat_height;
    for di in -3..=3i64 {
        for dj in -3..=3i64 {
            let i = centre_i.round() as i64 + di;
            let j = centre_j.round() as i64 + dj;
            if i < 0 || i >= heat_height as i64 || j < 0 || j >= heat_width as i64 {
                continue;
            }
            let dy = i as f64 - centre_i;
            let dx = j as f64 - centre_j;
            let value = (amplitude * (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp()) as f32;
            let at = offset + i as usize * heat_width + j as usize;
            heat[at] = heat[at].max(value);
        }
    }
}

mod decoding {
    use super::*;

    const HEAT_WIDTH: usize = 40;
    const HEAT_HEIGHT: usize = 20;
    const GRID: (usize, usize) = (HEAT_WIDTH, HEAT_HEIGHT);

    fn decode(heat: &[f32], offsets: Option<&[f32]>) -> Peaks {
        decode_heatmaps(heat, HEAT_WIDTH * 2, HEAT_HEIGHT * 2, &IDENTITY, offsets)
    }

    #[test]
    fn recovers_a_gaussian_centre_within_a_tenth_of_an_input_pixel() {
        for (i, j) in [(10.0, 20.0), (10.05, 20.03), (9.95, 19.92), (5.06, 8.08)] {
            let mut heat = vec![0.0; 7 * HEAT_WIDTH * HEAT_HEIGHT];
            render_gaussian(&mut heat, 0, GRID, (i, j), 1.0, 1.0);
            let corner = decode(&heat, None).corners[0].expect("a corner");
            assert!((corner.x - (2.0 * j + 0.5)).abs() < 0.1);
            assert!((corner.y - (2.0 * i + 0.5)).abs() < 0.1);
        }
    }

    #[test]
    fn picks_the_best_scoring_peak_as_a_channels_corner() {
        let mut heat = vec![0.0; 7 * HEAT_WIDTH * HEAT_HEIGHT];
        render_gaussian(&mut heat, 1, GRID, (5.0, 5.0), 1.0, 0.6);
        render_gaussian(&mut heat, 1, GRID, (5.0, 30.0), 1.0, 0.9);
        let peaks = decode(&heat, None);
        assert!((peaks.corners[1].expect("a corner").score - 0.9).abs() < 1e-5);
        assert!(peaks.corners[0].is_none());
    }

    #[test]
    fn finds_every_peak_in_the_gap_channel() {
        let mut heat = vec![0.0; 7 * HEAT_WIDTH * HEAT_HEIGHT];
        for j in [5.0, 15.0, 25.0] {
            render_gaussian(&mut heat, 4, GRID, (8.0, j), 1.0, 1.0);
        }
        assert_eq!(decode(&heat, None).gaps.len(), 3);
    }

    #[test]
    fn places_a_peak_by_the_predicted_offset_when_there_is_one() {
        let plane = HEAT_WIDTH * HEAT_HEIGHT;
        let mut heat = vec![0.0; 9 * plane];
        render_gaussian(&mut heat, 4, GRID, (8.0, 5.0), 1.0, 1.0);
        let mut offsets = vec![0.0; 18 * plane];
        offsets[2 * 4 * plane + 8 * HEAT_WIDTH + 5] = 0.4;
        offsets[(2 * 4 + 1) * plane + 8 * HEAT_WIDTH + 5] = -0.3;
        let peaks = decode(&heat, Some(&offsets));
        assert_eq!(peaks.gaps.len(), 1);
        assert!((peaks.gaps[0].x - (2.0 * 5.4 + 0.5)).abs() < 1e-5);
        assert!((peaks.gaps[0].y - (2.0 * 7.7 + 0.5)).abs() < 1e-5);
    }

    #[test]
    fn leaves_the_back_lists_empty_for_a_nine_channel_heatmap() {
        let mut heat = vec![0.0; 9 * HEAT_WIDTH * HEAT_HEIGHT];
        render_gaussian(&mut heat, 4, GRID, (8.0, 5.0), 1.0, 1.0);
        let peaks = decode(&heat, None);
        assert_eq!(peaks.gaps.len(), 1);
        assert!(peaks.back_gaps.is_empty());
        assert!(peaks.black_back_low.is_empty());
        assert!(peaks.black_back_high.is_empty());
    }

    #[test]
    fn reads_the_back_channels_of_a_twelve_channel_heatmap() {
        let mut heat = vec![0.0; 12 * HEAT_WIDTH * HEAT_HEIGHT];
        render_gaussian(&mut heat, 9, GRID, (4.0, 5.0), 1.0, 1.0);
        render_gaussian(&mut heat, 10, GRID, (4.0, 10.0), 1.0, 1.0);
        render_gaussian(&mut heat, 11, GRID, (4.0, 15.0), 1.0, 1.0);
        let peaks = decode(&heat, None);
        assert_eq!(peaks.back_gaps.len(), 1);
        assert_eq!(peaks.black_back_low.len(), 1);
        assert_eq!(peaks.black_back_high.len(), 1);
    }

    #[test]
    fn drops_anything_under_the_threshold() {
        let mut heat = vec![0.0; 7 * HEAT_WIDTH * HEAT_HEIGHT];
        render_gaussian(&mut heat, 5, GRID, (10.0, 10.0), 1.0, 0.25);
        assert!(decode(&heat, None).black_low.is_empty());
    }

    #[test]
    fn maps_a_peak_through_a_perspective_crop() {
        let mut heat = vec![0.0; 7 * HEAT_WIDTH * HEAT_HEIGHT];
        render_gaussian(&mut heat, 4, GRID, (10.0, 20.0), 1.0, 1.0);
        let halve = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0];
        let peaks = decode_heatmaps(&heat, HEAT_WIDTH * 2, HEAT_HEIGHT * 2, &halve, None);
        assert!((peaks.gaps[0].x - 20.25).abs() < 0.1);
    }
}

mod template {
    use super::*;

    #[test]
    fn spans_the_board_and_marks_every_white_key_gap() {
        let template = keyboard_template(7, Phase::C);
        assert_eq!(
            template.corners,
            [
                Point { x: 0.0, y: 0.0 },
                Point { x: 7.0, y: 0.0 },
                Point { x: 7.0, y: 1.0 },
                Point { x: 0.0, y: 1.0 },
            ]
        );
        let xs: Vec<f64> = template.gaps.iter().map(|g| g.x).collect();
        assert_eq!(xs, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(template.gaps.iter().all(|g| g.y == 1.0));
    }

    #[test]
    fn places_one_black_key_between_every_white_pair_but_e_f_and_b_c() {
        let template = keyboard_template(7, Phase::C);
        assert_eq!(template.black_low.len(), 5);
        assert_eq!(template.black_high.len(), 5);
        for (low, high) in template.black_low.iter().zip(&template.black_high) {
            assert!(high.x > low.x);
            assert!((low.y - BLACK_KEY_DEPTH).abs() < 1e-9);
            assert!((high.y - BLACK_KEY_DEPTH).abs() < 1e-9);
        }
    }

    #[test]
    fn agrees_with_the_key_layout_for_a_black_keys_position() {
        let template = keyboard_template(29, Phase::C);
        let base = f64::from(white_index(60));
        let expected = key_units(61);
        assert!((template.black_low[0].x - (expected.from - base)).abs() < 1e-9);
        assert!((template.black_high[0].x - (expected.to - base)).abs() < 1e-9);
    }
}

fn scored(p: Point, score: f64) -> ScoredPoint {
    ScoredPoint {
        x: p.x,
        y: p.y,
        score,
    }
}

fn homography_for(white_keys: usize, phase: Phase, quad: &[Point; 4]) -> Homography {
    find_homography(&keyboard_template(white_keys, phase).corners, quad)
}

fn in_frame(p: Point) -> bool {
    p.x > -0.02 && p.x < 1.02 && p.y > -0.02 && p.y < 1.02
}

#[derive(Clone, Copy)]
struct Synth {
    noise_px: f64,
    drop_fraction: f64,
    outlier_fraction: f64,
    seed: u32,
    lift: Option<Lift>,
}

impl Default for Synth {
    fn default() -> Self {
        Self {
            noise_px: 1.5,
            drop_fraction: 0.0,
            outlier_fraction: 0.0,
            seed: 1,
            lift: None,
        }
    }
}

fn lift_many(h: &Homography, lift: &Lift, points: &[Point]) -> Vec<Point> {
    points
        .iter()
        .map(|p| {
            let s = h[6] * p.x + h[7] * p.y + h[8] + lift[2];
            Point {
                x: (h[0] * p.x + h[1] * p.y + h[2] + lift[0]) / s,
                y: (h[3] * p.x + h[4] * p.y + h[5] + lift[1]) / s,
            }
        })
        .collect()
}

fn synth_peaks(white_keys: usize, phase: Phase, h: &Homography, options: Synth) -> Peaks {
    let mut rng = Mulberry32::new(options.seed);
    let template = keyboard_template(white_keys, phase);
    let with_noise = |rng: &mut Mulberry32, p: Point| Point {
        x: p.x + gaussian_noise(rng, options.noise_px) / FRAME.width,
        y: p.y + gaussian_noise(rng, options.noise_px) / FRAME.height,
    };
    let corners = template.corners.map(|c| {
        let proj = apply_homography(h, c.x, c.y);
        in_frame(proj).then(|| scored(with_noise(&mut rng, proj), 0.9))
    });
    let project_many = |rng: &mut Mulberry32, points: &[Point]| -> Vec<ScoredPoint> {
        let mut kept = Vec::new();
        for p in points {
            let proj = apply_homography(h, p.x, p.y);
            if !in_frame(proj) || rng.next_f64() < options.drop_fraction {
                continue;
            }
            kept.push(scored(with_noise(rng, proj), 0.9));
        }
        let outliers = (options.outlier_fraction * points.len() as f64).round() as usize;
        for _ in 0..outliers {
            let x = rng.next_f64();
            let y = rng.next_f64() * 0.5 + 0.2;
            kept.push(scored(Point { x, y }, 0.5));
        }
        kept
    };
    let gaps = project_many(&mut rng, &template.gaps);
    let black_low = project_many(&mut rng, &template.black_low);
    let black_high = project_many(&mut rng, &template.black_high);
    let tops = |rng: &mut Mulberry32, points: &[Point]| -> Vec<ScoredPoint> {
        options.lift.map_or_else(Vec::new, |lift| {
            lift_many(h, &lift, points)
                .into_iter()
                .map(|p| scored(with_noise(rng, p), 0.9))
                .collect()
        })
    };
    let black_top_low = tops(&mut rng, &template.black_low);
    let black_top_high = tops(&mut rng, &template.black_high);
    Peaks {
        corners,
        gaps,
        black_low,
        black_high,
        black_top_low,
        black_top_high,
        ..Peaks::default()
    }
}

fn with_back_points(peaks: &mut Peaks, template: &KeyboardTemplate, h: &Homography, lift: &Lift) {
    let project = |points: &[Point]| -> Vec<ScoredPoint> {
        points
            .iter()
            .map(|p| scored(apply_homography(h, p.x, p.y), 0.9))
            .collect()
    };
    let at_back = |points: &[Point]| -> Vec<Point> {
        points.iter().map(|p| Point { x: p.x, y: 0.0 }).collect()
    };
    let lifted_back = |points: &[Point]| -> Vec<ScoredPoint> {
        lift_many(h, lift, &at_back(points))
            .into_iter()
            .map(|p| scored(p, 0.9))
            .collect()
    };
    peaks.back_gaps = project(&template.back_gaps);
    peaks.black_back_low = lifted_back(&template.black_low);
    peaks.black_back_high = lifted_back(&template.black_high);
}

fn quad(points: [(f64, f64); 4]) -> [Point; 4] {
    points.map(|(x, y)| Point { x, y })
}

fn frontal() -> [Point; 4] {
    quad([(0.12, 0.3), (0.88, 0.3), (0.9, 0.62), (0.1, 0.62)])
}

fn steep() -> [Point; 4] {
    quad([(0.25, 0.35), (0.75, 0.35), (0.95, 0.55), (0.05, 0.55)])
}

fn rotated() -> [Point; 4] {
    let centre = Point { x: 0.5, y: 0.46 };
    let (cos, sin) = (0.25f64.cos(), 0.25f64.sin());
    frontal().map(|p| {
        let dx = p.x - centre.x;
        let dy = p.y - centre.y;
        Point {
            x: centre.x + dx * cos - dy * sin,
            y: centre.y + dx * sin + dy * cos,
        }
    })
}

fn receding() -> [Point; 4] {
    quad([(0.425, 0.88), (0.453, 0.144), (0.528, 0.138), (0.605, 0.88)])
}

fn partial() -> [Point; 4] {
    quad([(-0.06, 0.3), (0.88, 0.3), (0.9, 0.62), (0.1, 0.62)])
}

const BOARDS: [(usize, Phase); 4] = [
    (29, Phase::C),
    (36, Phase::C),
    (45, Phase::E),
    (52, Phase::A),
];

fn expect_close_quad(found: &[Point; 4], truth: &[Point; 4], tolerance_px: f64) {
    for (p, t) in found.iter().zip(truth) {
        let dx = (p.x - t.x) * FRAME.width;
        let dy = (p.y - t.y) * FRAME.height;
        assert!(dx.hypot(dy) < tolerance_px);
    }
}

const NOISY: Synth = Synth {
    noise_px: 1.0,
    drop_fraction: 0.1,
    outlier_fraction: 0.2,
    seed: 1,
    lift: None,
};

mod fitting {
    use super::*;

    const BLACK_ROW_BIAS_PX: f64 = 8.0;

    #[test]
    fn recovers_frontal_steep_and_rotated_views() {
        let (white_keys, phase) = BOARDS[1];
        for view in [frontal(), steep(), rotated()] {
            let h = homography_for(white_keys, phase, &view);
            let peaks = synth_peaks(white_keys, phase, &h, NOISY);
            let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
            assert_eq!(fit.white_keys, white_keys);
            assert_eq!(fit.phase, phase);
            expect_close_quad(&fit.quad, &view, 4.0);
        }
    }

    #[test]
    fn keeps_a_locked_board_that_runs_partly_out_of_frame() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &partial());
        let peaks = synth_peaks(
            white_keys,
            phase,
            &h,
            Synth {
                noise_px: 1.0,
                drop_fraction: 0.05,
                outlier_fraction: 0.1,
                ..Synth::default()
            },
        );
        assert_eq!(peaks.corners.iter().filter(|c| c.is_none()).count(), 1);
        let fit = fit_keyboard(&peaks, Some(Lock { white_keys, phase }), FRAME, None);
        assert_eq!(fit.expect("a fit").white_keys, white_keys);
    }

    #[test]
    fn will_not_find_a_board_for_the_first_time_without_seeing_its_ends() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &frontal());
        let mut peaks = synth_peaks(white_keys, phase, &h, NOISY);
        peaks.corners[0] = None;
        peaks.corners[3] = None;
        assert!(fit_keyboard(&peaks, None, FRAME, None).is_none());
    }

    #[test]
    fn reaches_the_near_end_of_a_receding_board_whose_near_corners_sit_a_key_in() {
        let (white_keys, phase) = BOARDS[2];
        let h = homography_for(white_keys, phase, &receding());
        let mut peaks = synth_peaks(
            white_keys,
            phase,
            &h,
            Synth {
                noise_px: 1.0,
                ..Synth::default()
            },
        );
        let corners = keyboard_template(white_keys, phase).corners;
        let one_key_in =
            |corner: Point| scored(apply_homography(&h, corner.x + 1.0, corner.y), 0.9);
        peaks.corners[0] = Some(one_key_in(corners[0]));
        peaks.corners[3] = Some(one_key_in(corners[3]));
        let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
        assert_eq!(fit.white_keys, white_keys);
        assert_eq!(fit.phase, phase);
        let near = apply_homography(&h, 1.0, 1.0);
        let truth = receding()[3];
        let near_key_px =
            ((near.x - truth.x) * FRAME.width).hypot((near.y - truth.y) * FRAME.height);
        let found = fit.quad[3];
        let off = ((found.x - truth.x) * FRAME.width).hypot((found.y - truth.y) * FRAME.height);
        assert!(off < 0.25 * near_key_px);
    }

    #[test]
    fn identifies_every_standard_board_and_phase() {
        for (white_keys, phase) in BOARDS {
            let h = homography_for(white_keys, phase, &frontal());
            let peaks = synth_peaks(white_keys, phase, &h, NOISY);
            let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
            assert_eq!((fit.white_keys, fit.phase), (white_keys, phase));
        }
    }

    #[test]
    fn places_the_rear_from_back_gaps_when_the_front_rows_are_biased() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &steep());
        let template = keyboard_template(white_keys, phase);
        let mut peaks = synth_peaks(white_keys, phase, &h, Synth::default());
        let lock = Some(Lock { white_keys, phase });
        for black in peaks.black_low.iter_mut().chain(&mut peaks.black_high) {
            black.y += BLACK_ROW_BIAS_PX / FRAME.height;
        }
        let rear_error = |peaks: &Peaks| -> f64 {
            let fit = fit_keyboard(peaks, lock, FRAME, None).expect("a fit");
            (0..=white_keys)
                .map(|x| {
                    let at = Point {
                        x: x as f64,
                        y: 0.0,
                    };
                    let found = apply_homography(&fit.homography, at.x, at.y);
                    let truth = apply_homography(&h, at.x, at.y);
                    let key = local_key_width(&h, at) * FRAME.width;
                    ((found.x - truth.x) * FRAME.width).hypot((found.y - truth.y) * FRAME.height)
                        / key
                })
                .fold(0.0, f64::max)
        };
        let without = rear_error(&peaks);
        with_back_points(&mut peaks, &template, &h, &[0.0; 3]);
        let with = rear_error(&peaks);
        assert!(with < 0.1, "with back points {with}");
        assert!(without > 0.1, "without back points {without}");
    }

    #[test]
    fn fits_directly_under_a_lock() {
        let (white_keys, phase) = BOARDS[0];
        let h = homography_for(white_keys, phase, &frontal());
        let peaks = synth_peaks(white_keys, phase, &h, NOISY);
        let fit = fit_keyboard(&peaks, Some(Lock { white_keys, phase }), FRAME, None);
        assert_eq!(fit.expect("a fit").white_keys, white_keys);
    }

    #[test]
    fn rejects_a_quad_shape_that_is_not_a_keybed() {
        let corners = [(0.2, 0.2), (0.7, 0.2), (0.7, 0.7), (0.2, 0.7)]
            .map(|(x, y)| Some(scored(Point { x, y }, 0.9)));
        let peaks = Peaks {
            corners,
            ..Peaks::default()
        };
        assert!(fit_keyboard(&peaks, None, FRAME, None).is_none());
    }

    #[test]
    fn needs_at_least_three_corner_peaks_to_place_gaps_and_black_keys() {
        let (white_keys, phase) = BOARDS[0];
        let h = homography_for(white_keys, phase, &frontal());
        let mut peaks = synth_peaks(white_keys, phase, &h, Synth::default());
        peaks.corners[2] = None;
        peaks.corners[3] = None;
        assert!(fit_keyboard(&peaks, None, FRAME, None).is_none());
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct SteepBoard {
    size: Size,
    white_keys: usize,
    peaks: Peaks,
    label_gaps: Vec<Point>,
}

mod counting {
    use super::*;

    #[test]
    fn numbers_each_gap_from_the_low_end_across_a_hidden_one_and_past_a_stray_peak() {
        let (white_keys, phase) = BOARDS[2];
        let h = homography_for(white_keys, phase, &steep());
        let truth_of = {
            let inverse = find_homography(&steep(), &keyboard_template(white_keys, phase).corners);
            move |p: ScoredPoint| apply_homography(&inverse, p.x, p.y).x
        };
        let mut peaks = synth_peaks(white_keys, phase, &h, Synth::default());
        peaks.gaps.remove(7);
        let a = apply_homography(&h, 20.0, 1.0);
        let b = apply_homography(&h, 21.0, 1.0);
        let stray = scored(
            Point {
                x: (a.x + b.x) / 2.0,
                y: (a.y + b.y) / 2.0,
            },
            0.9,
        );
        peaks.gaps.push(stray);
        let counted = count_gaps(&peaks, None, None).expect("a count");
        for c in &counted {
            assert_ne!(c.gap, stray);
            assert_eq!(f64::from(c.index), truth_of(c.gap).round());
        }
    }

    #[test]
    fn numbers_a_real_steep_boards_keys_right_where_the_hidden_back_corners_sit_off() {
        let board: SteepBoard = serde_json::from_str(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../web/src/fixtures/steep-board-peaks.json"
        )))
        .expect("the fixture parses");
        let fit = fit_keyboard(&board.peaks, None, board.size, None).expect("a fit");
        assert_eq!(fit.white_keys, board.white_keys);
        let front = ((fit.quad[2].x - fit.quad[3].x) * board.size.width)
            .hypot((fit.quad[2].y - fit.quad[3].y) * board.size.height);
        for (i, label) in board.label_gaps.iter().enumerate() {
            let gap = apply_homography(&fit.homography, (i + 1) as f64, 1.0);
            let off =
                ((gap.x - label.x) * board.size.width).hypot((gap.y - label.y) * board.size.height);
            assert!(off / (front / board.white_keys as f64) < 0.3);
        }
    }
}

mod lifting {
    use super::*;

    #[test]
    fn fits_the_lift_from_the_front_and_back_tops_alike() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &steep());
        let truth: Lift = [0.01, -0.008, 0.001];
        let mut peaks = synth_peaks(
            white_keys,
            phase,
            &h,
            Synth {
                lift: Some(truth),
                ..Synth::default()
            },
        );
        let template = keyboard_template(white_keys, phase);
        with_back_points(&mut peaks, &template, &h, &truth);
        peaks.black_top_low.clear();
        peaks.black_top_high.clear();
        let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
        let lift = estimate_lift(&peaks, &fit, FRAME).expect("a lift");
        for bottom in &template.black_low {
            let rear = Point {
                x: bottom.x,
                y: 0.0,
            };
            let truth_at = lift_many(&h, &truth, &[rear])[0];
            let found = lifted(&fit.homography, &lift, rear);
            assert!(((found.x - truth_at.x) * FRAME.width).abs() < 0.5);
            assert!(((found.y - truth_at.y) * FRAME.height).abs() < 0.5);
        }
    }

    #[test]
    fn recovers_a_lift_that_moves_the_tops_over_half_a_key() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &steep());
        let truth: Lift = [0.01, -0.008, 0.001];
        let peaks = synth_peaks(
            white_keys,
            phase,
            &h,
            Synth {
                lift: Some(truth),
                ..Synth::default()
            },
        );
        let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
        let lift = estimate_lift(&peaks, &fit, FRAME).expect("a lift");
        let template = keyboard_template(white_keys, phase);
        let drawn: Vec<_> = key_net_faces(&fit, &lift)
            .into_iter()
            .filter(|face| face.black)
            .collect();
        for (i, top) in lift_many(&h, &truth, &template.black_low)
            .iter()
            .enumerate()
        {
            let corner = drawn[2 * i].bar[3];
            assert!(((corner.x - top.x) * FRAME.width).abs() < 0.5);
            assert!(((corner.y - top.y) * FRAME.height).abs() < 0.5);
        }
    }

    #[test]
    fn is_none_for_a_model_that_finds_no_top_corners() {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &steep());
        let peaks = synth_peaks(white_keys, phase, &h, Synth::default());
        let fit = fit_keyboard(&peaks, None, FRAME, None).expect("a fit");
        assert!(estimate_lift(&peaks, &fit, FRAME).is_none());
    }
}

mod one_euro {
    use super::*;

    #[test]
    fn passes_the_first_sample_through_untouched() {
        let mut filter = OneEuro::new(1.0, 0.02, 1.0);
        let p = Point { x: 0.5, y: 0.5 };
        assert_eq!(filter.filter(p, 0.0), p);
    }

    #[test]
    fn smooths_noisy_samples_toward_a_held_still_point() {
        let mut filter = OneEuro::new(1.0, 0.02, 1.0);
        let mut rng = Mulberry32::new(4);
        let mut last = Point { x: 1.0, y: 1.0 };
        for t in 0..60 {
            let noisy = Point {
                x: 0.02 * (rng.next_f64() - 0.5),
                y: 0.02 * (rng.next_f64() - 0.5),
            };
            last = filter.filter(noisy, f64::from(t) * 33.0);
        }
        assert!(last.x.hypot(last.y) < 0.01);
    }

    #[test]
    fn tracks_a_step_input_without_permanent_lag() {
        let mut filter = OneEuro::new(1.0, 0.02, 1.0);
        let mut out = Point::default();
        for t in 0..200 {
            out = filter.filter(Point { x: 1.0, y: 0.0 }, f64::from(t) * 16.0);
        }
        assert!((out.x - 1.0).abs() < 0.005);
    }

    #[test]
    fn forgets_history_on_reset() {
        let mut filter = OneEuro::new(1.0, 0.02, 1.0);
        let one = Point { x: 1.0, y: 1.0 };
        filter.filter(one, 0.0);
        filter.filter(one, 16.0);
        filter.reset();
        let fresh = Point { x: 0.2, y: 0.2 };
        assert_eq!(filter.filter(fresh, 1000.0), fresh);
    }
}

mod tracking {
    use super::*;

    const TRACKING_NOISE: Synth = Synth {
        noise_px: 1.0,
        drop_fraction: 0.05,
        outlier_fraction: 0.1,
        seed: 1,
        lift: None,
    };

    fn good(board: usize) -> ModelFrame {
        let (white_keys, phase) = BOARDS[board];
        let h = homography_for(white_keys, phase, &frontal());
        ModelFrame {
            presence: 0.7,
            peaks: synth_peaks(white_keys, phase, &h, TRACKING_NOISE),
        }
    }

    fn garbage() -> ModelFrame {
        ModelFrame {
            presence: 0.7,
            peaks: Peaks {
                corners: [(0.2, 0.2), (0.7, 0.2), (0.7, 0.7), (0.2, 0.7)]
                    .map(|(x, y)| Some(scored(Point { x, y }, 0.9))),
                ..Peaks::default()
            },
        }
    }

    fn acquired() -> Tracker {
        let mut tracker = Tracker::default();
        for i in 0..5 {
            tracker.update(&good(0), FRAME, f64::from(i) * 33.0);
        }
        tracker
    }

    #[test]
    fn stays_none_below_the_presence_gate() {
        let mut tracker = Tracker::default();
        for i in 0..5 {
            let frame = ModelFrame {
                presence: 0.4,
                ..good(0)
            };
            assert!(tracker.update(&frame, FRAME, f64::from(i) * 33.0).is_none());
        }
    }

    #[test]
    fn finds_the_board_by_a_vote_over_five_frames() {
        let mut tracker = Tracker::default();
        for i in 0..4 {
            assert!(tracker
                .update(&good(0), FRAME, f64::from(i) * 33.0)
                .is_none());
        }
        let fit = tracker.update(&good(0), FRAME, 132.0).expect("a fit");
        assert_eq!((fit.white_keys, fit.phase), BOARDS[0]);
    }

    #[test]
    fn a_low_presence_frame_in_the_middle_of_acquisition_resets_the_streak() {
        let mut tracker = Tracker::default();
        tracker.update(&good(0), FRAME, 0.0);
        let weak = ModelFrame {
            presence: 0.4,
            ..good(0)
        };
        tracker.update(&weak, FRAME, 33.0);
        assert!(tracker.update(&good(0), FRAME, 66.0).is_none());
    }

    #[test]
    fn holds_the_last_pose_through_a_rejected_fit() {
        let mut tracker = acquired();
        let held = tracker.update(&good(0), FRAME, 165.0).expect("a fit");
        let after_bad = tracker
            .update(&garbage(), FRAME, 198.0)
            .expect("a held fit");
        assert_eq!(after_bad.lock(), held.lock());
    }

    #[test]
    fn is_lost_after_fifteen_consecutive_bad_frames() {
        let mut tracker = acquired();
        let mut last: Option<Fit> = None;
        for i in 0..15 {
            last = tracker.update(&garbage(), FRAME, 100.0 + f64::from(i) * 33.0);
        }
        assert!(last.is_none());
        let mut reacquired = None;
        for i in 0..5 {
            reacquired = tracker.update(&good(0), FRAME, 1000.0 + f64::from(i) * 33.0);
        }
        assert!(reacquired.is_some());
    }

    #[test]
    fn moves_to_another_board_only_once_the_frames_keep_showing_it() {
        let mut tracker = acquired();
        let mut fit = None;
        for i in 0..5 {
            fit = tracker.update(&good(1), FRAME, 100.0 + f64::from(i) * 33.0);
        }
        assert_eq!(fit.as_ref().expect("a held fit").white_keys, BOARDS[0].0);
        for i in 5..50 {
            fit = tracker.update(&good(1), FRAME, 100.0 + f64::from(i) * 33.0);
        }
        let fit = fit.expect("a fit");
        assert_eq!((fit.white_keys, fit.phase), BOARDS[1]);
    }
}

mod session {
    use super::*;

    #[test]
    fn starts_in_search_with_the_whole_frame_squashed() {
        let session = Session::new(false);
        let crop = session.next_crop(FRAME);
        assert_eq!(crop.mode, Mode::Search);
        assert_eq!((crop.width, crop.height), (256, 256));
        assert!(crop.quad.is_none());
    }

    #[test]
    fn crop_matrix_sends_the_crop_centre_to_the_quad_centre() {
        let quad = frontal().map(|p| Point {
            x: p.x * FRAME.width,
            y: p.y * FRAME.height,
        });
        let crop = crop_for(&quad, 768, 160);
        let m = crop_matrix(&crop, 768, 160, FRAME);
        let centre = apply_homography(&m, 384.0, 80.0);
        assert!((centre.x * FRAME.width - crop.centre.x).abs() < 1e-6);
        assert!((centre.y * FRAME.height - crop.centre.y).abs() < 1e-6);
    }
}

mod rectified {
    use super::*;

    const WIDTH: usize = 768;
    const HEIGHT: usize = 160;
    const LEFT: f64 = 768.0 * 0.08 / 1.16;
    const TOP: f64 = 160.0 * 0.15 / 1.3;

    fn to_crop(quad: &[Point; 4]) -> Homography {
        let m = rectified_matrix(quad, WIDTH, HEIGHT, FRAME).expect("a matrix");
        invert_homography(&m).expect("invertible")
    }

    fn close(a: Point, x: f64, y: f64) -> bool {
        (a.x - x).abs() < 1e-6 && (a.y - y).abs() < 1e-6
    }

    #[test]
    fn sends_the_quad_corners_to_the_rectangle() {
        let m = to_crop(&steep());
        let wanted = [
            (LEFT, TOP),
            (768.0 - LEFT, TOP),
            (768.0 - LEFT, 160.0 - TOP),
            (LEFT, 160.0 - TOP),
        ];
        for (corner, (x, y)) in steep().iter().zip(wanted) {
            assert!(close(apply_homography(&m, corner.x, corner.y), x, y));
        }
    }

    #[test]
    fn spaces_the_front_edge_evenly_whatever_the_perspective() {
        let n = 52.0;
        let template = [
            Point { x: 0.0, y: 0.0 },
            Point { x: n, y: 0.0 },
            Point { x: n, y: 1.0 },
            Point { x: 0.0, y: 1.0 },
        ];
        let to_frame = find_homography(&template, &steep());
        let corners = template.map(|p| apply_homography(&to_frame, p.x, p.y));
        let m = to_crop(&corners);
        for i in [0.0, 7.0, 26.0, 51.0, 52.0] {
            let p = apply_homography(&to_frame, i, 1.0);
            let c = apply_homography(&m, p.x, p.y);
            assert!(close(c, LEFT + (768.0 - 2.0 * LEFT) * i / n, 160.0 - TOP));
        }
    }

    #[test]
    fn decodes_a_heatmap_peak_to_the_frame_point_under_that_crop_pixel() {
        let quad = steep();
        let m = rectified_matrix(&quad, WIDTH, HEIGHT, FRAME).expect("a matrix");
        let (cols, rows) = (WIDTH / 2, HEIGHT / 2);
        let mut heat = vec![0.0; 7 * cols * rows];
        render_gaussian(&mut heat, 4, (cols, rows), (30.0, 200.0), 1.0, 1.0);
        let peak = decode_heatmaps(&heat, WIDTH, HEIGHT, &m, None).gaps[0];
        let forward = invert_homography(&m).expect("invertible");
        let back = apply_homography(&forward, peak.x, peak.y);
        assert!(close(back, 2.0 * 200.0 + 0.5, 2.0 * 30.0 + 0.5));
    }
}

mod input {
    use super::*;

    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];

    fn gradient(width: usize, height: usize) -> Vec<u8> {
        (0..width * height)
            .flat_map(|i| {
                let (x, y) = ((i % width) as u8, (i / width) as u8);
                [x * 10, y * 10, x + y, 255]
            })
            .collect()
    }

    fn normalised(value: f32, channel: usize) -> f32 {
        (value / 255.0 - MEAN[channel]) / STD[channel]
    }

    fn run(to_frame: &Homography, width: usize, height: usize) -> Vec<f32> {
        let rgba = gradient(8, 6);
        let pixels = Pixels {
            rgba: &rgba,
            width: 8,
            height: 6,
        };
        crop_input(&pixels, to_frame, width, height)
    }

    #[test]
    fn identity_gives_the_normalised_pixels_in_planes() {
        let out = run(&IDENTITY, 8, 6);
        let plane = 8 * 6;
        assert_eq!(out.len(), 3 * plane);
        for (x, y) in [(0, 0), (3, 2), (7, 5)] {
            let at = y * 8 + x;
            let expected = [x as f32 * 10.0, y as f32 * 10.0, (x + y) as f32];
            for c in 0..3 {
                assert!((out[c * plane + at] - normalised(expected[c], c)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn an_affine_shift_and_a_half_pixel_blend_bilinearly() {
        let shift = [1.0, 0.0, 2.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let out = run(&shift, 4, 3);
        // crop (1, 1) samples frame (3.5, 2): halfway between red 30 and 40
        assert!((out[4 + 1] - normalised(35.0, 0)).abs() < 1e-5);
    }

    #[test]
    fn leaves_black_where_the_crop_leaves_the_frame() {
        let out = run(&[1.0, 0.0, -100.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 2, 2);
        assert!((out[0] - normalised(0.0, 0)).abs() < 1e-6);
    }

    #[test]
    fn a_perspective_crop_samples_where_the_homography_points() {
        let skew = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.05, 0.0, 1.0];
        let out = run(&skew, 4, 3);
        let at = apply_homography(&skew, 3.0, 1.0);
        let expected = at.x * 10.0;
        assert!((out[4 + 3] - normalised(expected as f32, 0)).abs() < 1e-4);
    }

    #[test]
    fn a_shrinking_crop_samples_its_pixel_centre_like_the_training_warp() {
        let shrink = [2.0, 0.0, 0.5, 0.0, 2.0, 0.5, 0.0, 0.0, 1.0];
        let out = run(&shrink, 2, 2);
        // crop (0, 0) lands on frame (0.5, 0.5), halfway between red 0 and 10
        assert!((out[0] - normalised(5.0, 0)).abs() < 1e-5);
    }
}

mod lens_bend {
    use super::*;
    use crate::lens::{estimate_bend, moved_peaks, straighten_peaks, Lens};

    fn seen_through(lens: &Lens, view: [Point; 4]) -> (Peaks, Lock) {
        let (white_keys, phase) = BOARDS[1];
        let h = homography_for(white_keys, phase, &view);
        let straight = synth_peaks(
            white_keys,
            phase,
            &h,
            Synth {
                noise_px: 0.5,
                ..Synth::default()
            },
        );
        (
            moved_peaks(&straight, |p| lens.bend(p)),
            Lock { white_keys, phase },
        )
    }

    /// The bend after a few looks, each fitting the board under the bend the last one found,
    /// the way a session settles on its camera's lens.
    fn bend_found(truth: Lens) -> f64 {
        let (seen, lock) = seen_through(&truth, steep());
        let mut lens = Lens { k: 0.0, ..truth };
        for _ in 0..4 {
            let fit = fit_keyboard(&straighten_peaks(&seen, &lens), Some(lock), FRAME, None)
                .expect("a fit under the lock");
            lens.k = estimate_bend(&seen, &fit, &lens).expect("a bend");
        }
        lens.k
    }

    #[test]
    fn finds_the_bend_of_a_wide_lens_from_the_keys() {
        let k = bend_found(Lens {
            k: -0.15,
            aspect: FRAME.width / FRAME.height,
        });
        assert!((k + 0.15).abs() < 0.02, "{k}");
    }

    #[test]
    fn finds_no_bend_through_a_straight_lens() {
        let k = bend_found(Lens::straight(FRAME));
        assert!(k.abs() < 0.02, "{k}");
    }
}
