use serde::{Deserialize, Serialize};

use crate::decode::Peaks;
use crate::fit::{fit_candidates, fit_keyboard, Fit, Lift, Lock};
use crate::geom::{distance, find_homography, is_finite_homography, Point, Size};
use crate::keys::{keyboard_template, BLACK_KEY_DEPTH};
use crate::lift::{estimate_lift, lift_apart_keys, LIFT_JUMPS_TO_MOVE, LIFT_JUMP_KEYS};

fn low_pass(value: f64, previous: f64, alpha: f64) -> f64 {
    alpha * value + (1.0 - alpha) * previous
}

fn euro_alpha(cutoff: f64, dt: f64) -> f64 {
    let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

/// A standard One Euro filter over 2D points: a low-pass whose cutoff rises with the point's own
/// speed, so it holds still when the tracked point is still and reacts fast when it moves.
#[derive(Clone, Debug)]
pub struct OneEuro {
    min_cutoff: f64,
    beta: f64,
    d_cutoff: f64,
    last: Option<(Point, f64)>,
    last_derivative: Point,
}

impl OneEuro {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            last: None,
            last_derivative: Point::default(),
        }
    }

    pub fn reset(&mut self) {
        self.last = None;
        self.last_derivative = Point::default();
    }

    pub fn filter(&mut self, point: Point, timestamp_ms: f64) -> Point {
        let Some((last_point, last_time)) = self.last else {
            self.last = Some((point, timestamp_ms));
            return point;
        };
        let dt = ((timestamp_ms - last_time) / 1000.0).max(1e-3);
        let dx = (point.x - last_point.x) / dt;
        let dy = (point.y - last_point.y) / dt;
        let d_alpha = euro_alpha(self.d_cutoff, dt);
        let d_hat = Point {
            x: low_pass(dx, self.last_derivative.x, d_alpha),
            y: low_pass(dy, self.last_derivative.y, d_alpha),
        };
        let cutoff = self.min_cutoff + self.beta * d_hat.x.hypot(d_hat.y);
        let alpha = euro_alpha(cutoff, dt);
        let filtered = Point {
            x: low_pass(point.x, last_point.x, alpha),
            y: low_pass(point.y, last_point.y, alpha),
        };
        self.last = Some((filtered, timestamp_ms));
        self.last_derivative = d_hat;
        filtered
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OneEuroConfig {
    pub min_cutoff: f64,
    pub beta: f64,
    pub d_cutoff: f64,
}

impl Default for OneEuroConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.02,
            d_cutoff: 1.0,
        }
    }
}

/// The lift's cutoff in hertz, about three seconds' average. Points high above the keys move by the
/// lift times their height, so we hold it far steadier than the corners, and the camera stays put
/// while someone plays.
const LIFT_CUTOFF_HZ: f64 = 0.05;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFrame {
    pub presence: f64,
    pub peaks: Peaks,
}

const PRESENCE_THRESHOLD: f64 = 0.6;
// a board is found by a vote over this many frames' fits, since one frame's end corners can sit a
// key off and number every key wrong
const ACQUIRE_FRAMES: u32 = 5;
const LOST_FRAMES: u32 = 15;
// contrary checks, each CONTRARY_EVERY frames apart
const RELOCK_CHECKS: u32 = 3;
const RELOCK_MARGIN: f64 = 0.1;
// a fresh identification tries every board at several numberings, too slow for every frame, and
// the board under a still camera does not change between checks
const CONTRARY_EVERY: u32 = 15;
const VOTE_SAME_PLACE_KEYS: f64 = 0.5;

struct Tracking {
    lock: Lock,
    bad_frames: u32,
    contrary_streak: u32,
    contrary_candidate: Option<Lock>,
    since_contrary_check: u32,
    held_fit: Option<Fit>,
    /// How much of the keybed this board's black keys take, as measured so far.
    black_depth: f64,
}

enum State {
    Searching { voting_frames: u32, votes: Vec<Fit> },
    Tracking(Box<Tracking>),
}

fn searching() -> State {
    State::Searching {
        voting_frames: 0,
        votes: Vec::new(),
    }
}

/// The fit most frames agree on: fits of the same board whose front-low corners lie within half a
/// key of each other count as one placement, and the placement with the most support wins.
fn elect_fit(votes: &[Fit], frame: Size) -> Option<Fit> {
    struct Cluster<'a> {
        anchor: Point,
        fits: Vec<&'a Fit>,
        support: f64,
    }
    let mut clusters: Vec<Cluster> = Vec::new();
    for fit in votes {
        let anchor = Point {
            x: fit.quad[3].x * frame.width,
            y: fit.quad[3].y * frame.height,
        };
        let key_width = ((fit.quad[2].x - fit.quad[3].x) * frame.width)
            .hypot((fit.quad[2].y - fit.quad[3].y) * frame.height)
            / fit.white_keys as f64;
        let home = clusters.iter_mut().find(|cluster| {
            cluster.fits[0].lock() == fit.lock()
                && distance(cluster.anchor, anchor) < key_width * VOTE_SAME_PLACE_KEYS
        });
        match home {
            None => clusters.push(Cluster {
                anchor,
                fits: vec![fit],
                support: fit.explained,
            }),
            Some(home) => {
                home.fits.push(fit);
                home.support += fit.explained;
            }
        }
    }
    let winner = clusters
        .iter()
        .fold(None, |best: Option<&Cluster>, cluster| match best {
            Some(b) if cluster.support <= b.support => Some(b),
            _ => Some(cluster),
        })?;
    winner
        .fits
        .iter()
        .copied()
        .fold(None, |best: Option<&Fit>, fit| match best {
            Some(b) if fit.explained <= b.explained => Some(b),
            _ => Some(fit),
        })
        .cloned()
}

fn register_contrary(state: &mut Tracking, alt_fit: Option<&Fit>, locked_fit: Option<&Fit>) {
    // another board counts against the lock only where it explains clearly more of the keys
    let alt = alt_fit
        .filter(|alt| {
            locked_fit.is_none_or(|locked| alt.explained >= locked.explained + RELOCK_MARGIN)
        })
        .map(Fit::lock);
    let Some(alt) = alt.filter(|alt| *alt != state.lock) else {
        state.contrary_streak = 0;
        state.contrary_candidate = None;
        return;
    };
    if state.contrary_candidate == Some(alt) {
        state.contrary_streak += 1;
    } else {
        state.contrary_candidate = Some(alt);
        state.contrary_streak = 1;
    }
    if state.contrary_streak >= RELOCK_CHECKS {
        state.lock = alt;
        state.contrary_streak = 0;
        state.contrary_candidate = None;
    }
}

/// The four corners' filters, then the lift's first two terms and its third, and the lift the
/// tracker last accepted.
struct Smoother {
    filters: Vec<OneEuro>,
    last_lift: Option<Lift>,
    lift_jumps: u32,
}

impl Smoother {
    fn new(config: OneEuroConfig) -> Self {
        Self {
            filters: [
                vec![OneEuro::new(config.min_cutoff, config.beta, config.d_cutoff); 4],
                vec![OneEuro::new(LIFT_CUTOFF_HZ, 0.0, config.d_cutoff); 2],
            ]
            .concat(),
            last_lift: None,
            lift_jumps: 0,
        }
    }

    fn reset(&mut self) {
        for filter in &mut self.filters {
            filter.reset();
        }
        self.last_lift = None;
        self.lift_jumps = 0;
    }

    /// The keys are drawn through the homography, so it is rebuilt from the smoothed corners or
    /// the drawing would carry every frame's raw noise; the lift is smoothed the same way, and a
    /// frame whose top corners do not agree keeps the last one.
    fn smooth(&mut self, fit: &Fit, peaks: &Peaks, frame: Size, timestamp_ms: f64) -> Fit {
        let mut quad = fit.quad;
        for (i, corner) in quad.iter_mut().enumerate() {
            *corner = self.filters[i].filter(*corner, timestamp_ms);
        }
        let homography = find_homography(
            &keyboard_template(fit.white_keys, fit.phase, fit.black_depth).corners,
            &quad,
        );
        let held = Fit {
            quad,
            homography: if is_finite_homography(&homography) {
                homography
            } else {
                fit.homography
            },
            ..fit.clone()
        };
        let lift = estimate_lift(peaks, &held, frame);
        // the lift follows the camera slowly, so one far from the last is a bad frame's reading,
        // unless the frames keep disagreeing with it, when the last one was the bad reading
        let jumped = match (&lift, &self.last_lift) {
            (Some(lift), Some(last)) => lift_apart_keys(&held, lift, last, frame) >= LIFT_JUMP_KEYS,
            _ => false,
        };
        self.lift_jumps = if jumped { self.lift_jumps + 1 } else { 0 };
        if jumped && self.lift_jumps >= LIFT_JUMPS_TO_MOVE {
            self.filters[4].reset();
            self.filters[5].reset();
            self.lift_jumps = 0;
        }
        if let Some(lift) = lift {
            if !jumped || self.lift_jumps == 0 {
                let uv = self.filters[4].filter(
                    Point {
                        x: lift[0],
                        y: lift[1],
                    },
                    timestamp_ms,
                );
                let s = self.filters[5].filter(Point { x: lift[2], y: 0.0 }, timestamp_ms);
                self.last_lift = Some([uv.x, uv.y, s.x]);
            }
        }
        Fit {
            lift: self.last_lift,
            ..held
        }
    }
}

/// Search until presence gates at 0.6 for 5 voting frames, then track: fit under the locked board
/// every frame, holding the last smoothed pose on a rejected fit, lost (back to search) after 15
/// bad frames. Board size and phase lock at acquisition and only move once a different board
/// fits consistently for 3 checks running; a bad frame simply holds, it never counts as contrary
/// evidence on its own.
pub struct Tracker {
    state: State,
    smoother: Smoother,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new(OneEuroConfig::default())
    }
}

impl Tracker {
    pub fn new(config: OneEuroConfig) -> Self {
        Self {
            state: searching(),
            smoother: Smoother::new(config),
        }
    }

    pub fn reset(&mut self) {
        self.state = searching();
        self.smoother.reset();
    }

    /// Moves the tracked board's black-key depth `share` of the way to `reading`. A board being
    /// searched for is fitted at a typical board's depth, and each board found starts there.
    pub fn follow_black_depth(&mut self, reading: f64, share: f64) {
        if let State::Tracking(state) = &mut self.state {
            state.black_depth += share * (reading - state.black_depth);
        }
    }

    pub fn update(&mut self, frame: &ModelFrame, size: Size, timestamp_ms: f64) -> Option<Fit> {
        match &mut self.state {
            State::Searching {
                voting_frames,
                votes,
            } => {
                if frame.presence < PRESENCE_THRESHOLD {
                    *voting_frames = 0;
                    votes.clear();
                    return None;
                }
                let candidates = fit_candidates(&frame.peaks, None, size, None, BLACK_KEY_DEPTH);
                if candidates.is_empty() {
                    return None;
                }
                votes.extend(candidates);
                *voting_frames += 1;
                if *voting_frames < ACQUIRE_FRAMES {
                    return None;
                }
                let fit = elect_fit(votes, size)?;
                let held = self.smoother.smooth(&fit, &frame.peaks, size, timestamp_ms);
                self.state = State::Tracking(Box::new(Tracking {
                    lock: fit.lock(),
                    bad_frames: 0,
                    contrary_streak: 0,
                    contrary_candidate: None,
                    since_contrary_check: 0,
                    held_fit: Some(held.clone()),
                    black_depth: BLACK_KEY_DEPTH,
                }));
                Some(held)
            }
            State::Tracking(state) => {
                if frame.presence >= PRESENCE_THRESHOLD {
                    let locked = fit_keyboard(
                        &frame.peaks,
                        Some(state.lock),
                        size,
                        state.held_fit.as_ref(),
                        state.black_depth,
                    );
                    match &locked {
                        Some(fit) => {
                            state.bad_frames = 0;
                            state.held_fit =
                                Some(self.smoother.smooth(fit, &frame.peaks, size, timestamp_ms));
                        }
                        None => state.bad_frames += 1,
                    }
                    state.since_contrary_check += 1;
                    if state.since_contrary_check >= CONTRARY_EVERY {
                        state.since_contrary_check = 0;
                        let alt = fit_keyboard(&frame.peaks, None, size, None, state.black_depth);
                        register_contrary(state, alt.as_ref(), locked.as_ref());
                    }
                } else {
                    state.bad_frames += 1;
                }
                if state.bad_frames >= LOST_FRAMES {
                    self.reset();
                    return None;
                }
                state.held_fit.clone()
            }
        }
    }
}
