use serde::{Deserialize, Serialize};

use crate::geom::Point;

/// How much of the keybed's depth a black key takes, from the far edge where the black keys start.
pub const BLACK_KEY_DEPTH: f64 = 0.62;

const WHITE_PC_INDEX: [i32; 12] = [0, -1, 1, -1, 2, 3, -1, 4, -1, 5, -1, 6];
const BLACK_OFFSETS: [f64; 12] = [
    -1.0, 0.6, -1.0, 1.75, -1.0, -1.0, 3.6, -1.0, 4.63, -1.0, 5.66, -1.0,
];
const BLACK_WIDTH: f64 = 0.58;
const TEMPLATE_OCTAVE: i32 = 60;

/// The letter of a board's lowest white key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    C,
    D,
    E,
    F,
    G,
    A,
    B,
}

impl Phase {
    fn pitch_class(self) -> i32 {
        match self {
            Phase::C => 0,
            Phase::D => 2,
            Phase::E => 4,
            Phase::F => 5,
            Phase::G => 7,
            Phase::A => 9,
            Phase::B => 11,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Board {
    pub phase: Phase,
    pub white_keys: usize,
}

pub const STANDARD_BOARDS: [Board; 4] = [
    Board {
        phase: Phase::C,
        white_keys: 29,
    },
    Board {
        phase: Phase::C,
        white_keys: 36,
    },
    Board {
        phase: Phase::E,
        white_keys: 45,
    },
    Board {
        phase: Phase::A,
        white_keys: 52,
    },
];

pub fn is_black(pitch: i32) -> bool {
    BLACK_OFFSETS[pitch.rem_euclid(12) as usize] >= 0.0
}

pub fn white_index(pitch: i32) -> i32 {
    7 * pitch.div_euclid(12) - 12 + WHITE_PC_INDEX[pitch.rem_euclid(12) as usize]
}

pub struct KeyUnits {
    pub from: f64,
    pub to: f64,
}

/// Where a key sits on any keyboard, in white-key widths from the origin `white_index` counts from.
pub fn key_units(pitch: i32) -> KeyUnits {
    let offset = BLACK_OFFSETS[pitch.rem_euclid(12) as usize];
    if offset >= 0.0 {
        let from = f64::from(7 * pitch.div_euclid(12) - 12) + offset;
        return KeyUnits {
            from,
            to: from + BLACK_WIDTH,
        };
    }
    let from = f64::from(white_index(pitch));
    KeyUnits {
        from,
        to: from + 1.0,
    }
}

/// The lowest MIDI pitch of a standard board, or `None` for any other size or starting letter.
pub fn lowest_pitch(white_keys: usize, phase: Phase) -> Option<i32> {
    match (white_keys, phase) {
        (29 | 36, Phase::C) => Some(36),
        (45, Phase::E) => Some(28),
        (52, Phase::A) => Some(21),
        _ => None,
    }
}

fn first_white_pitch(phase: Phase) -> i32 {
    TEMPLATE_OCTAVE + phase.pitch_class()
}

/// The keyboard on the keybed plane, in white-key units: x runs 0 to `white_keys` along the board,
/// y runs 0 (back, under the black keys) to 1 (the player's edge).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KeyboardTemplate {
    /// Back-low, back-high, front-high, front-low.
    pub corners: [Point; 4],
    pub gaps: Vec<Point>,
    pub back_gaps: Vec<Point>,
    pub black_low: Vec<Point>,
    pub black_high: Vec<Point>,
}

pub fn keyboard_template(white_keys: usize, phase: Phase) -> KeyboardTemplate {
    let first = first_white_pitch(phase);
    let base = f64::from(white_index(first));
    let width = white_keys as f64;
    let corners = [
        Point { x: 0.0, y: 0.0 },
        Point { x: width, y: 0.0 },
        Point { x: width, y: 1.0 },
        Point { x: 0.0, y: 1.0 },
    ];
    let gaps = (1..white_keys)
        .map(|i| Point {
            x: i as f64,
            y: 1.0,
        })
        .collect();
    let back_gaps = (1..white_keys)
        .map(|i| Point {
            x: i as f64,
            y: 0.0,
        })
        .collect();
    let mut black_low = Vec::new();
    let mut black_high = Vec::new();
    let mut pitch = first;
    for i in 0..white_keys {
        while is_black(pitch) {
            pitch += 1;
        }
        if i + 1 < white_keys && is_black(pitch + 1) {
            let units = key_units(pitch + 1);
            black_low.push(Point {
                x: units.from - base,
                y: BLACK_KEY_DEPTH,
            });
            black_high.push(Point {
                x: units.to - base,
                y: BLACK_KEY_DEPTH,
            });
        }
        pitch += 1;
    }
    KeyboardTemplate {
        corners,
        gaps,
        back_gaps,
        black_low,
        black_high,
    }
}

pub struct BoardKey {
    pub black: bool,
    pub semitone: i32,
    pub from: f64,
    pub to: f64,
    pub depth: f64,
}

pub fn board_keys(white_keys: usize, phase: Phase) -> Vec<BoardKey> {
    let first = first_white_pitch(phase);
    let base = f64::from(white_index(first));
    let mut keys = Vec::new();
    let mut pitch = first;
    for i in 0..white_keys {
        while is_black(pitch) {
            pitch += 1;
        }
        keys.push(BoardKey {
            black: false,
            semitone: pitch - first,
            from: i as f64,
            to: (i + 1) as f64,
            depth: 1.0,
        });
        if i + 1 < white_keys && is_black(pitch + 1) {
            let units = key_units(pitch + 1);
            keys.push(BoardKey {
                black: true,
                semitone: pitch + 1 - first,
                from: units.from - base,
                to: units.to - base,
                depth: BLACK_KEY_DEPTH,
            });
        }
        pitch += 1;
    }
    keys
}
