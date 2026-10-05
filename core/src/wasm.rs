use serde::de::DeserializeOwned;
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::decode::{self, Peaks};
use crate::fit::{self, Fit, Lift, Lock};
use crate::geom::{self, Correspondence, Homography, Size};
use crate::input::{self, Pixels};
use crate::keys::{self, Phase};
use crate::lift;
use crate::session::Session;
use crate::track::ModelFrame;

fn parse<T: DeserializeOwned>(json: &str) -> Result<T, JsError> {
    serde_json::from_str(json).map_err(|e| JsError::new(&e.to_string()))
}

fn print<T: Serialize>(value: &T) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))
}

fn matrix(values: &[f64]) -> Result<Homography, JsError> {
    values
        .try_into()
        .map_err(|_| JsError::new("a matrix has 9 numbers"))
}

fn lock(white_keys: Option<u32>, phase: Option<String>) -> Result<Option<Lock>, JsError> {
    match (white_keys, phase) {
        (Some(white_keys), Some(phase)) => Ok(Some(Lock {
            white_keys: white_keys as usize,
            phase: parse::<Phase>(&format!("\"{phase}\""))?,
        })),
        _ => Ok(None),
    }
}

/// Decodes a `[C, H/2, W/2]` heatmap into peaks JSON. `to_frame` is the 9 numbers of the model
/// input pixel to frame fraction matrix.
#[wasm_bindgen]
pub fn decode_heatmaps(
    heat: &[f32],
    width: u32,
    height: u32,
    to_frame: &[f64],
    offsets: Option<Vec<f32>>,
) -> Result<String, JsError> {
    let peaks = decode::decode_heatmaps(
        heat,
        width as usize,
        height as usize,
        &matrix(to_frame)?,
        offsets.as_deref(),
    );
    print(&peaks)
}

/// The numbered gaps as JSON, or `null`.
#[wasm_bindgen]
pub fn count_gaps(peaks_json: &str, white_keys: Option<u32>) -> Result<String, JsError> {
    let peaks: Peaks = parse(peaks_json)?;
    print(&fit::count_gaps(
        &peaks,
        white_keys.map(|k| k as usize),
        None,
    ))
}

/// The best fit as JSON, or `null`. A lock needs both `white_keys` and `phase`.
#[wasm_bindgen]
pub fn fit_keyboard(
    peaks_json: &str,
    frame_width: f64,
    frame_height: f64,
    white_keys: Option<u32>,
    phase: Option<String>,
) -> Result<String, JsError> {
    let peaks: Peaks = parse(peaks_json)?;
    let frame = Size {
        width: frame_width,
        height: frame_height,
    };
    print(&fit::fit_keyboard(
        &peaks,
        lock(white_keys, phase)?,
        frame,
        None,
    ))
}

/// The black keys' lift as JSON, or `null`.
#[wasm_bindgen]
pub fn estimate_lift(
    peaks_json: &str,
    fit_json: &str,
    frame_width: f64,
    frame_height: f64,
) -> Result<String, JsError> {
    let peaks: Peaks = parse(peaks_json)?;
    let fit: Fit = parse(fit_json)?;
    let frame = Size {
        width: frame_width,
        height: frame_height,
    };
    print(&lift::estimate_lift(&peaks, &fit, frame))
}

/// The per-key outlines to draw, as JSON, in frame fractions.
#[wasm_bindgen]
pub fn key_net_faces(fit_json: &str, lift: &[f64]) -> Result<String, JsError> {
    let fit: Fit = parse(fit_json)?;
    let lift: Lift = lift
        .try_into()
        .map_err(|_| JsError::new("a lift has 3 numbers"))?;
    print(&lift::key_net_faces(&fit, &lift))
}

/// Every key of the fitted board once, low to high, as JSON: `{black, semitone, note, bar, top}`
/// in frame fractions.
#[wasm_bindgen]
pub fn key_outlines(fit_json: &str) -> Result<String, JsError> {
    print(&lift::key_outlines(&parse(fit_json)?))
}

/// Template points `[{x, y}]` mapped onto the black keys' top plane, as JSON in frame fractions.
#[wasm_bindgen]
pub fn lift_points(homography: &[f64], lift: &[f64], points_json: &str) -> Result<String, JsError> {
    let homography = matrix(homography)?;
    let lift: Lift = lift
        .try_into()
        .map_err(|_| JsError::new("a lift has 3 numbers"))?;
    let points: Vec<geom::Point> = parse(points_json)?;
    let lifted: Vec<geom::Point> = points
        .into_iter()
        .map(|p| lift::lifted(&homography, &lift, p))
        .collect();
    print(&lifted)
}

/// The template for a board as JSON, in white-key units.
#[wasm_bindgen]
pub fn keyboard_template(white_keys: u32, phase: &str) -> Result<String, JsError> {
    print(&keys::keyboard_template(
        white_keys as usize,
        parse::<Phase>(&format!("\"{phase}\""))?,
    ))
}

/// The least-squares homography over `[{src, dst}]` pairs as 9 numbers, or `null`.
#[wasm_bindgen]
pub fn refine_homography(pairs_json: &str) -> Result<String, JsError> {
    let pairs: Vec<Correspondence> = parse(pairs_json)?;
    print(&geom::refine_homography(&pairs))
}

/// The model input for a crop as planar float32, from `rgba` pixels of `pixel_width` by
/// `pixel_height` and the crop pixel to frame fraction matrix that `next_crop` gives.
#[wasm_bindgen]
pub fn prepare_input(
    rgba: &[u8],
    pixel_width: u32,
    pixel_height: u32,
    to_frame: &[f64],
    width: u32,
    height: u32,
) -> Result<Vec<f32>, JsError> {
    let to_pixels = input::fractions_to_pixels(
        &matrix(to_frame)?,
        f64::from(pixel_width),
        f64::from(pixel_height),
    );
    let (pixel_width, pixel_height) = (pixel_width as usize, pixel_height as usize);
    if rgba.len() != pixel_width * pixel_height * 4 {
        return Err(JsError::new("the pixels are not width by height RGBA"));
    }
    Ok(input::crop_input(
        &Pixels {
            rgba,
            width: pixel_width,
            height: pixel_height,
        },
        &to_pixels,
        width as usize,
        height as usize,
    ))
}

#[wasm_bindgen]
pub struct KeyNetSession(Session);

#[wasm_bindgen]
impl KeyNetSession {
    #[wasm_bindgen(constructor)]
    pub fn new(rectified: bool) -> Self {
        Self(Session::new(rectified))
    }

    pub fn reset(&mut self) {
        self.0 = Session::new(self.0.rectified());
    }

    /// The crop to run the model on next, as JSON: mode, input size and the pixel to frame matrix.
    pub fn next_crop(&self, frame_width: f64, frame_height: f64) -> Result<String, JsError> {
        print(&self.0.next_crop(Size {
            width: frame_width,
            height: frame_height,
        }))
    }

    /// Moves the session on with the model's outputs and returns `{fit, acquired}` as JSON.
    pub fn step(
        &mut self,
        presence: f64,
        peaks_json: &str,
        frame_width: f64,
        frame_height: f64,
        now_ms: f64,
    ) -> Result<String, JsError> {
        let frame = ModelFrame {
            presence,
            peaks: parse(peaks_json)?,
        };
        let size = Size {
            width: frame_width,
            height: frame_height,
        };
        print(&self.0.step(&frame, size, now_ms))
    }

    /// The keyboard's space under the current fit, as JSON, or `null` until a lift is seen.
    pub fn space(&self) -> Result<String, JsError> {
        print(&self.0.space())
    }
}
