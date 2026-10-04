use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use ort::session::Session as Model;
use ort::value::Tensor;
use serde::Serialize;

use crate::decode::decode_heatmaps;
use crate::fit::Fit;
use crate::geom::{Point, Size};
use crate::input::{crop_input, fractions_to_pixels, Pixels};
use crate::keys::{lowest_pitch, Phase};
use crate::lift::{key_hull, key_net_faces, Face};
use crate::session::Session;
use crate::track::ModelFrame;

const HEATMAP_CHANNELS: usize = 12;

#[derive(Debug)]
pub enum Error {
    Model(ort::Error),
    Pixels,
    Output(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::Model(e) => write!(f, "model: {e}"),
            Error::Pixels => write!(f, "the pixels are not width by height RGBA"),
            Error::Output(what) => write!(f, "the model's {what} output has an unexpected shape"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Model(e)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Board {
    pub white_keys: usize,
    pub lowest_phase: Phase,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Key {
    /// MIDI note number, absent when the board is not a standard size.
    pub note: Option<i32>,
    pub black: bool,
    /// The key's polygon in frame pixels.
    pub outline: Vec<Point>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Keys {
    pub tracked: bool,
    pub board: Option<Board>,
    pub confidence: f64,
    pub keys: Vec<Key>,
}

pub struct Engine {
    model: Model,
    session: Session,
}

fn keys_of(fit: &Fit, size: Size) -> Vec<Key> {
    let to_pixels = |bar: &[Point]| -> Vec<Point> {
        bar.iter()
            .map(|p| Point {
                x: p.x * size.width,
                y: p.y * size.height,
            })
            .collect()
    };
    let lift = fit.lift.unwrap_or([0.0; 3]);
    let faces = key_net_faces(fit, &lift);
    let lowest = lowest_pitch(fit.white_keys, fit.phase);
    let mut keys = Vec::new();
    let mut faces = faces.iter().peekable();
    while let Some(Face {
        black,
        semitone,
        bar,
    }) = faces.next()
    {
        let outline = match faces.next_if(|f| *black && f.black && f.semitone == *semitone) {
            Some(footprint) => key_hull(bar, &footprint.bar),
            None => bar.to_vec(),
        };
        keys.push(Key {
            note: lowest.map(|lowest| lowest + semitone),
            black: *black,
            outline: to_pixels(&outline),
        });
    }
    keys
}

impl Engine {
    pub fn new(model_path: impl AsRef<Path>) -> Result<Self, Error> {
        Ok(Self {
            model: Model::builder()?.commit_from_file(model_path)?,
            session: Session::new(false),
        })
    }

    /// Runs the model on one RGBA frame and moves the tracker on. Also gives how long the model
    /// alone took.
    pub fn process(
        &mut self,
        rgba: &[u8],
        width: usize,
        height: usize,
        timestamp_ms: f64,
    ) -> Result<(Keys, Duration), Error> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return Err(Error::Pixels);
        }
        let size = Size {
            width: width as f64,
            height: height as f64,
        };
        let crop = self.session.next_crop(size);
        let to_pixels = fractions_to_pixels(&crop.matrix, size.width, size.height);
        let input = crop_input(
            &Pixels {
                rgba,
                width,
                height,
            },
            &to_pixels,
            crop.width,
            crop.height,
        );
        let started = Instant::now();
        let (heat, presence) = self.run_model(input, crop.width, crop.height)?;
        let model_time = started.elapsed();
        let peaks = decode_heatmaps(&heat, crop.width, crop.height, &crop.matrix, None);
        let step = self
            .session
            .step(&ModelFrame { presence, peaks }, size, timestamp_ms);
        let keys = match step.fit {
            Some(fit) => Keys {
                tracked: true,
                board: Some(Board {
                    white_keys: fit.white_keys,
                    lowest_phase: fit.phase,
                }),
                confidence: presence,
                keys: keys_of(&fit, size),
            },
            None => Keys {
                tracked: false,
                board: None,
                confidence: presence,
                keys: Vec::new(),
            },
        };
        Ok((keys, model_time))
    }

    /// The heatmaps and presence the model gives for one input.
    fn run_model(
        &mut self,
        input: Vec<f32>,
        width: usize,
        height: usize,
    ) -> Result<(Vec<f32>, f64), Error> {
        let image = Tensor::from_array(([1, 3, height, width], input))?;
        let outputs = self.model.run(ort::inputs!["image" => image])?;
        let (_, heat) = outputs["heatmaps"].try_extract_tensor::<f32>()?;
        let (_, presence) = outputs["presence"].try_extract_tensor::<f32>()?;
        if heat.len() != HEATMAP_CHANNELS * (height / 2) * (width / 2) {
            return Err(Error::Output("heatmaps"));
        }
        let presence = *presence.first().ok_or(Error::Output("presence"))?;
        Ok((heat.to_vec(), f64::from(presence)))
    }
}

#[cfg(test)]
mod tests {
    use std::env;

    use super::*;

    const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/keynet.onnx");

    fn engine() -> Option<Engine> {
        if !Path::new(MODEL).exists() {
            assert!(
                env::var_os("CI").is_none(),
                "web/public/keynet.onnx is absent in CI"
            );
            eprintln!("skipping: web/public/keynet.onnx is absent, run make model");
            return None;
        }
        Some(Engine::new(MODEL).expect("the model loads"))
    }

    #[test]
    fn a_missing_model_is_an_error() {
        assert!(matches!(Engine::new("nope.onnx"), Err(Error::Model(_))));
    }

    #[test]
    fn bad_pixels_are_an_error() {
        let Some(mut engine) = engine() else { return };
        assert!(matches!(
            engine.process(&[0; 10], 4, 4, 0.0),
            Err(Error::Pixels)
        ));
        assert!(matches!(engine.process(&[], 0, 0, 0.0), Err(Error::Pixels)));
    }

    #[test]
    fn a_blank_frame_tracks_nothing() {
        let Some(mut engine) = engine() else { return };
        let (keys, _) = engine
            .process(&vec![128; 640 * 360 * 4], 640, 360, 0.0)
            .unwrap();
        assert!(!keys.tracked);
        assert!(keys.keys.is_empty());
    }
}
