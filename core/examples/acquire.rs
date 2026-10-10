mod video;

use std::env;
use std::error::Error;
use std::io::Read;
use std::path::Path;

use keycore::engine::Engine;
use keycore::geom::is_mirrored_quad;
use video::{decoder, probe};

/// Runs each video through a fresh engine every `restart every` frames, so every video gives many
/// first acquisitions, and counts the fits read end for end. A real camera keeps a board's handedness,
/// so on real footage every mirrored fit is a board read backwards.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [model, every, videos @ ..] = &args[..] else {
        return Err("usage: acquire <model> <restart every n frames> <video>...".into());
    };
    let every: usize = every.parse()?;
    let (mut all_fitted, mut all_mirrored) = (0, 0);
    for video in videos {
        let (width, height, fps) = match probe(Path::new(video)) {
            Ok(size) => size,
            Err(error) => {
                println!("{video}: {error}");
                continue;
            }
        };
        let mut input = decoder(Path::new(video))?;
        let mut frames = input.stdout.take().ok_or("ffmpeg gave no output")?;
        let mut rgba = vec![0u8; width * height * 4];
        let mut engine = Engine::new(model)?;
        let (mut seen, mut fitted, mut mirrored, mut acquisitions, mut backwards) = (0, 0, 0, 0, 0);
        let mut was_fitted = false;
        while frames.read_exact(&mut rgba).is_ok() {
            if seen > 0 && seen % every == 0 {
                engine = Engine::new(model)?;
                was_fitted = false;
            }
            engine.process(&rgba, width, height, seen as f64 * 1000.0 / fps)?;
            seen += 1;
            let Some(fit) = engine.fit() else {
                was_fitted = false;
                continue;
            };
            let is_mirrored = is_mirrored_quad(&fit.quad);
            fitted += 1;
            mirrored += usize::from(is_mirrored);
            if !was_fitted {
                acquisitions += 1;
                backwards += usize::from(is_mirrored);
            }
            was_fitted = true;
        }
        input.wait()?;
        all_fitted += fitted;
        all_mirrored += mirrored;
        println!(
            "{video}: {seen} frames, {fitted} fitted, {acquisitions} acquisitions, {backwards} end for end, {mirrored} frames end for end"
        );
    }
    println!("all: {all_fitted} fitted, {all_mirrored} frames end for end");
    Ok(())
}
