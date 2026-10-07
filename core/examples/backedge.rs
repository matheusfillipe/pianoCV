use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use keycore::engine::Engine;
use keycore::geom::{apply_homography, invert_homography, Point};
use serde_json::Value;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/keynet.onnx");
/// Model runs per frame, about what a 30 fps camera gives the browser between two frames.
const STEPS_PER_FRAME: usize = 3;

/// A labelled back point, kept only when it sits inside the picture.
fn inside(p: Point) -> Option<Point> {
    ((0.02..0.98).contains(&p.x) && (0.02..0.98).contains(&p.y)).then_some(p)
}

/// The labelled key backs of one frame in frame fractions: a hand-corrected frame lists its back
/// gaps in pixels, a synthetic one outlines every key with its back edge first.
fn labelled_backs(label: &Value, width: f64, height: f64) -> Vec<Point> {
    let at = |v: &Value| v.as_f64().unwrap_or(f64::NAN);
    if let Some(gaps) = label["backGaps"].as_array() {
        return gaps
            .iter()
            .filter_map(Value::as_array)
            .filter_map(|g| {
                inside(Point {
                    x: at(&g[0]) / width,
                    y: at(&g[1]) / height,
                })
            })
            .collect();
    }
    label["keys"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|key| key["black"] == false)
        .filter_map(|key| {
            let top = key["top"].as_array()?;
            inside(Point {
                x: (at(&top[0]["x"]) + at(&top[1]["x"])) / 2.0,
                y: (at(&top[0]["y"]) + at(&top[1]["y"])) / 2.0,
            })
        })
        .collect()
}

fn frame_pixels(path: &Path) -> Result<(Vec<u8>, usize, usize), Box<dyn Error>> {
    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
        ])
        .arg(path)
        .output()?;
    let text = String::from_utf8(probe.stdout)?;
    let (width, height) = text.trim().split_once(',').ok_or("ffprobe gave no size")?;
    let rgba = Command::new("ffmpeg")
        .args(["-v", "error", "-i"])
        .arg(path)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .output()?
        .stdout;
    Ok((rgba, width.parse()?, height.parse()?))
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted
        .get(((sorted.len() as f64 * share) as usize).min(sorted.len().saturating_sub(1)))
        .copied()
        .unwrap_or(f64::NAN)
}

fn report(name: &str, mut ahead: Vec<f64>) {
    ahead.sort_by(f64::total_cmp);
    let mut off: Vec<f64> = ahead.iter().map(|v| v.abs()).collect();
    off.sort_by(f64::total_cmp);
    println!(
        "{name:>10}: {:>6} key backs, ahead median {:>5.2} px, off median {:>5.2} px, p90 {:>5.2} px",
        ahead.len(),
        percentile(&ahead, 0.5),
        percentile(&off, 0.5),
        percentile(&off, 0.9),
    );
}

/// How far the fitted back edge lands from labelled key backs, in frame pixels, positive where it
/// sits in front of them and cuts into the keys. Frames run in order per recording or sequence, the
/// way a camera feeds them.
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let Some(labels) = args.get(1).map(PathBuf::from) else {
        return Err("usage: backedge <labels-dir> [images-dir]".into());
    };
    let images = args.get(2).map_or_else(|| labels.clone(), PathBuf::from);
    let mut runs: BTreeMap<String, Vec<PathBuf>> = BTreeMap::new();
    for entry in fs::read_dir(&labels)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "json") {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let run = stem
                .rsplit_once('-')
                .map_or(stem.to_string(), |(run, _)| run.to_string());
            runs.entry(run).or_default().push(path);
        }
    }
    let (mut frames, mut fitted) = (0, 0);
    let mut by_case: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for paths in runs.values_mut() {
        paths.sort();
        let mut engine = Engine::new(MODEL)?;
        for path in paths.iter() {
            let label: Value = serde_json::from_str(&fs::read_to_string(path)?)?;
            if label["piano"] == false {
                continue;
            }
            let image = images.join(path.with_extension("png").file_name().unwrap_or_default());
            let (rgba, width, height) = frame_pixels(&image)?;
            for step in 0..STEPS_PER_FRAME {
                let now = (frames * STEPS_PER_FRAME + step) as f64 * 33.0;
                engine.process(&rgba, width, height, now)?;
            }
            frames += 1;
            let Some(fit) = engine.fit() else { continue };
            let Some(inverse) = invert_homography(&fit.homography) else {
                continue;
            };
            fitted += 1;
            let case = label["caseColor"].as_str().unwrap_or("unnamed").to_string();
            for back in labelled_backs(&label, width as f64, height as f64) {
                let straight = fit.lens.straighten(back);
                let on = apply_homography(&inverse, straight.x, straight.y);
                if on.x < 0.0 || on.x > fit.white_keys as f64 || on.y.abs() > 0.5 {
                    continue;
                }
                let edge = fit.lens.bend(apply_homography(&fit.homography, on.x, 0.0));
                let apart =
                    ((edge.x - back.x) * width as f64).hypot((edge.y - back.y) * height as f64);
                by_case
                    .entry(case.clone())
                    .or_default()
                    .push(-on.y.signum() * apart);
            }
        }
    }
    println!("{} frames, {fitted} fitted", frames);
    report("all", by_case.values().flatten().copied().collect());
    for (case, ahead) in by_case {
        report(&case, ahead);
    }
    Ok(())
}
