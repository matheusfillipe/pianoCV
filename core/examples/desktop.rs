use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::time::Instant;

use keycore::engine::{Engine, Keys};
use keycore::geom::Point;
use serde::Serialize;

const MODEL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/public/keynet.onnx");
const WHITE: [u8; 3] = [60, 220, 90];
const BLACK: [u8; 3] = [240, 70, 70];

#[derive(Serialize)]
struct Line<'a> {
    frame: usize,
    timestamp_ms: f64,
    process_ms: f64,
    model_ms: f64,
    #[serde(flatten)]
    keys: &'a Keys,
}

fn probe(video: &Path) -> Result<(usize, usize, f64), Box<dyn Error>> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "stream=width,height,avg_frame_rate"])
        .args(["-of", "csv=p=0"])
        .arg(video)
        .output()?;
    if !out.status.success() {
        return Err("ffprobe could not read the video".into());
    }
    let text = String::from_utf8(out.stdout)?;
    let mut fields = text.trim().split(',');
    let mut next = || fields.next().ok_or("ffprobe gave no video stream");
    let width = next()?.parse()?;
    let height = next()?.parse()?;
    let (num, den) = next()?.split_once('/').ok_or("bad frame rate")?;
    let fps = num.parse::<f64>()? / den.parse::<f64>()?;
    Ok((
        width,
        height,
        if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        },
    ))
}

fn decoder(video: &Path) -> std::io::Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-noautorotate", "-i"])
        .arg(video)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(Stdio::piped())
        .spawn()
}

fn encoder(out: &Path, width: usize, height: usize, fps: f64) -> std::io::Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba"])
        .args([
            "-s",
            &format!("{width}x{height}"),
            "-r",
            &fps.to_string(),
            "-i",
            "-",
        ])
        .args([
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(out)
        .stdin(Stdio::piped())
        .spawn()
}

fn line(rgba: &mut [u8], width: usize, height: usize, from: Point, to: Point, colour: [u8; 3]) {
    let reach = 4.0 * width.max(height) as f64;
    if [from.x, from.y, to.x, to.y]
        .iter()
        .any(|v| !v.is_finite() || v.abs() > reach)
    {
        return;
    }
    let steps = (to.x - from.x)
        .abs()
        .max((to.y - from.y).abs())
        .ceil()
        .max(1.0) as usize;
    for step in 0..=steps {
        let t = step as f64 / steps as f64;
        let x = (from.x + (to.x - from.x) * t).round();
        let y = (from.y + (to.y - from.y) * t).round();
        if x >= 0.0 && y >= 0.0 && (x as usize) < width && (y as usize) < height {
            let at = (y as usize * width + x as usize) * 4;
            rgba[at..at + 3].copy_from_slice(&colour);
        }
    }
}

fn draw(rgba: &mut [u8], width: usize, height: usize, keys: &Keys) {
    for key in &keys.keys {
        let colour = if key.black { BLACK } else { WHITE };
        for (i, from) in key.outline.iter().enumerate() {
            let to = key.outline[(i + 1) % key.outline.len()];
            line(rgba, width, height, *from, to, colour);
        }
    }
}

fn percentile(sorted: &[f64], share: f64) -> f64 {
    sorted
        .get(((sorted.len() as f64 * share).ceil() as usize).saturating_sub(1))
        .copied()
        .unwrap_or(0.0)
}

fn summary(name: &str, mut times: Vec<f64>) {
    times.sort_by(f64::total_cmp);
    println!(
        "{name}: median {:.1} ms, p95 {:.1} ms",
        percentile(&times, 0.5),
        percentile(&times, 0.95)
    );
}

fn read_frame(reader: &mut impl Read, frame: &mut [u8]) -> io::Result<bool> {
    let mut filled = 0;
    while filled < frame.len() {
        match reader.read(&mut frame[filled..]) {
            Ok(0) if filled == 0 => return Ok(false),
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => filled += n,
            Err(e) if e.kind() == ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn transcode(
    engine: &mut Engine,
    input: &mut Child,
    output: &mut Child,
    out_dir: &Path,
    (width, height, fps): (usize, usize, f64),
) -> Result<(), Box<dyn Error>> {
    let mut frames_in = input.stdout.take().ok_or("no decoder output")?;
    let mut frames_out: ChildStdin = output.stdin.take().ok_or("no encoder input")?;
    let mut lines = BufWriter::new(File::create(out_dir.join("keys.jsonl"))?);

    let mut rgba = vec![0; width * height * 4];
    let (mut process_ms, mut model_ms, mut tracked) = (Vec::new(), Vec::new(), 0);
    while read_frame(&mut frames_in, &mut rgba)? {
        let frame = process_ms.len();
        let timestamp_ms = frame as f64 * 1000.0 / fps;
        let started = Instant::now();
        let (keys, model_time) = engine.process(&rgba, width, height, timestamp_ms)?;
        let took = started.elapsed().as_secs_f64() * 1000.0;
        let model = model_time.as_secs_f64() * 1000.0;
        serde_json::to_writer(
            &mut lines,
            &Line {
                frame,
                timestamp_ms,
                process_ms: took,
                model_ms: model,
                keys: &keys,
            },
        )?;
        writeln!(lines)?;
        draw(&mut rgba, width, height, &keys);
        frames_out.write_all(&rgba)?;
        tracked += usize::from(keys.tracked);
        process_ms.push(took);
        model_ms.push(model);
    }
    drop(frames_out);
    lines.flush()?;
    if !input.wait()?.success() {
        return Err("ffmpeg could not decode the input video".into());
    }
    if !output.wait()?.success() {
        return Err("ffmpeg could not encode the output video".into());
    }
    println!(
        "{} frames, tracked {:.0}%",
        process_ms.len(),
        100.0 * tracked as f64 / process_ms.len().max(1) as f64
    );
    summary("engine.process", process_ms);
    summary("model alone", model_ms);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    let [_, video, out_dir] = args.as_slice() else {
        return Err("usage: desktop <video> <out-dir>".into());
    };
    let video = PathBuf::from(video);
    let out_dir = PathBuf::from(out_dir);
    fs::create_dir_all(&out_dir)?;
    let (width, height, fps) = probe(&video)?;
    let mut engine = Engine::new(MODEL)?;
    let mut input = decoder(&video)?;
    let mut output = match encoder(&out_dir.join("keys.mp4"), width, height, fps) {
        Ok(output) => output,
        Err(e) => {
            input.kill()?;
            return Err(e.into());
        }
    };
    let result = transcode(
        &mut engine,
        &mut input,
        &mut output,
        &out_dir,
        (width, height, fps),
    );
    if result.is_err() {
        input.kill()?;
        output.kill()?;
    }
    result
}
