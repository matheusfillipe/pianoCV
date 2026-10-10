use std::error::Error;
use std::path::Path;
use std::process::{Child, Command, Stdio};

/// A video's frame width, height and frame rate.
pub fn probe(video: &Path) -> Result<(usize, usize, f64), Box<dyn Error>> {
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

/// An ffmpeg child streaming the video's frames as raw RGBA on its stdout.
pub fn decoder(video: &Path) -> std::io::Result<Child> {
    Command::new("ffmpeg")
        .args(["-v", "error", "-noautorotate", "-i"])
        .arg(video)
        .args(["-f", "rawvideo", "-pix_fmt", "rgba", "-"])
        .stdout(Stdio::piped())
        .spawn()
}
