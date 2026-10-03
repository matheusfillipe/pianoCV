use crate::geom::Homography;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

pub struct Pixels<'a> {
    pub rgba: &'a [u8],
    pub width: usize,
    pub height: usize,
}

impl Pixels<'_> {
    fn rgb(&self, x: isize, y: isize) -> [f32; 3] {
        if x < 0 || y < 0 || x >= self.width as isize || y >= self.height as isize {
            return [0.0; 3];
        }
        let at = (y as usize * self.width + x as usize) * 4;
        let p = &self.rgba[at..at + 3];
        [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])]
    }

    fn bilinear(&self, x: f32, y: f32) -> [f32; 3] {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (x0, y0) = (x0 as isize, y0 as isize);
        let (a, b) = (self.rgb(x0, y0), self.rgb(x0 + 1, y0));
        let (c, d) = (self.rgb(x0, y0 + 1), self.rgb(x0 + 1, y0 + 1));
        let mut out = [0.0; 3];
        for i in 0..3 {
            let top = a[i] + (b[i] - a[i]) * fx;
            let bottom = c[i] + (d[i] - c[i]) * fx;
            out[i] = top + (bottom - top) * fy;
        }
        out
    }
}

/// The model input for a crop: `to_frame` maps a crop pixel to a frame pixel (both with whole
/// numbers on pixel centres, any 3x3 so perspective crops work). The result is ImageNet
/// normalised NCHW float32 with a black border where the crop leaves the frame. We take one
/// bilinear sample per crop pixel because training warps its crops the same way.
pub fn crop_input(frame: &Pixels, to_frame: &Homography, width: usize, height: usize) -> Vec<f32> {
    let plane = width * height;
    let mut out = vec![0.0; 3 * plane];
    let h = to_frame;
    for row in 0..height {
        let y = row as f64;
        let (mut nx, mut ny, mut d) = (h[1] * y + h[2], h[4] * y + h[5], h[7] * y + h[8]);
        for column in 0..width {
            let sample = frame.bilinear((nx / d) as f32, (ny / d) as f32);
            let at = row * width + column;
            for c in 0..3 {
                out[c * plane + at] = (sample[c] / 255.0 - MEAN[c]) / STD[c];
            }
            nx += h[0];
            ny += h[3];
            d += h[6];
        }
    }
    out
}
