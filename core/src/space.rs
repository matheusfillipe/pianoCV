use serde::Serialize;

use crate::fit::{Fit, Lift};
use crate::geom::Point;

/// White-key widths from the keybed's far edge to the player's edge: a white key is about 150 mm
/// long and 23.5 mm wide.
const KEYBED_DEPTH_KEYS: f64 = 6.4;
/// White-key widths a black key's top stands above the white keys: about 12.5 mm on a 23.5 mm key.
const BLACK_RISE_KEYS: f64 = 0.53;
/// Closer to the camera than this share of the keybed's own distance, a point is on the lens or
/// behind it and has no place in the picture.
const NEAREST_SHARE: f64 = 0.1;

/// A point in the keyboard's own space, in white-key widths: `x` along the board from its first
/// key, `depth` from the keybed's far edge towards the player, `height` up off the white keys.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SpacePoint {
    pub x: f64,
    pub depth: f64,
    pub height: f64,
}

/// The keyboard's space as the camera sees it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Space {
    /// Maps `[x, depth, height, 1]` onto the frame in fractions, row by row.
    pub projection: [f64; 12],
    /// Where the camera stands, or None for a view so far off that its rays are parallel.
    pub camera: Option<SpacePoint>,
    pub keybed_depth: f64,
    /// The projection's last row at the middle of the keybed, which is how far away the keybed is.
    pub keybed_distance: f64,
    pub nearest_share: f64,
}

impl Space {
    pub fn new(fit: &Fit, lift: &Lift) -> Self {
        let h = fit.homography;
        let projection = [
            h[0],
            h[1] / KEYBED_DEPTH_KEYS,
            lift[0] / BLACK_RISE_KEYS,
            h[2],
            h[3],
            h[4] / KEYBED_DEPTH_KEYS,
            lift[1] / BLACK_RISE_KEYS,
            h[5],
            h[6],
            h[7] / KEYBED_DEPTH_KEYS,
            lift[2] / BLACK_RISE_KEYS,
            h[8],
        ];
        let middle = SpacePoint {
            x: fit.white_keys as f64 / 2.0,
            depth: KEYBED_DEPTH_KEYS / 2.0,
            height: 0.0,
        };
        Self {
            projection,
            camera: camera_of(&projection),
            keybed_depth: KEYBED_DEPTH_KEYS,
            keybed_distance: homogeneous(&projection, middle)[2],
            nearest_share: NEAREST_SHARE,
        }
    }

    /// Where a point of the keyboard's space lands in the frame, in fractions, or None where it
    /// sits on the lens or behind it.
    pub fn project(&self, p: SpacePoint) -> Option<Point> {
        let [u, v, s] = homogeneous(&self.projection, p);
        (s / self.keybed_distance >= self.nearest_share).then(|| Point { x: u / s, y: v / s })
    }
}

fn homogeneous(m: &[f64; 12], p: SpacePoint) -> [f64; 3] {
    let row =
        |r: usize| m[4 * r] * p.x + m[4 * r + 1] * p.depth + m[4 * r + 2] * p.height + m[4 * r + 3];
    [row(0), row(1), row(2)]
}

fn det3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
        + c[0] * (a[1] * b[2] - a[2] * b[1])
}

/// The camera's centre is the one point the projection sends nowhere, the null vector of its
/// three rows, which we read off as the signed minors of its columns.
fn camera_of(m: &[f64; 12]) -> Option<SpacePoint> {
    let column = |c: usize| [m[c], m[4 + c], m[8 + c]];
    let minor = |skip: usize| -> f64 {
        let kept: Vec<[f64; 3]> = (0..4).filter(|&c| c != skip).map(column).collect();
        det3(kept[0], kept[1], kept[2])
    };
    let null = [minor(0), -minor(1), minor(2), -minor(3)];
    let scale = null.iter().map(|v| v.abs()).fold(0.0, f64::max);
    (null[3].abs() > scale * 1e-9).then(|| SpacePoint {
        x: null[0] / null[3],
        depth: null[1] / null[3],
        height: null[2] / null[3],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::Phase;

    /// A pinhole camera at `eye`, looking at the middle of a 52-key keybed, with the projection
    /// split into the homography of the keybed plane and the lift of a black key's top.
    fn fit_seen_from(eye: SpacePoint) -> Fit {
        let target = SpacePoint {
            x: 26.0,
            depth: KEYBED_DEPTH_KEYS / 2.0,
            height: 0.0,
        };
        let forward = normalise([
            target.x - eye.x,
            target.depth - eye.depth,
            target.height - eye.height,
        ]);
        let right = normalise(cross(forward, [0.0, 0.0, 1.0]));
        let down = cross(forward, right);
        let focal = 1.2;
        let row = |axis: [f64; 3], centre: f64| -> [f64; 4] {
            let r = [
                focal * axis[0] + centre * forward[0],
                focal * axis[1] + centre * forward[1],
                focal * axis[2] + centre * forward[2],
            ];
            [
                r[0],
                r[1],
                r[2],
                -(r[0] * eye.x + r[1] * eye.depth + r[2] * eye.height),
            ]
        };
        let rows = [row(right, 0.5), row(down, 0.5), {
            let f = forward;
            [
                f[0],
                f[1],
                f[2],
                -(f[0] * eye.x + f[1] * eye.depth + f[2] * eye.height),
            ]
        }];
        let scale = rows[2][3];
        let homography = std::array::from_fn(|i| {
            let r = rows[i / 3];
            let v = match i % 3 {
                0 => r[0],
                1 => r[1] * KEYBED_DEPTH_KEYS,
                _ => r[3],
            };
            v / scale
        });
        let lift = std::array::from_fn(|i| rows[i][2] * BLACK_RISE_KEYS / scale);
        Fit {
            homography,
            quad: [Point { x: 0.0, y: 0.0 }; 4],
            white_keys: 52,
            phase: Phase::A,
            inlier_share: 1.0,
            explained: 1.0,
            both_ends: true,
            beyond_ends: 0,
            gap_spacing: None,
            reprojection_error: 0.0,
            lift: Some(lift),
        }
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn normalise(v: [f64; 3]) -> [f64; 3] {
        let length = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.map(|x| x / length)
    }

    #[test]
    fn finds_the_camera_where_it_stands() {
        let eye = SpacePoint {
            x: 20.0,
            depth: 30.0,
            height: 25.0,
        };
        let fit = fit_seen_from(eye);
        let camera = Space::new(&fit, &fit.lift.unwrap()).camera.unwrap();
        assert!((camera.x - eye.x).abs() < 1e-6, "{camera:?}");
        assert!((camera.depth - eye.depth).abs() < 1e-6, "{camera:?}");
        assert!((camera.height - eye.height).abs() < 1e-6, "{camera:?}");
    }

    #[test]
    fn hides_a_point_behind_the_camera_and_keeps_one_in_front() {
        let eye = SpacePoint {
            x: 26.0,
            depth: 30.0,
            height: 25.0,
        };
        let fit = fit_seen_from(eye);
        let space = Space::new(&fit, &fit.lift.unwrap());
        let above_the_keys = SpacePoint {
            x: 26.0,
            depth: 0.0,
            height: 10.0,
        };
        let behind_the_camera = SpacePoint {
            x: 26.0,
            depth: 60.0,
            height: 50.0,
        };
        assert!(space.project(above_the_keys).is_some());
        assert!(space.project(behind_the_camera).is_none());
    }
}
