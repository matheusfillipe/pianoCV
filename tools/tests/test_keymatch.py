import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.keymatch import (
    PerturbConfig,
    SidecarKey,
    SynthKeysFrame,
    load_synth_keys,
    parse_synth_keys_sidecar,
    perturb_quad,
)

FAR_LEFT, FAR_RIGHT = (64.0, 96.0), (576.0, 96.0)
NEAR_LEFT, NEAR_RIGHT = (64.0, 384.0), (576.0, 384.0)
QUAD_PX = np.array([FAR_LEFT, FAR_RIGHT, NEAR_RIGHT, NEAR_LEFT])
BOARD_WIDTH = 3.0


def _face(u0: float, u1: float, v0: float, v1: float) -> np.ndarray:
    def point(u: float, v: float) -> tuple[float, float]:
        x = FAR_LEFT[0] + (u / BOARD_WIDTH) * (FAR_RIGHT[0] - FAR_LEFT[0])
        y = FAR_LEFT[1] + v * (NEAR_LEFT[1] - FAR_LEFT[1])
        return x, y

    return np.array([point(u0, v0), point(u1, v0), point(u1, v1), point(u0, v1)])


def _three_white_two_black_frame() -> SynthKeysFrame:
    white = [
        SidecarKey(
            pitch=60 + 2 * i, black=False, top=_face(float(i), float(i + 1), 0.0, 1.0), front=None
        )
        for i in range(3)
    ]
    black = [
        SidecarKey(
            pitch=pitch,
            black=True,
            top=_face(boundary - 0.3, boundary + 0.3, 0.05, 0.35),
            front=_face(boundary - 0.3, boundary + 0.3, 0.35, 0.42),
        )
        for pitch, boundary in ((61, 1.0), (63, 2.0))
    ]
    return SynthKeysFrame(
        image_path=Path("unused.png"),
        corners_px=QUAD_PX,
        keys=[*white, *black],
        image_size=(640, 480),
    )


def test_load_synth_keys_round_trips_a_sidecar(tmp_path: Path) -> None:
    frame = _three_white_two_black_frame()
    assert frame.corners_px is not None
    width, height = 640, 480
    scale = np.array([float(width), float(height)])

    def face_fraction(face_px: np.ndarray) -> list[dict[str, float]]:
        return [{"x": float(x), "y": float(y)} for x, y in (face_px / scale)]

    sidecar = {
        "corners": face_fraction(frame.corners_px),
        "imageWidth": width,
        "imageHeight": height,
        "board": {"keys": 5, "whiteKeys": 3, "lowestPitch": 60, "lowestNote": "C4"},
        "keys": [
            {
                "pitch": key.pitch,
                "black": key.black,
                "top": face_fraction(key.top),
                "front": None if key.front is None else face_fraction(key.front),
            }
            for key in frame.keys
        ],
    }
    cv2.imwrite(str(tmp_path / "a.png"), np.zeros((height, width, 3), dtype=np.uint8))
    (tmp_path / "a.json").write_text(json.dumps(sidecar))

    loaded = load_synth_keys(tmp_path)

    assert len(loaded) == 1
    assert loaded[0].corners_px is not None
    assert np.allclose(loaded[0].corners_px, frame.corners_px, atol=1e-3)
    assert sorted(key.pitch for key in loaded[0].keys) == sorted(key.pitch for key in frame.keys)


def test_perturb_quad_keeps_a_valid_non_degenerate_quad() -> None:
    rng = np.random.default_rng(0)
    original_area = float(cv2.contourArea(QUAD_PX.astype(np.float32)))
    for _ in range(500):
        perturbed = perturb_quad(QUAD_PX, rng)
        area = float(cv2.contourArea(perturbed.astype(np.float32)))
        assert area > 0.1 * original_area


def test_perturb_quad_leaves_the_configured_fraction_exactly_unperturbed() -> None:
    rng = np.random.default_rng(1)
    config = PerturbConfig(unperturbed_fraction=0.15)
    exact = sum(np.array_equal(perturb_quad(QUAD_PX, rng, config), QUAD_PX) for _ in range(4000))
    assert abs(exact / 4000 - 0.15) < 0.03


def test_perturb_quad_always_unperturbed_returns_the_input_unchanged() -> None:
    rng = np.random.default_rng(2)
    config = PerturbConfig(unperturbed_fraction=1.0)
    for _ in range(20):
        assert np.array_equal(perturb_quad(QUAD_PX, rng, config), QUAD_PX)


_FACE = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}, {"x": 1.0, "y": 1.0}, {"x": 0.0, "y": 1.0}]
_KEY = {"pitch": 60, "black": False, "top": _FACE, "front": None}


@pytest.mark.parametrize(
    ("sidecar", "message"),
    [
        ([], "not a json object"),
        ({"imageWidth": "w", "imageHeight": 1, "keys": []}, "invalid image dimensions"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": "x"}, "must list keys"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": [_KEY]}, "no corners"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": [], "ignoreMask": 3}, "ignoreMask"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": [], "sequence": 3}, "sequence"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": [], "frameIndex": "x"}, "frameIndex"),
        ({"imageWidth": 1, "imageHeight": 1, "keys": [], "cornerVisible": [True]}, "cornerVisible"),
        (
            {"imageWidth": 1, "imageHeight": 1, "corners": _FACE[:3], "keys": [_KEY]},
            "exactly 4 corners",
        ),
        (
            {"imageWidth": 1, "imageHeight": 1, "corners": _FACE, "keys": [[]]},
            "key entry must be an object",
        ),
        (
            {"imageWidth": 1, "imageHeight": 1, "corners": _FACE, "keys": [{**_KEY, "pitch": "x"}]},
            "invalid pitch",
        ),
        (
            {"imageWidth": 1, "imageHeight": 1, "corners": [1, 2, 3, 4], "keys": []},
            "is not an object",
        ),
        (
            {
                "imageWidth": 1,
                "imageHeight": 1,
                "corners": [{"x": "a", "y": 0}, *_FACE[1:]],
                "keys": [],
            },
            "invalid coordinates",
        ),
    ],
)
def test_a_malformed_sidecar_is_rejected(tmp_path: Path, sidecar: object, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(sidecar))

    with pytest.raises(ValueError, match=message):
        parse_synth_keys_sidecar(path)


def test_a_sidecar_keeps_its_optional_fields(tmp_path: Path) -> None:
    path = tmp_path / "ok.json"
    path.write_text(
        json.dumps(
            {
                "imageWidth": 10,
                "imageHeight": 10,
                "corners": _FACE,
                "keys": [_KEY],
                "ignoreMask": "ok.ignore.png",
                "sequence": "clip",
                "frameIndex": 3,
                "cornerVisible": [True, True, False, True],
                "kind": "real-keys",
            }
        )
    )

    frame = parse_synth_keys_sidecar(path)

    assert (frame.sequence, frame.frame_index, frame.real) == ("clip", 3, True)
    assert frame.corner_visible == (True, True, False, True)
    assert frame.ignore_mask == tmp_path / "ok.ignore.png"
