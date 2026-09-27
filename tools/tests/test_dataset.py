import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.dataset import (
    canonical_quad,
    extract,
    load_frames,
    orient_quad,
    parse_sidecar,
    scan_recordings,
)


def _write_sidecar(
    path: Path, kind: str, width: int, height: int, session: str | None = None
) -> None:
    sidecar: dict[str, object] = {
        "kind": kind,
        "startedAt": 0,
        "durationMs": 0,
        "corners": [
            {"x": 0.25, "y": 0.4},
            {"x": 0.75, "y": 0.4},
            {"x": 0.8, "y": 0.6},
            {"x": 0.2, "y": 0.6},
        ],
        "imageWidth": width,
        "imageHeight": height,
        "mimeType": "image/png" if kind in ("snap", "auto") else "video/webm",
    }
    if session is not None:
        sidecar["session"] = session
    path.write_text(json.dumps(sidecar))


def _keybed_image(corners: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    image = np.full((height, width, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [corners.astype(np.int32)], (235, 235, 235))
    back = np.array(
        [
            corners[0],
            corners[1],
            (corners[1] + corners[2]) / 2.0,
            (corners[0] + corners[3]) / 2.0,
        ]
    )
    cv2.fillPoly(image, [back.astype(np.int32)], (20, 20, 20))
    return image


SNAP_CORNERS = np.array([[80.0, 96.0], [240.0, 96.0], [256.0, 144.0], [64.0, 144.0]])
GEMINI_CORNERS = np.array([[32.0, 48.0], [224.0, 48.0], [256.0, 192.0], [64.0, 216.0]])


def _write_snap(path: Path) -> None:
    cv2.imwrite(str(path), _keybed_image(SNAP_CORNERS))


def _write_clip(path: Path, frames: int) -> None:
    avi = path.with_suffix(".avi")
    writer = cv2.VideoWriter(str(avi), cv2.VideoWriter.fourcc(*"MJPG"), 30, (320, 240))
    assert writer.isOpened()
    image = _keybed_image(SNAP_CORNERS)
    for _ in range(frames):
        writer.write(image)
    writer.release()
    avi.replace(path)


def _make_recordings(recordings_dir: Path) -> None:
    recordings_dir.mkdir(parents=True)
    _write_snap(recordings_dir / "snap-test.png")
    _write_sidecar(recordings_dir / "snap-test.json", "snap", 320, 240)
    _write_clip(recordings_dir / "rec-test.webm", 90)
    _write_sidecar(recordings_dir / "rec-test.json", "rec", 320, 240)


def test_extract_samples_snap_and_clip(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    frames = extract(recordings_dir, frames_dir)
    assert [f.image_path.name for f in frames if f.kind == "snap"] == ["snap-test.png"]
    rec_frames = [f for f in frames if f.kind == "rec"]
    assert len(rec_frames) == 45
    assert rec_frames[0].image_path.name == "rec-test.000000.png"
    snap = frames[0] if frames[0].kind == "snap" else frames[-1]
    assert snap.corners_px is not None
    assert np.allclose(snap.corners_px, SNAP_CORNERS)
    labels = json.loads((frames_dir / "labels.json").read_text())
    assert set(labels["extracted"]) == {"snap-test", "rec-test"}
    assert (frames_dir / "snap-test.png").is_file()


def test_extract_is_idempotent(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    first = extract(recordings_dir, frames_dir)
    (recordings_dir / "rec-test.webm").unlink()
    second = extract(recordings_dir, frames_dir)
    assert [f.image_path.name for f in second] == [f.image_path.name for f in first]
    assert load_frames(frames_dir)[0].image_path == first[0].image_path


def test_scan_recordings_skips_sidecar_less_media(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "snap-a.png")
    (recordings_dir / "snap-a.json").write_text("{}")
    _write_snap(recordings_dir / "snap-b.png")
    (recordings_dir / "stray.txt").write_text("nope")
    recordings = scan_recordings(recordings_dir)
    assert [r.stem for r in recordings] == ["snap-a"]
    assert recordings[0].kind == "snap"
    assert recordings[0].media_path.name == "snap-a.png"


def test_extract_skips_unreadable_snapshot(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "snap-bad.png").write_text("not an image")
    _write_sidecar(recordings_dir / "snap-bad.json", "snap", 320, 240)
    frames = extract(recordings_dir, tmp_path / "frames")
    assert frames == []
    assert json.loads((tmp_path / "frames" / "labels.json").read_text()) == {
        "extracted": {},
        "frames": {},
    }


def test_extract_skips_unopenable_clip(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "rec-bad.webm").write_text("not a video")
    _write_sidecar(recordings_dir / "rec-bad.json", "rec", 320, 240)
    frames = extract(recordings_dir, tmp_path / "frames")
    assert frames == []
    assert json.loads((tmp_path / "frames" / "labels.json").read_text()) == {
        "extracted": {},
        "frames": {},
    }


def test_extract_persists_valid_recordings_when_one_is_unreadable(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "snap-good.png")
    _write_sidecar(recordings_dir / "snap-good.json", "snap", 320, 240)
    (recordings_dir / "rec-bad.webm").write_text("not a video")
    _write_sidecar(recordings_dir / "rec-bad.json", "rec", 320, 240)

    frames = extract(recordings_dir, frames_dir)

    assert [frame.source_stem for frame in frames] == ["snap-good"]
    labels = json.loads((frames_dir / "labels.json").read_text())
    assert labels["extracted"] == {"snap-good": 1}
    assert set(labels["frames"]) == {"snap-good.png"}


def test_extract_skips_invalid_sidecars(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "snap-x.png")
    sidecar = recordings_dir / "snap-x.json"
    cases = [
        "[]",
        json.dumps({"corners": [], "imageWidth": 1, "imageHeight": 1}),
        json.dumps({"kind": "snap", "corners": [], "imageWidth": 1, "imageHeight": 1}),
        json.dumps(
            {"kind": "snap", "corners": [{}, {}, {}, {}], "imageWidth": 1, "imageHeight": 1}
        ),
        json.dumps(
            {
                "kind": "snap",
                "corners": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": "a", "y": 1},
                ],
                "imageWidth": 1,
                "imageHeight": 1,
            }
        ),
        json.dumps(
            {
                "kind": "auto",
                "session": 5,
                "corners": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": 0, "y": 1},
                ],
                "imageWidth": 1,
                "imageHeight": 1,
            }
        ),
    ]
    for content in cases:
        sidecar.write_text(content)
        assert extract(recordings_dir, frames_dir) == []
        assert json.loads((frames_dir / "labels.json").read_text()) == {
            "extracted": {},
            "frames": {},
        }


def _write_gemini_scene(gemini_dir: Path, stem: str, with_kind: bool) -> None:
    gemini_dir.mkdir(parents=True, exist_ok=True)
    image = _keybed_image(GEMINI_CORNERS)
    for suffix in ("-orig.png", "-mask.png", "-check.png"):
        cv2.imwrite(str(gemini_dir / f"{stem}{suffix}"), image)
    sidecar: dict[str, object] = {
        "startedAt": 0,
        "durationMs": 0,
        "corners": [
            {"x": 0.1, "y": 0.2},
            {"x": 0.7, "y": 0.2},
            {"x": 0.8, "y": 0.8},
            {"x": 0.2, "y": 0.9},
        ],
        "imageWidth": 320,
        "imageHeight": 240,
        "mimeType": "image/png",
    }
    if with_kind:
        sidecar["kind"] = "snap"
    (gemini_dir / f"{stem}.json").write_text(json.dumps(sidecar))


def test_extract_gemini_scenes_into_frames(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    gemini_dir = tmp_path / "gemini"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_gemini_scene(gemini_dir, "00", with_kind=False)
    _write_gemini_scene(gemini_dir, "01", with_kind=True)
    (gemini_dir / "02-orig.png").write_text("no sidecar")
    frames = extract(recordings_dir, frames_dir, gemini_dir)
    gemini_frames = [f for f in frames if f.kind == "gemini"]
    assert [f.image_path.name for f in gemini_frames] == ["00-orig.png", "01-orig.png"]
    assert [f.source_stem for f in gemini_frames] == ["00", "01"]
    assert gemini_frames[0].corners_px is not None
    assert np.allclose(gemini_frames[0].corners_px, GEMINI_CORNERS)
    again = extract(recordings_dir, frames_dir, gemini_dir)
    assert [f.image_path.name for f in again] == [f.image_path.name for f in frames]
    assert (frames_dir / "00-orig.png").is_file()


def test_extract_without_gemini_dir_leaves_frames_unchanged(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    _make_recordings(recordings_dir)
    with_gemini = extract(recordings_dir, frames_dir, tmp_path / "missing")
    without_gemini = extract(recordings_dir, tmp_path / "frames-2")
    assert [f.kind for f in with_gemini] == [f.kind for f in without_gemini]
    assert all(f.kind != "gemini" for f in without_gemini)


def test_load_frames_ignores_malformed_labels(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_snap(frames_dir / "snap-keep.png")
    labels = {
        "extracted": {"ok": 1, 5: "no", "bad": "no"},
        "frames": {
            "snap-keep.png": {
                "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "source_stem": "snap-keep",
                "kind": "snap",
            },
            "missing.png": {"corners_px": "junk", "source_stem": "m", "kind": "snap"},
            "no-fields.png": {"corners_px": None},
            "bad-corners.png": {
                "corners_px": [[0], [1], [2], [3]],
                "source_stem": "b",
                "kind": "snap",
            },
            "bad-point.png": {
                "corners_px": [[0, 0], [1, 0], [1, "x"], [0, 1]],
                "source_stem": "b",
                "kind": "snap",
            },
            "bad-session.png": {
                "corners_px": [[0, 0], [1, 0], [1, 1], [0, 1]],
                "source_stem": "b",
                "kind": "snap",
                "session": 5,
            },
            "junk-entry": "nope",
        },
    }
    (frames_dir / "labels.json").write_text(json.dumps(labels))
    frames = load_frames(frames_dir)
    assert len(frames) == 1
    assert frames[0].image_path.name == "snap-keep.png"
    corners = frames[0].corners_px
    assert corners is not None
    assert np.allclose(corners, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_canonical_quad_puts_the_key_span_on_the_first_edge() -> None:
    depth_first = np.array([[254.0, 5.0], [314.0, 5.0], [298.0, 472.0], [185.0, 469.0]])
    canonical = canonical_quad(depth_first)
    span = np.linalg.norm(canonical[1] - canonical[0])
    depth = np.linalg.norm(canonical[3] - canonical[0])
    assert span > depth
    assert sorted(map(tuple, canonical)) == sorted(map(tuple, depth_first))


def test_canonical_quad_is_idempotent() -> None:
    quad = np.array([[254.0, 5.0], [314.0, 5.0], [298.0, 472.0], [185.0, 469.0]])
    once = canonical_quad(quad)
    assert np.array_equal(canonical_quad(once), once)


def _banded_keybed(back_dark: bool) -> tuple[np.ndarray, np.ndarray]:
    image = np.zeros((200, 400, 3), dtype=np.uint8)
    top, bottom = (30, 220) if back_dark else (220, 30)
    image[80:120] = top
    image[120:160] = bottom
    quad = np.array([[0.0, 80.0], [399.0, 80.0], [399.0, 159.0], [0.0, 159.0]])
    return image, quad


def test_orient_quad_puts_the_black_keys_on_the_first_edge() -> None:
    image, quad = _banded_keybed(back_dark=True)
    assert np.array_equal(orient_quad(image, quad), quad)


def test_orient_quad_flips_a_quad_that_starts_at_the_key_fronts() -> None:
    image, quad = _banded_keybed(back_dark=False)
    assert np.array_equal(orient_quad(image, quad), np.roll(quad, 2, axis=0))


def test_orient_quad_is_idempotent() -> None:
    image, quad = _banded_keybed(back_dark=False)
    once = orient_quad(image, quad)
    assert np.array_equal(orient_quad(image, once), once)


def test_parse_sidecar_returns_canonical_corners(tmp_path: Path) -> None:
    path = tmp_path / "rec-vertical.json"
    path.write_text(
        json.dumps(
            {
                "kind": "rec",
                "corners": [
                    {"x": 0.398, "y": 0.012},
                    {"x": 0.491, "y": 0.012},
                    {"x": 0.467, "y": 0.983},
                    {"x": 0.289, "y": 0.977},
                ],
                "imageWidth": 640,
                "imageHeight": 480,
            }
        )
    )
    corners = parse_sidecar(path).corners * np.array([640.0, 480.0])
    assert np.linalg.norm(corners[1] - corners[0]) > np.linalg.norm(corners[3] - corners[0])


def test_parse_sidecar_reads_a_session(tmp_path: Path) -> None:
    path = tmp_path / "auto-x.json"
    _write_sidecar(path, "auto", 320, 240, session="hold-1")
    assert parse_sidecar(path).session == "hold-1"


def test_parse_sidecar_defaults_session_to_none(tmp_path: Path) -> None:
    path = tmp_path / "snap-x.json"
    _write_sidecar(path, "snap", 320, 240)
    assert parse_sidecar(path).session is None


def test_parse_sidecar_rejects_a_non_string_session(tmp_path: Path) -> None:
    path = tmp_path / "auto-bad.json"
    path.write_text(
        json.dumps(
            {
                "kind": "auto",
                "session": 5,
                "corners": [
                    {"x": 0, "y": 0},
                    {"x": 1, "y": 0},
                    {"x": 1, "y": 1},
                    {"x": 0, "y": 1},
                ],
                "imageWidth": 1,
                "imageHeight": 1,
            }
        )
    )
    with pytest.raises(ValueError, match="session"):
        parse_sidecar(path)


def test_extract_treats_an_auto_sidecar_like_a_snap(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    frames_dir = tmp_path / "frames"
    recordings_dir.mkdir()
    _write_snap(recordings_dir / "auto-x.png")
    _write_sidecar(recordings_dir / "auto-x.json", "auto", 320, 240, session="hold-1")

    frames = extract(recordings_dir, frames_dir)

    assert len(frames) == 1
    assert frames[0].kind == "auto"
    assert frames[0].session == "hold-1"
    assert frames[0].corners_px is not None
    assert np.allclose(frames[0].corners_px, SNAP_CORNERS)

    again = load_frames(frames_dir)
    assert again[0].session == "hold-1"


def test_load_frames_keeps_null_corners(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    labels = {
        "extracted": {},
        "frames": {"f.png": {"corners_px": None, "source_stem": "f", "kind": "rec"}},
    }
    (frames_dir / "labels.json").write_text(json.dumps(labels))
    frames = load_frames(frames_dir)
    assert frames[0].corners_px is None


def test_extract_forgets_a_recording_that_was_withdrawn(tmp_path: Path) -> None:
    recordings = tmp_path / "recordings"
    frames = tmp_path / "frames"
    recordings.mkdir()
    for stem in ("snap-keep", "snap-gone"):
        _write_snap(recordings / f"{stem}.png")
        _write_sidecar(recordings / f"{stem}.json", "snap", 320, 240)
    assert {f.source_stem for f in extract(recordings, frames)} == {"snap-keep", "snap-gone"}

    (recordings / "snap-gone.png").unlink()
    (recordings / "snap-gone.json").unlink()
    served = extract(recordings, frames)
    assert {f.source_stem for f in served} == {"snap-keep"}
    assert not (frames / "snap-gone.png").exists()
    assert all(f.image_path.exists() for f in served)
