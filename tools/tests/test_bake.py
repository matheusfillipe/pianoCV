import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.bake import bake
from pianocv.model import INPUT_SIZE

QUAD = [
    {"x": 0.1, "y": 0.4},
    {"x": 0.9, "y": 0.4},
    {"x": 0.92, "y": 0.55},
    {"x": 0.08, "y": 0.55},
]


def _synth_frame(directory: Path, stem: str, width: int = 640, height: int = 480) -> None:
    image = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.imwrite(str(directory / f"{stem}.png"), image)
    (directory / f"{stem}.json").write_text(
        json.dumps(
            {
                "kind": "synth",
                "startedAt": 0,
                "durationMs": 0,
                "corners": QUAD,
                "imageWidth": width,
                "imageHeight": height,
                "mimeType": "image/png",
            }
        )
    )


def test_bake_writes_one_input_sized_frame_per_source(tmp_path: Path) -> None:
    source = tmp_path / "synth"
    source.mkdir()
    _synth_frame(source, "a")
    _synth_frame(source, "b")
    out = tmp_path / "corpus"

    assert bake(source, out) == 2
    frames = sorted((out / "frames").glob("*.png"))
    assert [f.stem for f in frames] == ["a", "b"]
    baked = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    assert baked is not None
    assert baked.shape[:2] == (INPUT_SIZE, INPUT_SIZE)


def test_bake_keeps_corners_normalised_so_the_squash_does_not_move_them(
    tmp_path: Path,
) -> None:
    source = tmp_path / "synth"
    source.mkdir()
    _synth_frame(source, "a")
    out = tmp_path / "corpus"
    bake(source, out)

    corners = json.loads((out / "corners.json").read_text())
    baked = np.array(corners["a"], dtype=np.float64)
    expected = np.array([[c["x"], c["y"]] for c in QUAD])
    assert np.allclose(baked, expected, atol=1e-6)


def test_bake_ignores_a_frame_with_no_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "synth"
    source.mkdir()
    _synth_frame(source, "a")
    cv2.imwrite(str(source / "orphan.png"), np.zeros((480, 640, 3), dtype=np.uint8))
    assert bake(source, tmp_path / "corpus") == 1


def test_bake_clears_frames_from_a_previous_corpus(tmp_path: Path) -> None:
    source = tmp_path / "synth"
    source.mkdir()
    _synth_frame(source, "new")
    out = tmp_path / "corpus"
    (out / "frames").mkdir(parents=True)
    (out / "frames" / "stale.png").write_bytes(b"")

    bake(source, out)
    assert [p.name for p in (out / "frames").glob("*.png")] == ["new.png"]


def test_bake_merges_several_synth_directories_into_one_corpus(tmp_path: Path) -> None:
    keys_only = tmp_path / "synth"
    keys_only.mkdir()
    _synth_frame(keys_only, "a")
    cased = tmp_path / "synth-case"
    cased.mkdir()
    _synth_frame(cased, "b")
    out = tmp_path / "corpus-mix"

    assert bake([keys_only, cased], out) == 2
    frames = sorted((out / "frames").glob("*.png"))
    assert [f.stem for f in frames] == ["a", "b"]


def test_bake_fails_loudly_on_a_stem_shared_between_directories(tmp_path: Path) -> None:
    first = tmp_path / "synth"
    first.mkdir()
    _synth_frame(first, "dup")
    second = tmp_path / "synth-case"
    second.mkdir()
    _synth_frame(second, "dup")

    with pytest.raises(ValueError, match="dup"):
        bake([first, second], tmp_path / "corpus-mix")
