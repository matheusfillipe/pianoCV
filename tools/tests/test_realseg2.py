import json
from pathlib import Path

import cv2
import numpy as np

from kvt.realseg2 import prepare


def _frame(path: Path, color: tuple[int, int, int]) -> None:
    image = np.full((40, 80, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _labels(path: Path, names: list[str], sessions: dict[str, str] | None = None) -> None:
    sessions = sessions or {}
    entries = {
        name: {
            "corners_px": [[8.0, 4.0], [72.0, 4.0], [76.0, 36.0], [4.0, 36.0]],
            "source_stem": name.split(".")[0],
            "kind": "rec",
            "session": sessions.get(name),
        }
        for name in names
    }
    (path / "labels.json").write_text(json.dumps({"frames": entries, "extracted": {}}))


def test_prepare_writes_rgb_square_masks_and_normalised_quads(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    names = ["train-a.000000.png", "held.000000.png"]
    _frame(frames / names[0], (10, 20, 30))
    _frame(frames / names[1], (30, 20, 10))
    _labels(frames, names)
    output = tmp_path / "out"

    assert prepare(frames, output, held_out_stem="held", validation_fraction=0.0) == 2
    image = cv2.imread(str(output / "frames" / names[0]), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(output / "masks" / names[0]), cv2.IMREAD_GRAYSCALE)
    assert image is not None and image.shape == (288, 288, 3)
    assert tuple(image[0, 0]) == (10, 20, 30)
    assert mask is not None and mask.shape == (144, 144)
    corners = json.loads((output / "corners.json").read_text())[names[0].removesuffix(".png")]
    assert np.allclose(corners[0], [0.1, 0.1])
    splits = json.loads((output / "splits.json").read_text())
    assert splits["held_out"] == [names[1]]
    assert splits["train"] == [names[0]]


def test_prepare_splits_all_frames_from_a_source_together(tmp_path: Path) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    names = ["a.000000.png", "a.000001.png", "b.000000.png", "c.000000.png"]
    for index, name in enumerate(names):
        _frame(frames / name, (index, index, index))
    _labels(frames, names)
    output = tmp_path / "out"

    prepare(frames, output, held_out_stem=None, validation_fraction=0.5, seed=2)
    splits = json.loads((output / "splits.json").read_text())
    groups = {
        group for group, members in splits.items() if any(name.startswith("a.") for name in members)
    }
    assert groups == {"train"} or groups == {"validation"}


def test_prepare_splits_a_session_together_even_across_different_source_stems(
    tmp_path: Path,
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    names = ["a.png", "b.png", "c.png", "d.png"]
    for index, name in enumerate(names):
        _frame(frames / name, (index, index, index))
    _labels(
        frames,
        names,
        sessions={"a.png": "hold-1", "b.png": "hold-1", "c.png": "hold-2", "d.png": "hold-2"},
    )
    output = tmp_path / "out"

    prepare(frames, output, held_out_stem=None, validation_fraction=0.5, seed=2)
    splits = json.loads((output / "splits.json").read_text())
    group_of = {name: group for group, members in splits.items() for name in members}
    assert group_of["a.png"] == group_of["b.png"]
    assert group_of["c.png"] == group_of["d.png"]
