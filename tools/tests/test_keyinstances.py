import json
from pathlib import Path

import cv2
import numpy as np

from kvt.keyinstances import archive, prepare


def _sample(raw_dir: Path, stem: str) -> None:
    image = np.full((12, 16, 3), 127, dtype=np.uint8)
    mask = np.zeros_like(image)
    mask[:, :8] = (32, 32, 32)
    mask[:, 8:] = (32, 32, 76)
    cv2.imwrite(str(raw_dir / f"{stem}.png"), image)
    cv2.imwrite(str(raw_dir / f"{stem}-keys.png"), mask)
    (raw_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "corners": [
                    {"x": 0.0, "y": 0.0},
                    {"x": 1.0, "y": 0.0},
                    {"x": 1.0, "y": 1.0},
                    {"x": 0.0, "y": 1.0},
                ],
                "instanceMask": f"{stem}-keys.png",
                "instances": [
                    {"id": 1, "color": 0x202020, "pitch": 17, "label": "F0", "black": False},
                    {"id": 2, "color": 0x4C2020, "pitch": 18, "label": "F#0", "black": True},
                ],
            }
        )
    )


def test_prepare_writes_isolated_multitask_targets(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _sample(raw, "keyinst-a")
    out = tmp_path / "kaggle"

    counts = prepare(raw, out)

    assert sum(counts.values()) == 1
    instance = cv2.imread(str(out / "instance" / "keyinst-a.png"), cv2.IMREAD_UNCHANGED)
    black = cv2.imread(str(out / "black" / "keyinst-a.png"), cv2.IMREAD_UNCHANGED)
    coordinate = cv2.imread(str(out / "coordinate" / "keyinst-a.png"), cv2.IMREAD_UNCHANGED)
    assert instance is not None and set(np.unique(instance)) == {1, 2}
    assert black is not None and set(np.unique(black)) == {0, 255}
    assert coordinate is not None and coordinate.dtype == np.uint16
    assert int(coordinate[0, 0]) < int(coordinate[0, -1])


def test_prepare_never_replaces_an_existing_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _sample(raw, "keyinst-a")
    out = tmp_path / "kaggle"
    out.mkdir()

    try:
        prepare(raw, out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("prepare must not overwrite an existing dataset")


def test_archive_packages_the_prepared_dataset(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    _sample(raw, "keyinst-a")
    out = tmp_path / "kaggle"
    prepare(raw, out)

    path = archive(out)

    assert path.is_file()
