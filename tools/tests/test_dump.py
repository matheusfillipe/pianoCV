import json
from pathlib import Path

import cv2

from pianocv.dump import _one, dump
from pianocv.segnet2 import SEG2_INPUT_SIZE


def test_dump_writes_a_frame_beside_its_sidecar(tmp_path: Path) -> None:
    assert dump(2, tmp_path, workers=1) == 2
    images = sorted(tmp_path.glob("*.png"))
    assert len(images) == 2
    for image_path in images:
        frame = cv2.imread(str(image_path))
        assert frame is not None
        assert frame.shape == (SEG2_INPUT_SIZE, SEG2_INPUT_SIZE, 3)
        sidecar = json.loads(image_path.with_suffix(".json").read_text())
        assert sidecar["kind"] == "synth"
        assert sidecar["imageWidth"] == SEG2_INPUT_SIZE
        assert len(sidecar["corners"]) == 4


def test_corners_are_normalised_to_the_frame(tmp_path: Path) -> None:
    dump(3, tmp_path, workers=1)
    for sidecar_path in tmp_path.glob("*.json"):
        for corner in json.loads(sidecar_path.read_text())["corners"]:
            # off-frame corners are expected and wanted, wildly out of range ones are a bug
            assert -3.0 < corner["x"] < 4.0
            assert -3.0 < corner["y"] < 4.0


def test_one_renders_in_process(tmp_path: Path) -> None:
    # the pool runs this in a subprocess, so it is exercised directly here as well
    _one((7, 0, str(tmp_path)))
    assert (tmp_path / "proc-000007.png").is_file()
    assert (tmp_path / "proc-000007.json").is_file()
