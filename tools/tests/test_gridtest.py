import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from pianocv.gridtest import _error, score_grid
from pianocv.model import MASK_SIZE
from pianocv.segnet2 import KeybedSegNet2

KEYBED = np.array([[100.0, 200.0], [500.0, 190.0], [505.0, 250.0], [104.0, 262.0]])


def test_error_splits_the_far_end_from_the_near_end() -> None:
    # the far end is the thinner one; here corners 1 and 2 are pushed out
    quad = KEYBED + np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 0.0], [0.0, 0.0]])
    error, near, far = _error(quad, KEYBED)
    assert near == 0.0
    assert far == 20.0
    assert error == 10.0


def test_error_tolerates_any_corner_order() -> None:
    for roll in range(4):
        assert _error(np.roll(KEYBED, roll, axis=0), KEYBED)[0] == 0.0
        assert _error(np.roll(KEYBED[::-1], roll, axis=0), KEYBED)[0] == 0.0


def test_score_grid_reads_pose_from_the_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [KEYBED.astype(np.int32)], (235, 235, 235))
    cv2.imwrite(str(tmp_path / "a.png"), image)
    (tmp_path / "a.json").write_text(
        json.dumps(
            {
                "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                "pose": {"elevation": 40, "azimuth": -30, "distance": 12.4},
            }
        )
    )
    (tmp_path / "b.json").write_text(json.dumps({"corners": []}))

    class Drawn(KeybedSegNet2):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
            grid = KEYBED / np.array([640.0, 480.0]) * (MASK_SIZE - 1)
            cv2.fillPoly(mask, [grid.astype(np.int32)], 1.0)
            logits = torch.from_numpy(mask * 20.0 - 10.0)
            return logits.view(1, 1, MASK_SIZE, MASK_SIZE).expand(x.shape[0], 1, -1, -1)

    model_path = tmp_path / "m.pt"
    torch.save(Drawn(pretrained=False).state_dict(), model_path)
    monkeypatch.setattr("pianocv.gridtest.load_seg2", lambda _: Drawn(pretrained=False))
    rows = score_grid(model_path, tmp_path)
    assert len(rows) == 1
    assert rows[0]["elevation"] == 40.0
    assert rows[0]["found"] == 1.0
    assert float(rows[0]["error"]) < 25.0
