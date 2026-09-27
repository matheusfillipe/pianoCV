import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import pianocv.gridtest as gridtest
import pianocv.jitter as jitter
from pianocv.jitter import Filter
from pianocv.model import MASK_SIZE
from pianocv.segnet2 import KeybedSegNet2

KEYBED = np.array([[100.0, 200.0], [500.0, 190.0], [505.0, 250.0], [104.0, 262.0]])


def _frame() -> np.ndarray:
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [KEYBED.astype(np.int32)], (235, 235, 235))
    return image


class Drawn(KeybedSegNet2):
    """Answers with the keybed mask whatever it is shown."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
        grid = KEYBED / np.array([640.0, 480.0]) * (MASK_SIZE - 1)
        cv2.fillPoly(mask, [grid.astype(np.int32)], 1.0)
        logits = torch.from_numpy(mask * 20.0 - 10.0)
        return logits.view(1, 1, MASK_SIZE, MASK_SIZE).expand(x.shape[0], 1, -1, -1)


class _Clip:
    """Stands in for cv2.VideoCapture over a handful of identical frames."""

    def __init__(self, _path: str, frames: int = 6) -> None:
        self.left = frames

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.left == 0:
            return False, None
        self.left -= 1
        return True, _frame()

    def release(self) -> None:
        pass


@pytest.mark.parametrize("mode", ["raw", "oneeuro", "accumulate", "both"])
def test_measure_clip_reports_a_still_keybed_as_still(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Filter
) -> None:
    monkeypatch.setattr("pianocv.jitter.cv2.VideoCapture", _Clip)
    result = jitter.measure_clip(
        Drawn(pretrained=False), tmp_path / "clip.webm", KEYBED, mode, dump_dir=tmp_path
    )
    assert result is not None
    assert result.frames == 6
    assert result.accepted == 6
    assert result.jumps == 0
    assert result.step_p95_px < 1.0
    assert result.error_px is not None and result.error_px < 25.0
    jitter.report([result])


def test_measure_clip_constrained_votes_a_focal_and_stays_on_the_keybed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("pianocv.jitter.cv2.VideoCapture", _Clip)
    result = jitter.measure_clip(
        Drawn(pretrained=False), tmp_path / "clip.webm", KEYBED, "accumulate", constrained=True
    )
    assert result is not None
    assert result.frames == 6
    assert result.accepted == 6
    assert "+constrained" in result.stem
    assert result.error_px is not None and result.error_px < 25.0


def test_dump_writes_a_sheet_of_rejected_frames(tmp_path: Path) -> None:
    small = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    rejected = [(1, _frame(), small, KEYBED), (2, _frame(), small, KEYBED)]
    jitter._dump(tmp_path, "clip", rejected)
    assert (tmp_path / "clip-rejected.png").is_file()


def test_jitter_main_runs_over_recordings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recordings = tmp_path / "recordings"
    recordings.mkdir()
    (recordings / "rec-a.webm").write_bytes(b"")
    (recordings / "rec-a.json").write_text(
        json.dumps(
            {
                "kind": "rec",
                "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                "imageWidth": 640,
                "imageHeight": 480,
            }
        )
    )
    monkeypatch.setattr("pianocv.jitter.cv2.VideoCapture", _Clip)
    monkeypatch.setattr("pianocv.jitter.load_seg2", lambda _: Drawn(pretrained=False))
    monkeypatch.setattr(
        "sys.argv",
        ["jitter", "--recordings-dir", str(recordings), "--filter", "raw", "--filter", "both"],
    )
    jitter.main()
    out = capsys.readouterr().out
    assert "rec-a [raw]" in out
    assert "rec-a [both]" in out


def test_gridtest_main_groups_by_pose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, elevation in (("a", 10), ("b", 40)):
        cv2.imwrite(str(tmp_path / f"{name}.png"), _frame())
        (tmp_path / f"{name}.json").write_text(
            json.dumps(
                {
                    "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                    "pose": {"elevation": elevation, "azimuth": 0, "distance": 12.0},
                }
            )
        )
    monkeypatch.setattr("pianocv.gridtest.load_seg2", lambda _: Drawn(pretrained=False))
    monkeypatch.setattr("sys.argv", ["gridtest", "--grid-dir", str(tmp_path)])
    gridtest.main()
    out = capsys.readouterr().out
    assert "all corners in frame: 2 poses" in out
    assert "by elevation" in out
    monkeypatch.setattr("sys.argv", ["gridtest", "--grid-dir", str(tmp_path), "--constrained"])
    gridtest.main()
    assert "all corners in frame: 2 poses" in capsys.readouterr().out
