import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from pianocv.realseg2 import _mask
from pianocv.relabelkeys import _sampled, draw_masks, main, write_contact_sheet

QUAD = [[0.1, 0.3], [0.9, 0.3], [0.9, 0.7], [0.1, 0.7]]
TRIMMED = [[0.1, 0.4], [0.9, 0.4], [0.9, 0.7], [0.1, 0.7]]


def _corpus(directory: Path, stems: list[str]) -> Path:
    (directory / "frames").mkdir(parents=True)
    for stem in stems:
        cv2.imwrite(str(directory / "frames" / f"{stem}.png"), np.full((60, 80, 3), 200, np.uint8))
    (directory / "corners.json").write_text(json.dumps(dict.fromkeys(stems, TRIMMED)))
    original = directory / "original.json"
    original.write_text(json.dumps(dict.fromkeys(stems, QUAD)))
    return original


def test_draws_one_mask_per_quad_with_the_corpus_builders_own_fill(tmp_path: Path) -> None:
    _corpus(tmp_path, ["a", "b"])
    assert draw_masks(tmp_path) == 2
    written = cv2.imread(str(tmp_path / "masks" / "a.png"), cv2.IMREAD_GRAYSCALE)
    assert written is not None
    assert np.array_equal(written, _mask(TRIMMED))


def test_lays_the_contact_sheet_out_in_rows_of_six_padding_the_last(tmp_path: Path) -> None:
    original = _corpus(tmp_path, [f"f{i}" for i in range(7)])
    write_contact_sheet(tmp_path, original)
    sheet = cv2.imread(str(tmp_path / "contact-sheet.png"))
    assert sheet is not None
    assert sheet.shape == (2 * 240, 6 * 320, 3)
    assert not sheet[240:, 320:].any()


def test_skips_frames_it_cannot_read_and_writes_nothing_when_none_remain(tmp_path: Path) -> None:
    original = _corpus(tmp_path, ["gone"])
    (tmp_path / "frames" / "gone.png").unlink()
    write_contact_sheet(tmp_path, original)
    assert not (tmp_path / "contact-sheet.png").exists()


def test_samples_evenly_and_keeps_both_ends() -> None:
    stems = [str(i) for i in range(10)]
    assert _sampled(stems, 4) == ["0", "3", "6", "9"]
    assert _sampled(stems[:3], 4) == stems[:3]


def test_runs_from_the_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    original = _corpus(tmp_path, ["a"])
    monkeypatch.setattr(
        sys, "argv", ["relabelkeys", "--dir", str(tmp_path), "--original-corners", str(original)]
    )
    main()
    assert "drew 1 masks" in capsys.readouterr().out
    assert (tmp_path / "contact-sheet.png").exists()
