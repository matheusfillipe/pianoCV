import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import pianocv.compare as compare
from pianocv.dataset import canonical_quad, scan_recordings
from pianocv.model import MASK_SIZE

KEYBED = np.array([[100.0, 200.0], [500.0, 190.0], [505.0, 250.0], [104.0, 262.0]])


def _frame() -> np.ndarray:
    image = np.full((480, 640, 3), 25, dtype=np.uint8)
    cv2.fillPoly(image, [KEYBED.astype(np.int32)], (235, 235, 235))
    return image


def _keybed_mask() -> np.ndarray:
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    grid = KEYBED / np.array([640.0, 480.0]) * (MASK_SIZE - 1)
    cv2.fillPoly(mask, [grid.astype(np.int32)], 1.0)
    return mask


class FakeSession:
    """Answers with a fixed mask probability, whatever it is shown."""

    def __init__(self, mask: np.ndarray | None = None) -> None:
        self._mask = _keybed_mask() if mask is None else mask

    def run(
        self, output_names: list[str] | None, _input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]:
        return [self._mask[None, None]]


class _Clip:
    """Stands in for cv2.VideoCapture; a path containing "bad" refuses to open."""

    def __init__(self, path: str, frames: int = 6) -> None:
        self._ok = "bad" not in path
        self.left = frames if self._ok else 0

    def isOpened(self) -> bool:
        return self._ok

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.left == 0:
            return False, None
        self.left -= 1
        return True, _frame()

    def release(self) -> None:
        pass


def _write_grid_frame(grid_dir: Path, name: str, elevation: float, azimuth: float = 0.0) -> None:
    cv2.imwrite(str(grid_dir / f"{name}.png"), _frame())
    (grid_dir / f"{name}.json").write_text(
        json.dumps(
            {
                "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                "imageWidth": 640,
                "imageHeight": 480,
                "pose": {"elevation": elevation, "azimuth": azimuth, "distance": 12.0},
            }
        )
    )


def _write_recording(recordings_dir: Path, stem: str) -> None:
    (recordings_dir / f"{stem}.webm").write_bytes(b"")
    (recordings_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "kind": "rec",
                "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                "imageWidth": 640,
                "imageHeight": 480,
            }
        )
    )


def _write_keys_truth(tmp_path: Path, entries: dict[str, np.ndarray]) -> Path:
    path = tmp_path / "recordings-keys-truth.json"
    path.write_text(
        json.dumps(
            {
                stem: {"corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in quad]}
                for stem, quad in entries.items()
            }
        )
    )
    return path


def test_jitter_steps_on_a_known_sequence() -> None:
    base = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    quads = [base, base + np.array([3.0, 0.0]), base + np.array([7.0, 0.0])]
    steps = compare.jitter_steps(quads)
    assert steps.shape == (8,)
    assert float(np.median(steps)) == pytest.approx(3.5)
    assert float(np.percentile(steps, 95)) == pytest.approx(4.0)


def test_jitter_steps_needs_at_least_two_quads() -> None:
    assert compare.jitter_steps([np.zeros((4, 2))]).size == 0


def test_format_jitter_handles_no_steps() -> None:
    assert compare.format_jitter(np.empty(0)) == "jitter      -"


def test_mask_iou_is_high_for_a_near_perfect_overlap() -> None:
    mask = _keybed_mask()
    assert compare.mask_iou(mask, KEYBED, 640, 480) > 0.9


def test_mask_iou_is_zero_when_nothing_overlaps() -> None:
    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    assert compare.mask_iou(mask, KEYBED, 640, 480) == 0.0


def test_score_frame_finds_the_keybed() -> None:
    truth = canonical_quad(KEYBED)
    score = compare.score_frame(FakeSession(), _frame(), truth, render=False)
    assert score.iou > 0.9
    assert score.near is not None and score.near < 25.0
    assert score.far is not None and score.far < 25.0
    assert score.quad is not None


def test_score_frame_reports_none_when_no_quad_is_found() -> None:
    empty = FakeSession(mask=np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32))
    truth = canonical_quad(KEYBED)
    score = compare.score_frame(empty, _frame(), truth, render=False)
    assert score.near is None
    assert score.far is None
    assert score.quad is None


def test_score_frame_reports_none_when_truth_is_off_frame() -> None:
    truth = canonical_quad(KEYBED) + np.array([1000.0, 0.0])
    score = compare.score_frame(FakeSession(), _frame(), truth, render=False)
    assert score.near is None
    assert score.far is None


def test_score_grid_groups_by_pose(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_grid_frame(tmp_path, "a", elevation=10)
    _write_grid_frame(tmp_path, "b", elevation=40)
    sessions = {"old": FakeSession(), "new": FakeSession()}
    rows = compare.score_grid(sessions, tmp_path)
    assert len(rows) == 2
    assert {row.elevation for row in rows} == {10.0, 40.0}

    compare.print_grid_report(rows, ["old", "new"])
    out = capsys.readouterr().out
    assert "all 2" in out
    assert "elevation 10 deg" in out
    assert "elevation 40 deg" in out
    assert "azimuth  0 deg" in out


def test_role_of_matches_recording_frames_in_splits() -> None:
    splits = {
        "train": ["rec-a.000000.png", "rec-a.000006.png"],
        "validation": ["snap-b.png"],
        "held_out": ["rec-c.000000.png"],
    }
    assert compare.role_of("rec-a", "rec", splits) == "train"
    assert compare.role_of("snap-b", "snap", splits) == "validation"
    assert compare.role_of("rec-c", "rec", splits) == "held_out"
    assert compare.role_of("rec-unseen", "rec", splits) == "unseen"


def test_load_splits_returns_empty_when_file_is_missing(tmp_path: Path) -> None:
    assert compare.load_splits(tmp_path / "missing.json") == {}


def test_read_recording_frames_returns_none_for_a_broken_clip(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "rec-bad.webm").write_bytes(b"not a real video")
    _write_recording(recordings_dir, "rec-bad")
    recording = scan_recordings(recordings_dir)[0]
    assert compare.read_recording_frames(recording, stride=1) is None


def test_read_recording_frames_returns_none_for_an_unreadable_snapshot(tmp_path: Path) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    (recordings_dir / "snap-bad.png").write_text("not an image")
    (recordings_dir / "snap-bad.json").write_text(
        json.dumps(
            {
                "kind": "snap",
                "corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED],
                "imageWidth": 640,
                "imageHeight": 480,
            }
        )
    )
    recording = scan_recordings(recordings_dir)[0]
    assert compare.read_recording_frames(recording, stride=1) is None


def test_evaluate_recordings_tags_role_and_skips_unreadable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    _write_recording(recordings_dir, "rec-good")
    _write_recording(recordings_dir, "rec-bad")
    monkeypatch.setattr("pianocv.compare.cv2.VideoCapture", _Clip)
    splits = {"held_out": ["rec-good.000000.png"]}
    results = compare.evaluate_recordings({"only": FakeSession()}, recordings_dir, splits, stride=1)
    assert [result.stem for result in results] == ["rec-good"]
    assert results[0].role == "held_out"
    assert results[0].frames == 6
    assert "skipping unreadable recording rec-bad" in capsys.readouterr().out


def test_load_keys_truth_reads_corners_and_marks_skipped(tmp_path: Path) -> None:
    path = tmp_path / "truth.json"
    path.write_text(
        json.dumps(
            {
                "rec-a": {"corners": [{"x": x / 640.0, "y": y / 480.0} for x, y in KEYBED]},
                "rec-b": {"skipped": "far edge could not be measured"},
            }
        )
    )
    truth = compare.load_keys_truth(path)
    assert truth["rec-b"] is None
    assert truth["rec-a"] is not None
    assert truth["rec-a"].shape == (4, 2)


def test_load_keys_truth_returns_empty_when_file_is_missing(tmp_path: Path) -> None:
    assert compare.load_keys_truth(tmp_path / "missing.json") == {}


def test_evaluate_recordings_with_keys_truth_overrides_corners_and_skips_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    _write_recording(recordings_dir, "rec-good")
    _write_recording(recordings_dir, "rec-other")
    monkeypatch.setattr("pianocv.compare.cv2.VideoCapture", _Clip)
    keys_truth = compare.load_keys_truth(_write_keys_truth(tmp_path, {"rec-good": KEYBED}))
    results = compare.evaluate_recordings(
        {"only": FakeSession()},
        recordings_dir,
        splits={},
        stride=1,
        truth_source="keys",
        keys_truth=keys_truth,
    )
    assert [result.stem for result in results] == ["rec-good"]
    assert "skipping rec-other: no keys-only truth" in capsys.readouterr().out


def _clip_result(stem: str, role: compare.Role, iou: float) -> compare.ClipResult:
    score = compare.FrameScore(iou=iou, near=None, far=None, quad=None)
    return compare.ClipResult(
        stem=stem, kind="rec", role=role, frames=1, scores={"m": [score]}, jitter={"m": np.empty(0)}
    )


def test_evidence_summary_only_counts_held_out_and_unseen(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        _clip_result("a", "train", 0.1),
        _clip_result("b", "held_out", 0.9),
        _clip_result("c", "unseen", 0.8),
    ]
    compare.print_evidence_summary(results, ["m"])
    out = capsys.readouterr().out
    assert "2 recordings" in out
    assert "IoU 0.85" in out


def test_parse_models_rejects_a_spec_without_equals() -> None:
    with pytest.raises(ValueError, match="NAME=PATH"):
        compare._parse_models(["bad-spec"])


def test_main_requires_at_least_two_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["compare", "--model", "only=one.onnx"])
    with pytest.raises(SystemExit):
        compare.main()


def test_main_rejects_a_model_spec_without_equals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv", ["compare", "--model", "bad-spec", "--model", "other=path.onnx"]
    )
    with pytest.raises(SystemExit):
        compare.main()


def test_main_runs_end_to_end(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    grid_dir = tmp_path / "grid"
    grid_dir.mkdir()
    _write_grid_frame(grid_dir, "a", elevation=10)
    recordings_dir = tmp_path / "recordings"
    recordings_dir.mkdir()
    _write_recording(recordings_dir, "rec-good")
    splits_path = tmp_path / "splits.json"
    splits_path.write_text(json.dumps({"train": ["rec-good.000000.png"]}))
    monkeypatch.setattr("pianocv.compare.cv2.VideoCapture", _Clip)
    monkeypatch.setattr("pianocv.compare.ort.InferenceSession", lambda path: FakeSession())
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare",
            "--model",
            "old=old.onnx",
            "--model",
            "new=new.onnx",
            "--grid",
            str(grid_dir),
            "--recordings",
            str(recordings_dir),
            "--splits",
            str(splits_path),
            "--every",
            "1",
        ],
    )
    compare.main()
    out = capsys.readouterr().out
    assert "SYNTHETIC GRID" in out
    assert "REAL RECORDINGS" in out
    assert "rec-good" in out
    assert "train" in out
    assert "EVIDENCE (held_out + unseen only, 0 recordings)" in out
