from pathlib import Path

import pytest

from pianocv.keymatch import SynthKeysFrame
from pianocv.trainkeyseg import _device, _held_out


def test_held_out_matches_frames_by_clip_name() -> None:
    frame = SynthKeysFrame(Path("clipA_0001.png"), None, [], (640, 480))
    assert _held_out(frame, ("clipA",))
    assert not _held_out(frame, ("clipB",))


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"), [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: cuda)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: mps)

    assert _device().type == expected
