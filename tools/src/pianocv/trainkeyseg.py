"""Device choice and held-out split shared by the KeyNet trainer."""

import torch

from pianocv.keymatch import SynthKeysFrame


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _held_out(frame: SynthKeysFrame, clips: tuple[str, ...]) -> bool:
    return any(clip in frame.image_path.stem for clip in clips)
