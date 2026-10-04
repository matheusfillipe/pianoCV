"""Synthetic keyboard sidecar loading and keybed quad perturbation."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data" / "synth-keys"


@dataclass(frozen=True)
class SidecarKey:
    pitch: int
    black: bool
    top: np.ndarray
    front: np.ndarray | None


@dataclass(frozen=True)
class SynthKeysFrame:
    image_path: Path
    # None for a negative: no keyboard in view, "keys" is empty too
    corners_px: np.ndarray | None
    keys: list[SidecarKey]
    image_size: tuple[int, int]
    # white where something covers the keys, as a hand does on a real frame, which training
    # leaves out
    ignore_mask: Path | None = None
    # labelled by the app from its own fit on a real recording, rather than rendered
    real: bool = False
    # a synth-motion clip id and its frame's position in it, when the sidecar carries them
    sequence: str | None = None
    frame_index: int | None = None
    # per keybed corner, in the order of corners_px: False when the case or a cheek hides it
    corner_visible: tuple[bool, ...] | None = None


def _parse_face(raw: object, scale: np.ndarray, what: str) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{what} must list exactly 4 corners")
    points = np.zeros((4, 2), dtype=np.float64)
    for i, corner in enumerate(raw):
        if not isinstance(corner, dict):
            raise ValueError(f"{what} corner {i} is not an object")
        x, y = corner.get("x"), corner.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"{what} corner {i} has invalid coordinates")
        points[i] = (float(x), float(y))
    return np.asarray(points * scale, dtype=np.float64)


def _parse_key(raw: object, scale: np.ndarray) -> SidecarKey:
    if not isinstance(raw, dict):
        raise ValueError("key entry must be an object")
    pitch, black, top, front = raw.get("pitch"), raw.get("black"), raw.get("top"), raw.get("front")
    if not isinstance(pitch, int) or not isinstance(black, bool):
        raise ValueError("key entry has invalid pitch or black flag")
    return SidecarKey(
        pitch=pitch,
        black=black,
        top=_parse_face(top, scale, "key top face"),
        front=None if front is None else _parse_face(front, scale, "key front face"),
    )


def parse_synth_keys_sidecar(sidecar_path: Path) -> SynthKeysFrame:
    data = json.loads(sidecar_path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sidecar {sidecar_path} is not a json object")
    width, height = data.get("imageWidth"), data.get("imageHeight")
    corners, keys = data.get("corners"), data.get("keys")
    if not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"sidecar {sidecar_path} has invalid image dimensions")
    if not isinstance(keys, list):
        raise ValueError(f"sidecar {sidecar_path} must list keys")
    if corners is None and keys:
        raise ValueError(f"sidecar {sidecar_path} has keys but no corners")
    ignore_mask = data.get("ignoreMask")
    if ignore_mask is not None and not isinstance(ignore_mask, str):
        raise ValueError(f"sidecar {sidecar_path} ignoreMask must name a file")
    sequence = data.get("sequence")
    if sequence is not None and not isinstance(sequence, str):
        raise ValueError(f"sidecar {sidecar_path} sequence must be a string")
    frame_index = data.get("frameIndex")
    if frame_index is not None and not isinstance(frame_index, int):
        raise ValueError(f"sidecar {sidecar_path} frameIndex must be an int")
    corner_visible = data.get("cornerVisible")
    if corner_visible is not None and (
        not isinstance(corner_visible, list)
        or len(corner_visible) != 4
        or not all(isinstance(flag, bool) for flag in corner_visible)
    ):
        raise ValueError(f"sidecar {sidecar_path} cornerVisible must list 4 booleans")
    scale = np.array([float(width), float(height)])
    return SynthKeysFrame(
        image_path=sidecar_path.with_suffix(".png"),
        corners_px=None if corners is None else _parse_face(corners, scale, "keybed quad"),
        keys=[_parse_key(key, scale) for key in keys],
        image_size=(width, height),
        ignore_mask=None if ignore_mask is None else sidecar_path.parent / ignore_mask,
        real=data.get("kind") == "real-keys",
        sequence=sequence,
        frame_index=frame_index,
        corner_visible=None if corner_visible is None else tuple(corner_visible),
    )


def load_synth_keys(data_dir: Path = DEFAULT_DATA_DIR) -> list[SynthKeysFrame]:
    if not data_dir.is_dir():
        return []
    frames: list[SynthKeysFrame] = []
    for sidecar_path in sorted(data_dir.glob("*.json")):
        frame = parse_synth_keys_sidecar(sidecar_path)
        if frame.image_path.is_file():
            frames.append(frame)
    return frames


@dataclass(frozen=True)
class PerturbConfig:
    end_shift_fraction: float = 0.12
    far_outward_max_fraction: float = 0.25
    far_tilt_fraction: float = 0.05
    near_shift_fraction: float = 0.05
    jitter_fraction: float = 0.01
    unperturbed_fraction: float = 0.15


def _unit(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector / np.linalg.norm(vector), dtype=np.float64)


def quad_axes(corners_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The keyboard's left-right and far-near directions, from the true (unperturbed) quad.

    A face's own corners carry no order guarantee, so every consumer classifies them by
    projecting onto these two axes instead of trusting a corner index.
    """
    far_left, far_right, near_right, near_left = corners_px
    key_axis = (far_right - far_left) + (near_right - near_left)
    depth_axis = (near_left - far_left) + (near_right - far_right)
    return _unit(key_axis), _unit(depth_axis)


def perturb_quad(
    corners_px: np.ndarray, rng: np.random.Generator, config: PerturbConfig | None = None
) -> np.ndarray:
    """A quad a real segmentation mask could have produced from the true one.

    The far edge only ever moves outward (a mask undershoots the case, never the keys), while
    the near edge and the two ends can move either way, so the strip sometimes crops a key off
    and sometimes runs onto the case.
    """
    config = config or PerturbConfig()
    if rng.random() < config.unperturbed_fraction:
        return np.array(corners_px, dtype=np.float64, copy=True)
    far_left, far_right, near_right, near_left = corners_px.astype(np.float64)
    key_axis, depth_axis = quad_axes(corners_px)
    quad_length = 0.5 * (
        np.linalg.norm(far_right - far_left) + np.linalg.norm(near_right - near_left)
    )
    depth = 0.5 * (np.linalg.norm(near_left - far_left) + np.linalg.norm(near_right - far_right))

    left_shift = rng.uniform(-config.end_shift_fraction, config.end_shift_fraction) * quad_length
    right_shift = rng.uniform(-config.end_shift_fraction, config.end_shift_fraction) * quad_length
    far_outward = rng.uniform(0.0, config.far_outward_max_fraction) * depth
    far_tilt_left = rng.uniform(-config.far_tilt_fraction, config.far_tilt_fraction) * depth
    far_tilt_right = rng.uniform(-config.far_tilt_fraction, config.far_tilt_fraction) * depth
    near_shift = rng.uniform(-config.near_shift_fraction, config.near_shift_fraction) * depth

    far_left = far_left + left_shift * key_axis - (far_outward + far_tilt_left) * depth_axis
    far_right = far_right + right_shift * key_axis - (far_outward + far_tilt_right) * depth_axis
    near_left = near_left + left_shift * key_axis + near_shift * depth_axis
    near_right = near_right + right_shift * key_axis + near_shift * depth_axis

    jitter_scale = config.jitter_fraction * quad_length
    jitter = rng.normal(scale=jitter_scale, size=(4, 2))
    return np.stack([far_left, far_right, near_right, near_left]) + jitter
