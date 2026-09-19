"""Prepare labelled camera frames for real-image SegNet2 fine-tuning."""

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kvt.dataset import DEFAULT_FRAMES_DIR, Frame, load_frames
from kvt.segnet2 import SEG2_INPUT_SIZE

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "data" / "real-seg2"
DEFAULT_HELD_OUT_STEM = "rec-20260913-091631"
_MASK_SIZE = 144
_DEFAULT_VALIDATION_FRACTION = 0.2
_DEFAULT_SEED = 0


@dataclass(frozen=True)
class Prepared:
    name: str
    source_stem: str
    corners: list[list[float]]


def _normalised_quad(frame: Frame) -> list[list[float]]:
    if frame.corners_px is None:
        raise ValueError(f"frame {frame.image_path} has no quad label")
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"cannot read frame {frame.image_path}")
    source_height, source_width = image.shape[:2]
    scale = np.array([float(source_width), float(source_height)])
    normalized = frame.corners_px / scale
    if not np.isfinite(normalized).all() or (normalized < 0.0).any() or (normalized > 1.0).any():
        raise ValueError(f"frame {frame.image_path} has an out-of-bounds quad")
    # The image is stretched to the square model input, so normalized coordinates stay valid.
    rounded = np.asarray(normalized, dtype=np.float64).round(8)
    return [[float(rounded[row, column]) for column in range(2)] for row in range(4)]


def _mask(corners: list[list[float]]) -> np.ndarray:
    polygon = np.asarray(corners, dtype=np.float32) * (_MASK_SIZE - 1)
    large_size = _MASK_SIZE * 4
    large = np.zeros((large_size, large_size), dtype=np.uint8)
    cv2.fillPoly(large, [(polygon * 4.0).astype(np.int32)], 255)
    return cv2.resize(large, (_MASK_SIZE, _MASK_SIZE), interpolation=cv2.INTER_AREA)


def _split_sources(
    groups: list[str], held_out_stem: str | None, validation_fraction: float, seed: int
) -> dict[str, set[str]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation fraction must be in [0, 1)")
    held_out = {held_out_stem} if held_out_stem in groups else set()
    remaining = sorted(set(groups) - held_out)
    rng = np.random.default_rng(seed)
    shuffled = [remaining[index] for index in rng.permutation(len(remaining))]
    validation_count = min(len(shuffled), round(len(shuffled) * validation_fraction))
    validation = set(shuffled[:validation_count])
    train = set(remaining) - validation
    if not train and validation:
        train.add(validation.pop())
    return {"train": train, "validation": validation, "held_out": held_out}


def prepare(
    frames_dir: Path = DEFAULT_FRAMES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    held_out_stem: str | None = DEFAULT_HELD_OUT_STEM,
    validation_fraction: float = _DEFAULT_VALIDATION_FRACTION,
    seed: int = _DEFAULT_SEED,
) -> int:
    frames = [frame for frame in load_frames(frames_dir) if frame.corners_px is not None]
    if not frames:
        raise ValueError(f"no labelled frames found in {frames_dir}")
    # a session groups every frame from one held keyboard, so they land in one split together
    # rather than tearing one hold across train and validation
    groups = [frame.session or frame.source_stem for frame in frames]
    splits = _split_sources(groups, held_out_stem, validation_fraction, seed)
    # Every invocation is a complete corpus. Retaining old frames would silently mix labels from
    # another split or camera session into the upload.
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)
    image_dir = output_dir / "frames"
    mask_dir = output_dir / "masks"
    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)
    prepared: list[Prepared] = []
    split_names: dict[str, list[str]] = {name: [] for name in splits}
    for frame in frames:
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read frame {frame.image_path}")
        name = frame.image_path.name
        corners = _normalised_quad(frame)
        # PNG stores RGB pixels; cv2's BGR array is the correct byte order for writing them.
        resized = cv2.resize(image, (SEG2_INPUT_SIZE, SEG2_INPUT_SIZE))
        if not cv2.imwrite(str(image_dir / name), resized):
            raise OSError(f"cannot write {image_dir / name}")
        if not cv2.imwrite(str(mask_dir / name), _mask(corners)):
            raise OSError(f"cannot write {mask_dir / name}")
        prepared.append(Prepared(name, frame.source_stem, corners))
        member = frame.session or frame.source_stem
        split = next(group for group, sources in splits.items() if member in sources)
        split_names[split].append(name)
    (output_dir / "corners.json").write_text(
        json.dumps({item.name.removesuffix(".png"): item.corners for item in prepared}, indent=2)
        + "\n"
    )
    (output_dir / "labels.json").write_text(
        json.dumps(
            {
                item.name: {"source_stem": item.source_stem, "corners": item.corners}
                for item in prepared
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "splits.json").write_text(
        json.dumps({key: sorted(value) for key, value in split_names.items()}, indent=2) + "\n"
    )
    return len(prepared)


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare real frames for SegNet2 fine-tuning")
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--held-out-stem", default=DEFAULT_HELD_OUT_STEM)
    parser.add_argument("--no-held-out", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=_DEFAULT_VALIDATION_FRACTION)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    args = parser.parse_args()
    held_out = None if args.no_held_out else args.held_out_stem
    count = prepare(args.frames_dir, args.output_dir, held_out, args.validation_fraction, args.seed)
    print(f"prepared {count} frames in {args.output_dir}")


if __name__ == "__main__":
    main()
