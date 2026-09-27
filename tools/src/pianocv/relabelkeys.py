"""Redraw masks and a visual contact sheet for a keys-only real-frame corpus.

The relabel-keys browser pipeline (web/relabel-keys-runner.mjs) writes the keys-only quads;
this module only turns those quads into pixels, reusing realseg2's own mask fill exactly.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pianocv.realseg2 import _mask

_CONTACT_SHEET_FRAMES = 24
_CONTACT_SHEET_COLUMNS = 6
_TILE_SIZE = (320, 240)
_ORIGINAL_COLOR = (0, 0, 255)
_KEYS_COLOR = (0, 200, 0)


def draw_masks(directory: Path) -> int:
    corners: dict[str, list[list[float]]] = json.loads((directory / "corners.json").read_text())
    mask_dir = directory / "masks"
    mask_dir.mkdir(exist_ok=True)
    for stem, quad in corners.items():
        if not cv2.imwrite(str(mask_dir / f"{stem}.png"), _mask(quad)):
            raise OSError(f"cannot write mask for {stem}")
    return len(corners)


def _quad_px(quad: list[list[float]], width: int, height: int) -> np.ndarray:
    scale = np.array([width, height], dtype=np.float64)
    return (np.asarray(quad, dtype=np.float64) * scale).astype(np.int32)


def _tile(
    image_path: Path, original_quad: list[list[float]] | None, keys_quad: list[list[float]]
) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    if original_quad is not None:
        cv2.polylines(image, [_quad_px(original_quad, width, height)], True, _ORIGINAL_COLOR, 2)
    cv2.polylines(image, [_quad_px(keys_quad, width, height)], True, _KEYS_COLOR, 2)
    return cv2.resize(image, _TILE_SIZE)


def _sampled(stems: list[str], count: int) -> list[str]:
    if len(stems) <= count:
        return stems
    return [stems[round(index * (len(stems) - 1) / (count - 1))] for index in range(count)]


def write_contact_sheet(directory: Path, original_corners_path: Path) -> None:
    corners: dict[str, list[list[float]]] = json.loads((directory / "corners.json").read_text())
    original_corners: dict[str, list[list[float]]] = json.loads(original_corners_path.read_text())
    stems = sorted(corners)
    if not stems:
        return
    picked = _sampled(stems, _CONTACT_SHEET_FRAMES)
    tiles = [
        tile
        for stem in picked
        if (
            tile := _tile(
                directory / "frames" / f"{stem}.png", original_corners.get(stem), corners[stem]
            )
        )
        is not None
    ]
    if not tiles:
        return
    rows = [
        np.hstack(tiles[start : start + _CONTACT_SHEET_COLUMNS])
        for start in range(0, len(tiles), _CONTACT_SHEET_COLUMNS)
    ]
    full_width = rows[0].shape[1]
    if rows[-1].shape[1] < full_width:
        pad = np.zeros((rows[-1].shape[0], full_width - rows[-1].shape[1], 3), dtype=np.uint8)
        rows[-1] = np.hstack([rows[-1], pad])
    sheet = np.vstack(rows)
    if not cv2.imwrite(str(directory / "contact-sheet.png"), sheet):
        raise OSError("cannot write contact sheet")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="draw masks and a contact sheet for a keys-only real-frame corpus"
    )
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--original-corners", type=Path, required=True)
    args = parser.parse_args()
    count = draw_masks(args.dir)
    write_contact_sheet(args.dir, args.original_corners)
    print(f"drew {count} masks and a contact sheet in {args.dir}")


if __name__ == "__main__":
    main()
