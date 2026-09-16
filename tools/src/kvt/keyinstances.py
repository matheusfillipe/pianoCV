"""Prepare isolated per-key instance labels for a Kaggle training run."""

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RAW_DIR = _REPO_ROOT / "data" / "key-instances"
DEFAULT_OUT_DIR = DEFAULT_RAW_DIR / "kaggle"
_VALIDATION_PERCENT = 10
_PALETTE_DISTANCE = 8


@dataclass(frozen=True)
class KeyInstance:
    identifier: int
    color: int
    pitch: int
    label: str
    black: bool


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    mask_path: Path
    corners: np.ndarray
    instances: tuple[KeyInstance, ...]


def _object(value: object, path: Path) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return cast(dict[str, object], value)


def _integer(value: object, path: Path, field: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{path} field {field} must be an integer")
    return value


def _text(value: object, path: Path, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} field {field} must be text")
    return value


def _instances(value: object, path: Path) -> tuple[KeyInstance, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{path} field instances must be a list")
    instances: list[KeyInstance] = []
    for item in value:
        entry = _object(item, path)
        black = entry.get("black")
        if not isinstance(black, bool):
            raise ValueError(f"{path} field instances.black must be boolean")
        instances.append(
            KeyInstance(
                identifier=_integer(entry.get("id"), path, "instances.id"),
                color=_integer(entry.get("color"), path, "instances.color"),
                pitch=_integer(entry.get("pitch"), path, "instances.pitch"),
                label=_text(entry.get("label"), path, "instances.label"),
                black=black,
            )
        )
    identifiers = [instance.identifier for instance in instances]
    colors = [instance.color for instance in instances]
    if len(identifiers) != len(set(identifiers)) or len(colors) != len(set(colors)):
        raise ValueError(f"{path} has duplicate instance identifiers or colors")
    return tuple(instances)


def _corners(value: object, path: Path) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{path} field corners must contain four points")
    points: list[tuple[float, float]] = []
    for point in value:
        entry = _object(point, path)
        x, y = entry.get("x"), entry.get("y")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError(f"{path} corner coordinates must be numeric")
        points.append((float(x), float(y)))
    return np.asarray(points, dtype=np.float32)


def load_samples(raw_dir: Path = DEFAULT_RAW_DIR) -> list[Sample]:
    samples: list[Sample] = []
    if not raw_dir.is_dir():
        return samples
    for sidecar_path in sorted(raw_dir.glob("keyinst-*.json")):
        sidecar = _object(json.loads(sidecar_path.read_text()), sidecar_path)
        mask_name = _text(sidecar.get("instanceMask"), sidecar_path, "instanceMask")
        image_path = sidecar_path.with_suffix(".png")
        mask_path = raw_dir / mask_name
        if not image_path.is_file() or not mask_path.is_file():
            continue
        samples.append(
            Sample(
                stem=sidecar_path.stem,
                image_path=image_path,
                mask_path=mask_path,
                corners=_corners(sidecar.get("corners"), sidecar_path),
                instances=_instances(sidecar.get("instances"), sidecar_path),
            )
        )
    return samples


def _labels(mask_bgr: np.ndarray, instances: tuple[KeyInstance, ...]) -> np.ndarray:
    palette = np.asarray(
        [[item.color >> 16, (item.color >> 8) & 255, item.color & 255] for item in instances],
        dtype=np.int16,
    )
    rgb = mask_bgr[:, :, ::-1].astype(np.int16)
    distances = np.abs(rgb[:, :, None, :] - palette[None, None, :, :]).max(axis=3)
    nearest = distances.argmin(axis=2)
    close = distances.min(axis=2) <= _PALETTE_DISTANCE
    identifiers = np.asarray([item.identifier for item in instances], dtype=np.uint8)
    return np.where(close, identifiers[nearest], 0).astype(np.uint8)


def _coordinate_map(labels: np.ndarray, corners: np.ndarray) -> np.ndarray:
    height, width = labels.shape
    source = corners * np.asarray([width, height], dtype=np.float32)
    destination = np.asarray([[0, 0], [65535, 0], [65535, 65535], [0, 65535]], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    points = np.stack((x, y), axis=-1).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(points, transform)[0, :, 0].reshape(height, width)
    return np.where(labels > 0, np.clip(mapped, 0, 65535), 0).astype(np.uint16)


def _edge_map(labels: np.ndarray) -> np.ndarray:
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    edge = np.zeros(labels.shape, dtype=np.uint8)
    edge[:, 1:] |= horizontal
    edge[:, :-1] |= horizontal
    edge[1:, :] |= vertical
    edge[:-1, :] |= vertical
    return cv2.dilate(edge * 255, np.ones((3, 3), dtype=np.uint8))


def _split(stem: str) -> str:
    digest = hashlib.sha256(stem.encode()).digest()[0]
    return "validation" if digest % 100 < _VALIDATION_PERCENT else "train"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True))
            output.write("\n")


def prepare(raw_dir: Path = DEFAULT_RAW_DIR, out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, int]:
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; remove it before rebuilding")
    samples = load_samples(raw_dir)
    if not samples:
        raise ValueError(f"no complete key-instance samples in {raw_dir}")
    directories = {
        name: out_dir / name
        for name in ("images", "instance", "visible", "black", "coordinate", "edge")
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_COLOR)
        if image is None or mask is None or image.shape != mask.shape:
            raise ValueError(
                f"{sample.stem} RGB and instance mask must be readable and the same size"
            )
        labels = _labels(mask, sample.instances)
        if not np.any(labels):
            raise ValueError(f"{sample.stem} contains no recognized key-instance colors")
        black_ids = {item.identifier for item in sample.instances if item.black}
        black = np.isin(labels, list(black_ids)).astype(np.uint8) * 255
        visible = (labels > 0).astype(np.uint8) * 255
        coordinate = _coordinate_map(labels, sample.corners)
        edge = _edge_map(labels)
        cv2.imwrite(str(directories["images"] / f"{sample.stem}.png"), image)
        cv2.imwrite(str(directories["instance"] / f"{sample.stem}.png"), labels)
        cv2.imwrite(str(directories["visible"] / f"{sample.stem}.png"), visible)
        cv2.imwrite(str(directories["black"] / f"{sample.stem}.png"), black)
        cv2.imwrite(str(directories["coordinate"] / f"{sample.stem}.png"), coordinate)
        cv2.imwrite(str(directories["edge"] / f"{sample.stem}.png"), edge)
        split = _split(sample.stem)
        manifests[split].append(
            {
                "image": f"images/{sample.stem}.png",
                "instance": f"instance/{sample.stem}.png",
                "visible": f"visible/{sample.stem}.png",
                "black": f"black/{sample.stem}.png",
                "coordinate": f"coordinate/{sample.stem}.png",
                "edge": f"edge/{sample.stem}.png",
                "instances": [item.__dict__ for item in sample.instances],
            }
        )
    _write_jsonl(out_dir / "train.jsonl", manifests["train"])
    _write_jsonl(out_dir / "validation.jsonl", manifests["validation"])
    (out_dir / "README.txt").write_text(
        "Each instance pixel identifies a physical key. "
        "Coordinate is canonical horizontal position "
        "encoded as uint16. Visible, black, and edge are uint8 binary targets.\n"
    )
    return {"train": len(manifests["train"]), "validation": len(manifests["validation"])}


def archive(out_dir: Path = DEFAULT_OUT_DIR) -> Path:
    if not out_dir.is_dir():
        raise FileNotFoundError(f"prepare {out_dir} before archiving")
    path = out_dir.with_suffix(".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as output:
        for file_path in sorted(out_dir.rglob("*")):
            if file_path.is_file():
                output.write(file_path, file_path.relative_to(out_dir.parent))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="prepare isolated per-key labels for Kaggle")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--archive", action="store_true")
    args = parser.parse_args()
    if args.archive:
        print(archive(args.out_dir))
        return
    counts = prepare(args.raw_dir, args.out_dir)
    print(f"prepared {counts['train']} train and {counts['validation']} validation samples")


if __name__ == "__main__":
    main()
