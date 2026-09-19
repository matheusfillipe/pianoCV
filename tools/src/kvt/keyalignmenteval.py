"""Evaluate a Kaggle per-key alignment ONNX artifact on its held-out split."""

import argparse
import json
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = _REPO_ROOT / "data" / "key-instances" / "kaggle-20260916-v2"
DEFAULT_MODEL_PATH = (
    _REPO_ROOT / "data" / "models" / "kaggle-key-alignment" / "piano_key_alignment.onnx"
)
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "models" / "kaggle-key-alignment" / "evaluation"
_INPUT_SIZE = (320, 240)
_THRESHOLD = 0.5
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def _entries(path: Path) -> list[dict[str, object]]:
    return [cast(dict[str, object], json.loads(line)) for line in path.read_text().splitlines()]


def _path(entry: dict[str, object], name: str, root: Path) -> Path:
    value = entry.get(name)
    if not isinstance(value, str):
        raise ValueError(f"validation entry field {name} must be text")
    return root / value


def _input(image_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image_rgb, _INPUT_SIZE, interpolation=cv2.INTER_AREA)
    floating = image.astype(np.float32) / 255.0
    normalized = (floating - _MEAN) / _STD
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def _iou(predicted: np.ndarray, target: np.ndarray) -> float:
    intersection = np.logical_and(predicted, target).sum()
    union = np.logical_or(predicted, target).sum()
    return float(intersection / union) if union else 1.0


def _preview(image: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    overlay[target] = (0, 220, 0)
    overlay[predicted] = (0, 0, 220)
    overlay[np.logical_and(predicted, target)] = (0, 220, 220)
    return cv2.addWeighted(image, 0.55, overlay, 0.45, 0.0)


def evaluate(model_path: Path, dataset_dir: Path, out_dir: Path) -> float:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    entries = _entries(dataset_dir / "validation.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)
    scores: list[float] = []
    for index, entry in enumerate(entries):
        image_path = _path(entry, "image", dataset_dir)
        visible_path = _path(entry, "visible", dataset_dir)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        target = cv2.imread(str(visible_path), cv2.IMREAD_GRAYSCALE)
        if image is None or target is None:
            raise ValueError(f"cannot read validation sample {image_path.stem}")
        output = cast(list[np.ndarray], session.run(None, {"image": _input(image)}))[0]
        probability = 1.0 / (1.0 + np.exp(-output[0, 0]))
        resized = cv2.resize(
            probability, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        predicted = resized >= _THRESHOLD
        expected = target >= 128
        scores.append(_iou(predicted, expected))
        if index < 12:
            cv2.imwrite(
                str(out_dir / f"{image_path.stem}.png"), _preview(image, predicted, expected)
            )
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="evaluate a Kaggle key-alignment ONNX artifact")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    print(f"validation visible IoU {evaluate(args.model, args.dataset, args.out_dir):.4f}")


if __name__ == "__main__":
    main()
