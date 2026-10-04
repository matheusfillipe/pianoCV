"""Convert a KeyNet export to half precision for phones and GPUs, keeping float32 inputs and outputs
so every runtime feeds and reads it unchanged."""

import argparse
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import convert_float_to_float16


def to_fp16(source: Path, target: Path) -> None:
    model = onnx.load(str(source))
    onnx.save(convert_float_to_float16(model, keep_io_types=True), str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description="convert a KeyNet export to half precision")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    to_fp16(args.source, args.target)
    print(f"wrote {args.target} ({args.target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
