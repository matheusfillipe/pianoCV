"""Export the trained keybed detector to ONNX for the browser runtime."""

import argparse
from pathlib import Path

import onnx
import torch
from torch import nn

from pianocv.model import (
    INPUT_SIZE,
    KeybedNet,
    KeybedSegNet,
    decode_heatmaps,
    load_model,
    load_seg_model,
)
from pianocv.segnet2 import SEG2_INPUT_SIZE, KeybedSegNet2, load_seg2

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_net.pt"
DEFAULT_ONNX_PATH = _REPO_ROOT / "web" / "public" / "keybed_net.onnx"
DEFAULT_SEG_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_seg.pt"
DEFAULT_SEG_ONNX_PATH = _REPO_ROOT / "web" / "public" / "keybed_seg.onnx"
DEFAULT_SEG2_MODEL_PATH = _REPO_ROOT / "data" / "models" / "keybed_seg2.tuned.pt"
DEFAULT_SEG2_ONNX_PATH = _REPO_ROOT / "web" / "public" / "keybed_seg2.onnx"
_OPSET = 20


def strip_source_paths(onnx_path: Path) -> Path:
    """Drop the python tracebacks torch writes into the graph.

    The exporter records a stack trace on every node, and those lines carry the absolute path
    of the machine that ran the export. The model ships publicly, so the paths go.
    """
    model = onnx.load(str(onnx_path))
    model.ClearField("doc_string")
    model.graph.ClearField("doc_string")
    graphs = [model.graph, *model.functions]
    for graph in graphs:
        for node in graph.node:
            node.ClearField("doc_string")
            node.ClearField("metadata_props")
    for entry in list(model.metadata_props):
        if "/" in entry.value or "\\" in entry.value:
            model.metadata_props.remove(entry)
    onnx.save(model, str(onnx_path))
    return onnx_path


class _Decoded(nn.Module):
    def __init__(self, model: KeybedNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        heatmaps, present_logits = self.model(image)
        return decode_heatmaps(heatmaps), torch.sigmoid(present_logits)


class _Mask(nn.Module):
    def __init__(self, model: KeybedSegNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image))


class _Mask2(nn.Module):
    def __init__(self, model: KeybedSegNet2) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image))


def _write(
    wrapped: nn.Module,
    onnx_path: Path,
    output_names: list[str],
    example: torch.Tensor | None = None,
) -> Path:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped.eval(),
        (torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE) if example is None else example,),
        str(onnx_path),
        dynamo=True,
        optimize=True,
        # one self-contained file: the browser fetches a single url with no external tensor sidecar
        external_data=False,
        opset_version=_OPSET,
        input_names=["image"],
        output_names=output_names,
    )
    return strip_source_paths(onnx_path)


def export_seg_onnx(model_path: Path, onnx_path: Path) -> Path:
    return _write(_Mask(load_seg_model(model_path)), onnx_path, ["mask"])


def export_onnx(model_path: Path, onnx_path: Path) -> Path:
    wrapped = _Decoded(load_model(model_path)).eval()
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapped,
        (torch.zeros(1, 1, INPUT_SIZE, INPUT_SIZE),),
        str(onnx_path),
        dynamo=True,
        optimize=True,
        # one self-contained file: the browser fetches a single url with no external tensor sidecar
        external_data=False,
        opset_version=_OPSET,
        input_names=["image"],
        output_names=["corners", "present"],
    )
    return strip_source_paths(onnx_path)


def export_seg2_onnx(
    model_path: Path = DEFAULT_SEG2_MODEL_PATH, onnx_path: Path = DEFAULT_SEG2_ONNX_PATH
) -> Path:
    return _write(
        _Mask2(load_seg2(model_path)),
        onnx_path,
        ["mask"],
        torch.zeros(1, 3, SEG2_INPUT_SIZE, SEG2_INPUT_SIZE),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="export the keybed detector to onnx")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--seg", action="store_true", help="export the segmentation net")
    parser.add_argument(
        "--seg2", action="store_true", help="export the pretrained segmentation net"
    )
    args = parser.parse_args()
    if args.seg2:
        path = export_seg2_onnx(
            args.model or DEFAULT_SEG2_MODEL_PATH, args.out or DEFAULT_SEG2_ONNX_PATH
        )
    elif args.seg:
        path = export_seg_onnx(
            args.model or DEFAULT_SEG_MODEL_PATH, args.out or DEFAULT_SEG_ONNX_PATH
        )
    else:
        path = export_onnx(args.model or DEFAULT_MODEL_PATH, args.out or DEFAULT_ONNX_PATH)
    print(f"exported {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
