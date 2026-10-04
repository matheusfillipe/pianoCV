"""KeySegNet: the pretrained U-Net whose published export initialises KeyNet's encoder."""

from itertools import pairwise

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from pianocv.segnet2 import normalise

CROP_WIDTH = 1024
CROP_HEIGHT = 224
CLASSES = 4

_SKIP_LAYERS = (1, 3, 8, 12)
_SKIP_CHANNELS = (16, 24, 48, 576)
_DECODER_CHANNELS = (96, 48, 24, 16, 16)


class _Up(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs + skip, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out: torch.Tensor = self.block(x)
        return out


class KeySegNet(nn.Module):
    """A MobileNetV3-Small U-Net whose decoder climbs back to the crop's full resolution, since
    the gap between two white keys is only a pixel or two wide in the crop."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        ups: list[_Up] = []
        inputs = _SKIP_CHANNELS[-1]
        for index, outputs in enumerate(_DECODER_CHANNELS):
            skip = _SKIP_CHANNELS[len(_SKIP_CHANNELS) - 2 - index] if index < 3 else 0
            ups.append(_Up(inputs, skip, outputs))
            inputs = outputs
        self.ups = nn.ModuleList(ups)
        self.out = nn.Conv2d(inputs, CLASSES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in _SKIP_LAYERS:
                skips.append(x)
        y = skips[-1]
        for index, up in enumerate(self.ups):
            y = up(y, skips[len(skips) - 2 - index] if index < 3 else None)
        logits: torch.Tensor = self.out(y)
        return logits


def preprocess_crop(crop_rgb: np.ndarray) -> np.ndarray:
    return np.asarray(normalise(crop_rgb[None])[0], dtype=np.float32)


def keyseg_from_onnx(onnx_path: str) -> KeySegNet:
    """The KeySegNet an exported ONNX file holds, to train on from where that model left off.

    The export folds each batch norm into the convolution before it, so we copy every
    convolution in the order the network runs them and set each batch norm to pass values
    through, with the folded bias as its own. Training on it should keep the batch norms in eval
    mode, since their running statistics now stand for the folded ones.
    """
    graph = onnx.load(onnx_path).graph
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in graph.initializer}
    # the exporter keeps one copy of identical tensors and names the others through Identity nodes
    for node in graph.node:
        if node.op_type == "Identity" and node.input[0] in weights:
            weights[node.output[0]] = weights[node.input[0]]
    folded = [node for node in graph.node if node.op_type == "Conv"]
    model = KeySegNet(pretrained=False).eval()
    ran: list[nn.Conv2d] = []
    hooks = [
        module.register_forward_hook(lambda conv, _inputs, _output: ran.append(conv))
        for module in model.modules()
        if isinstance(module, nn.Conv2d)
    ]
    with torch.no_grad():
        model(torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH))
    for hook in hooks:
        hook.remove()
    norm_after: dict[nn.Conv2d, nn.BatchNorm2d] = {}
    for parent in model.modules():
        children = list(parent.children())
        for first, second in pairwise(children):
            if isinstance(first, nn.Conv2d) and isinstance(second, nn.BatchNorm2d):
                norm_after[first] = second
    if len(ran) != len(folded):
        raise ValueError(f"{onnx_path} has {len(folded)} convolutions, KeySegNet runs {len(ran)}")
    with torch.no_grad():
        for conv, node in zip(ran, folded, strict=True):
            weight = torch.from_numpy(weights[node.input[1]].copy())
            if weight.shape != conv.weight.shape:
                raise ValueError(f"{onnx_path} convolution {node.name} does not fit KeySegNet")
            bias = (
                torch.from_numpy(weights[node.input[2]].copy())
                if len(node.input) > 2
                else torch.zeros(weight.shape[0])
            )
            conv.weight.copy_(weight)
            norm = norm_after.get(conv)
            if norm is None:
                if conv.bias is not None:
                    conv.bias.copy_(bias)
                continue
            if conv.bias is not None:
                conv.bias.zero_()
            if norm.running_mean is None or norm.running_var is None:
                raise ValueError("KeySegNet batch norms must track running statistics")
            norm.weight.fill_(1.0)
            norm.bias.copy_(bias)
            norm.running_mean.zero_()
            norm.running_var.fill_(1.0 - norm.eps)
    return model
