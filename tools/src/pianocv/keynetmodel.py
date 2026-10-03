"""KeyNet: one small network that finds the keyboard and its keys as keypoints.

A MobileNetV3-Small encoder with a light depthwise-separable decoder down to stride 2 gives
seven keypoint heatmaps (the keybed's four corners, the white-key gaps at the front edge, and
each black key's two front-bottom corners), and a pooled head says whether a keyboard is in
view. It is fully convolutional, so the same weights run on the oriented track crop and on the
squashed whole frame.
"""

import math

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from pianocv.keynet import CHANNELS, TRACK_HEIGHT, TRACK_WIDTH
from pianocv.keyseg import keyseg_from_onnx

_SKIP_LAYERS = (1, 3, 8, 12)
_SKIP_CHANNELS = (16, 24, 48, 576)
_DECODER_CHANNELS = (64, 32, 24, 16)
# a heatmap starts out predicting almost nothing, as CenterNet's does, so the few positive
# cells are not drowned by the loss of every empty one in the first steps
_HEAT_PRIOR = 0.01


def _separable(channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(channels, channels, 1, bias=False),
        nn.BatchNorm2d(channels),
        nn.ReLU(inplace=True),
    )


class _Up(nn.Module):
    def __init__(self, inputs: int, skip: int, outputs: int) -> None:
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(inputs + skip, outputs, 1, bias=False),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )
        self.refine = _separable(outputs)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out: torch.Tensor = self.refine(self.mix(x))
        return out


class KeyNet(nn.Module):
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
        self.heat = nn.Conv2d(inputs, CHANNELS, 1)
        self.heat.bias = nn.Parameter(
            torch.full((CHANNELS,), -math.log((1 - _HEAT_PRIOR) / _HEAT_PRIOR))
        )
        self.presence = nn.Linear(_SKIP_CHANNELS[-1], 1)
        # where each channel's point sits inside its cell, x then y in cells; it starts at
        # zero, the cell centre, so a model trained without it behaves the same
        self.offset = nn.Conv2d(inputs, 2 * CHANNELS, 1)
        nn.init.zeros_(self.offset.weight)
        self.offset.bias = nn.Parameter(torch.zeros(2 * CHANNELS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Heatmap logits at stride 2, a presence logit per image, and each cell's sub-cell
        offset for every channel."""
        skips: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in _SKIP_LAYERS:
                skips.append(x)
        y = skips[-1]
        for index, up in enumerate(self.ups):
            y = up(y, skips[len(skips) - 2 - index] if index < 3 else None)
        presence = self.presence(skips[-1].mean(dim=(2, 3)))
        heat: torch.Tensor = self.heat(y)
        offset: torch.Tensor = self.offset(y)
        return heat, presence, offset


def keynet_from_keyseg(onnx_path: str) -> KeyNet:
    """A KeyNet whose encoder starts from a published keyseg export, which has already learned
    what piano keys look like. That encoder's batch norms carry folded statistics, so training
    should keep them in eval mode."""
    model = KeyNet(pretrained=False)
    model.features.load_state_dict(keyseg_from_onnx(onnx_path).features.state_dict())
    return model


# a head saved with fewer channels grows new ones from these, the corner each new one sits
# nearest: the black keys' top-front corners lie just above their bottom-front corners, and
# each back-edge point is the front-edge point of the same kind at the other end of the key
_GROWN_FROM = {7: 5, 8: 6, 9: 4, 10: 7, 11: 8}


def keynet_from_checkpoint(pt_path: str) -> KeyNet:
    """A KeyNet that starts from a trained KeyNet's state dict, so a run that adds heatmap
    channels keeps what the saved model already finds. Its encoder came from a keyseg export
    too, so training should keep those batch norms in eval mode."""
    model = KeyNet(pretrained=False)
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    saved = state["heat.weight"].shape[0]
    if saved < CHANNELS:
        fresh = model.state_dict()
        weight, bias = fresh["heat.weight"].clone(), fresh["heat.bias"].clone()
        weight[:saved], bias[:saved] = state["heat.weight"], state["heat.bias"]
        for channel in range(saved, CHANNELS):
            source = _GROWN_FROM[channel]
            weight[channel] = weight[source]
            bias[channel] = bias[source]
        state["heat.weight"], state["heat.bias"] = weight, bias
    # a checkpoint from before the offset head leaves it at its zero start
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or any(not key.startswith("offset.") for key in missing):
        raise ValueError(f"checkpoint does not fit KeyNet: {missing} {unexpected}")
    return model


class _Probabilities(nn.Module):
    def __init__(self, model: KeyNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        heat, presence, offset = self.model(x)
        return torch.sigmoid(heat), torch.sigmoid(presence), offset


def export_keynet_onnx(model: KeyNet, onnx_path: str) -> None:
    """Exports with a free input size, so the one file serves the track crop and the search
    frame."""
    model.eval()
    torch.onnx.export(
        _Probabilities(model),
        (torch.zeros(1, 3, TRACK_HEIGHT, TRACK_WIDTH),),
        onnx_path,
        input_names=["image"],
        output_names=["heatmaps", "presence", "offsets"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "heatmaps": {2: "rows", 3: "cols"},
            "offsets": {2: "rows", 3: "cols"},
        },
        opset_version=17,
        dynamo=False,
    )
