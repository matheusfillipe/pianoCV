"""KeySegNet: segments every key's own pixels in an oriented crop around the keybed.

The crop only rotates and scales the frame around a rough keybed quad, so the keys keep the
shape the camera gives them and the model is free to find them anywhere in it; the quad says
where to look, never what shape a key has.
"""

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from pianocv.keymatch import SidecarKey, SynthKeysFrame, quad_axes
from pianocv.segnet2 import normalise

CROP_WIDTH = 1024
CROP_HEIGHT = 224
BACKGROUND, WHITE, BLACK, BOUNDARY = 0, 1, 2, 3
CLASSES = 4
# crop pixels outside the camera frame carry no picture to learn from
IGNORE = 255
# the crop reaches this far past the quad's ends and across its depth, as a share of the
# quad's length, so keys a wrong quad cuts off still land inside it
CROP_MARGIN_ALONG = 0.08
CROP_MARGIN_ACROSS = 0.15
BOUNDARY_PX = 1

_SKIP_LAYERS = (1, 3, 8, 12)
_SKIP_CHANNELS = (16, 24, 48, 576)
_DECODER_CHANNELS = (96, 48, 24, 16, 16)


@dataclass(frozen=True)
class Crop:
    """The affine that takes frame pixels to crop pixels."""

    to_crop: np.ndarray

    def points(self, frame_px: np.ndarray) -> np.ndarray:
        ones = np.ones((len(frame_px), 1))
        return np.asarray(np.hstack([frame_px, ones]) @ self.to_crop.T, dtype=np.float64)


def crop_for(quad_px: np.ndarray) -> Crop:
    """An oriented crop with the keys running left to right and the player's edge at the bottom.

    Scale is chosen so both the quad's length and its depth fit with their margins, the same
    way the browser runtime crops, which is what lets the model run on a live frame.
    """
    key_axis, depth_axis = quad_axes(quad_px)
    across = np.array([-key_axis[1], key_axis[0]])
    if float(across @ depth_axis) < 0:
        across = -across
    centre = quad_px.mean(axis=0)
    along_extent = np.ptp((quad_px - centre) @ key_axis) * (1 + 2 * CROP_MARGIN_ALONG)
    across_extent = np.ptp((quad_px - centre) @ across) + along_extent * CROP_MARGIN_ACROSS / (
        1 + 2 * CROP_MARGIN_ALONG
    )
    scale = max(along_extent / CROP_WIDTH, across_extent / CROP_HEIGHT, 1e-6)
    rows = np.stack([key_axis / scale, across / scale])
    offset = np.array([CROP_WIDTH / 2, CROP_HEIGHT / 2]) - rows @ centre
    return Crop(to_crop=np.hstack([rows, offset[:, None]]))


def crop_image(image_bgr: np.ndarray, crop: Crop) -> np.ndarray:
    warped = cv2.warpAffine(
        image_bgr, crop.to_crop, (CROP_WIDTH, CROP_HEIGHT), flags=cv2.INTER_LINEAR
    )
    return np.asarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def black_silhouette(key: SidecarKey) -> np.ndarray:
    """A black key's visible outline: the convex hull of its eight corners.

    The sidecar carries the top face and the front face; the front's two corners that are not
    on the top give the drop from the top to the white keys, which lowers all four top corners.
    """
    if key.front is None:
        return key.top
    shared = [
        corner for corner in key.front if np.min(np.linalg.norm(key.top - corner, axis=1)) < 1e-6
    ]
    lower = [
        corner for corner in key.front if np.min(np.linalg.norm(key.top - corner, axis=1)) >= 1e-6
    ]
    if len(shared) != 2 or len(lower) != 2:
        return np.vstack([key.top, key.front])
    drop = np.mean(lower, axis=0) - np.mean(shared, axis=0)
    corners = np.vstack([key.top, key.top + drop]).astype(np.float32)
    return np.asarray(cv2.convexHull(corners)[:, 0, :], dtype=np.float64)


def label_map(frame: SynthKeysFrame, crop: Crop) -> np.ndarray:
    labels = np.zeros((CROP_HEIGHT, CROP_WIDTH), dtype=np.uint8)
    whites = [crop.points(key.top) for key in frame.keys if not key.black]
    for face in whites:
        cv2.fillPoly(labels, [np.round(face).astype(np.int32)], WHITE)
    edges = np.zeros_like(labels)
    for face in whites:
        cv2.polylines(edges, [np.round(face).astype(np.int32)], True, 1, thickness=BOUNDARY_PX)
    labels[(edges > 0) & (labels == WHITE)] = BOUNDARY
    for key in frame.keys:
        if key.black:
            outline = crop.points(black_silhouette(key))
            cv2.fillPoly(labels, [np.round(outline).astype(np.int32)], BLACK)
    width, height = frame.image_size
    inside = cv2.warpAffine(
        np.ones((height, width), dtype=np.uint8), crop.to_crop, (CROP_WIDTH, CROP_HEIGHT)
    )
    labels[inside == 0] = IGNORE
    return labels


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


def class_iou(predicted: np.ndarray, truth: np.ndarray) -> list[float]:
    ious: list[float] = []
    seen = truth != IGNORE
    for label in range(CLASSES):
        union = (seen & ((predicted == label) | (truth == label))).sum()
        inter = (seen & (predicted == label) & (truth == label)).sum()
        ious.append(float(inter / union) if union > 0 else 1.0)
    return ious


class _Softmax(nn.Module):
    def __init__(self, model: KeySegNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.model(x), dim=1)


def export_keyseg_onnx(model: KeySegNet, onnx_path: str) -> None:
    model.eval()
    torch.onnx.export(
        _Softmax(model),
        (torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH),),
        onnx_path,
        input_names=["crop"],
        output_names=["classes"],
        opset_version=17,
        dynamo=False,
    )
