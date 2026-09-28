"""KeySegNet: segments every key's own pixels in an oriented crop around the keybed.

The crop only rotates and scales the frame around a rough keybed quad, so the keys keep the
shape the camera gives them and the model is free to find them anywhere in it; the quad says
where to look, never what shape a key has.
"""

from dataclasses import dataclass
from itertools import pairwise

import cv2
import numpy as np
import onnx
import torch
from onnx import numpy_helper
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
# a hand mask comes from a segmenter's low-resolution output, so its edge is grown before use
IGNORE_GROWTH_PX = 7
# a real label is only corrected where its pixel sits this share of the way past the split
# towards the other class's median, so a soft edge or a glint keeps its label
_REFINE_MARGIN = 0.35
_REFINE_LEAST_PIXELS = 100

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
    if frame.ignore_mask is not None:
        covered = cv2.imread(str(frame.ignore_mask), cv2.IMREAD_GRAYSCALE)
        if covered is None:
            raise FileNotFoundError(f"cannot read ignore mask {frame.ignore_mask}")
        covered = cv2.dilate(covered, np.ones((IGNORE_GROWTH_PX, IGNORE_GROWTH_PX), np.uint8))
        warped = cv2.warpAffine(covered, crop.to_crop, (CROP_WIDTH, CROP_HEIGHT))
        labels[warped > 127] = IGNORE
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


def refine_by_brightness(labels: np.ndarray, crop_rgb: np.ndarray) -> np.ndarray:
    """A real frame's labels, corrected by its own pixels.

    The app's fit places a black key within a fraction of a key, and far away that is enough to
    run one black key into the next. A pixel keeps a black label only where it is clearly dark:
    clearly bright it is the white key between two black ones, and in between it is left out,
    as the shadowed strip between two black keys at the back is. A pixel labelled white that
    is clearly dark is a shadow or a black key the fit missed, and is left out too, as is every
    black label on a row whose white keys are dark, which is the case's lip or a shadow.
    """
    grey = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    keys = (labels == WHITE) | (labels == BLACK)
    if int(keys.sum()) < _REFINE_LEAST_PIXELS:
        return labels
    split, _ = cv2.threshold(grey[keys].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = float(np.median(grey[keys & (grey >= split)]))
    dark = float(np.median(grey[keys & (grey < split)]))
    clearly_dark = grey < split - _REFINE_MARGIN * (split - dark)
    clearly_bright = grey > split + _REFINE_MARGIN * (bright - split)
    refined = labels.copy()
    refined[(labels == BLACK) & ~clearly_dark] = IGNORE
    refined[(labels == BLACK) & clearly_bright] = WHITE
    refined[(labels == WHITE) & clearly_dark] = IGNORE
    # a crop row whose white keys are dark too runs through the case or a shadow, where a dark
    # pixel says nothing about a black key; the crop's rows run across the keys
    for row in range(labels.shape[0]):
        whites = grey[row][labels[row] == WHITE]
        if whites.size and float(np.median(whites)) < split:
            refined[row][labels[row] == BLACK] = IGNORE
    return refined


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
