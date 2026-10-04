"""Train KeyNet on still renders, rendered camera paths and real frames the app labelled itself.

Each batch is either track crops, the oriented crop the runtime takes around the previous
frame's keyboard (jittered so the model learns to recover when the camera moves), or search
frames, the whole frame squashed to a square. Frames without a keyboard teach it to say so.
Runs on a GPU or locally.
"""

import argparse
import os
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, get_worker_info

from pianocv.keymatch import PerturbConfig, SynthKeysFrame, load_synth_keys, perturb_quad
from pianocv.keynet import (
    BACK_CHANNELS,
    BLACK_TOP_CHANNELS,
    SEARCH_SIZE,
    STRIDE,
    TRACK_HEIGHT,
    TRACK_MARGIN_ACROSS,
    TRACK_MARGIN_ALONG,
    TRACK_WIDTH,
    Crop,
    KeyPoints,
    fixed_path,
    frame_points,
    has_back_labels,
    heatmap_targets,
    offset_targets,
    oriented_crop,
    presence_target,
    rectified_crop,
    squash_crop,
)
from pianocv.keynetmodel import (
    KeyNet,
    export_keynet_onnx,
    keynet_from_checkpoint,
    keynet_from_keyseg,
)
from pianocv.keyseg import preprocess_crop
from pianocv.trainkeymatch import _augment_strip, _to_uint8
from pianocv.trainkeyseg import _device, _held_out

try:
    import mlflow
except ImportError:  # pragma: no cover - exercised only where mlflow is absent
    mlflow = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "models" / "keynet"

_DEFAULT_EPOCHS = 30
_DEFAULT_BATCH = 32
_DEFAULT_LR = 2e-3
_DEFAULT_STEPS_PER_EPOCH = 400
_VAL_FRACTION = 0.03
_VAL_SEED = 12345
_SEARCH_EVERY = 4
# how far a track crop may sit from the true keyboard, as the previous frame's pose leaves it
# when the camera moves
_JITTER_SHIFT = 0.08
_JITTER_ROTATE_DEG = 6.0
_JITTER_SCALE = (0.85, 1.15)
# a rectified crop is built from the previous fit, whose corners are each a little off
_CORNER_JITTER = 0.015
# the share of a track crop's white-key gaps that must lie inside it to count as still tracking
_TRACKING_SHARE = 0.5
_PEAK_THRESHOLD = 0.3
_MATCH_CELLS = 1.5
_PRESENCE_WEIGHT = 1.0
# a target cell under a point's Gaussian peak, which an off-centre point keeps above this
_SINGLE_PEAK_LEAST = 0.5
# CenterNet weighs its offset loss the same as its heatmap loss
_OFFSET_WEIGHT = 1.0
# phone and webcam captures run from 480 to 960 pixels wide
_LOW_RES_WIDTHS = (480.0, 960.0)
_CAMERA_SCALE = (0.4, 0.9)
_CAMERA_JPEG_QUALITY = (30, 90)


@dataclass(frozen=True)
class Sources:
    """The frames to draw from, and how often each set is drawn."""

    sets: tuple[tuple[list[SynthKeysFrame], float], ...]

    def draw(self, rng: np.random.Generator) -> SynthKeysFrame:
        weights = np.array([share for frames, share in self.sets if frames], dtype=np.float64)
        pools = [frames for frames, _ in self.sets if frames]
        pool = pools[int(rng.choice(len(pools), p=weights / weights.sum()))]
        return pool[int(rng.integers(len(pool)))]


def _jitter(quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    centre = quad.mean(axis=0)
    length = float(np.linalg.norm(quad[1] - quad[0]))
    angle = np.deg2rad(rng.uniform(-_JITTER_ROTATE_DEG, _JITTER_ROTATE_DEG))
    scale = rng.uniform(*_JITTER_SCALE)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = rng.uniform(-_JITTER_SHIFT, _JITTER_SHIFT, size=2) * length
    return np.asarray((quad - centre) @ rotation.T * scale + centre + shift, dtype=np.float64)


def _jitter_corners(quad: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    length = float(np.linalg.norm(quad[1] - quad[0]))
    return np.asarray(
        quad + rng.uniform(-_CORNER_JITTER, _CORNER_JITTER, size=quad.shape) * length,
        dtype=np.float64,
    )


def _decoy_quad(frame_size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """A keyboard-shaped quad over nothing in particular, for a frame without a keyboard."""
    width, height = frame_size
    length = rng.uniform(0.4, 0.9) * width
    depth = length * rng.uniform(0.12, 0.3)
    centre = np.array([rng.uniform(0.3, 0.7) * width, rng.uniform(0.3, 0.7) * height])
    box = np.array(
        [
            (-length / 2, -depth / 2),
            (length / 2, -depth / 2),
            (length / 2, depth / 2),
            (-length / 2, depth / 2),
        ]
    )
    angle = np.deg2rad(rng.uniform(-30.0, 30.0))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(box @ rotation.T + centre, dtype=np.float64)


def _tracking(points: KeyPoints | None, crop: Crop) -> float:
    if points is None or len(points.gaps) == 0:
        return 0.0
    inside = crop.points(points.gaps)
    within = (
        (inside[:, 0] >= 0)
        & (inside[:, 0] < TRACK_WIDTH)
        & (inside[:, 1] >= 0)
        & (inside[:, 1] < TRACK_HEIGHT)
    )
    return 1.0 if float(within.mean()) >= _TRACKING_SHARE else 0.0


def _ignore_mask(frame: SynthKeysFrame) -> np.ndarray | None:
    if frame.ignore_mask is None:
        return None
    mask = cv2.imread(str(frame.ignore_mask), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"cannot read ignore mask {frame.ignore_mask}")
    return np.asarray(mask, dtype=np.uint8)


@dataclass(frozen=True)
class Recipe:
    """Training choices a run can switch on, so each run changes one thing against the last."""

    # the share of samples that pass through _lower_resolution
    low_res_share: float = 0.0
    # whether real frames teach the black-key tops, which their labels only guess
    mask_real_tops: bool = False
    # whether only each point's own heatmap cell counts as a positive
    single_peak: bool = False
    # whether the offset head learns where each point sits inside its cell
    offsets: bool = False
    # where hand-corrected labels live; a real frame with a file there trains on it, tops included
    fixed_dir: Path | None = None
    # whether track crops are the rectified homography around the previous fit instead of the
    # oriented box
    rectified: bool = False
    # whether every input crop passes through the camera pipeline simulation
    camera_sim: bool = False
    # whether only the output layers learn, so new points or labels leave what the model sees alone
    heads_only: bool = False


BASE_RECIPE = Recipe()


def sample(
    frame: SynthKeysFrame,
    search: bool,
    rng: np.random.Generator,
    config: PerturbConfig,
    augment: bool,
    recipe: Recipe = BASE_RECIPE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """One model input with its heatmap targets, their loss weights, the presence target, and
    the sub-cell offset targets with their weights."""
    image = cv2.imread(str(frame.image_path))
    if image is None:
        raise FileNotFoundError(f"cannot read frame {frame.image_path}")
    size = (image.shape[1], image.shape[0])
    points = frame_points(frame, recipe.fixed_dir)
    if search:
        crop = squash_crop(size, SEARCH_SIZE, SEARCH_SIZE)
        width, height = SEARCH_SIZE, SEARCH_SIZE
        presence = presence_target(frame)
    else:
        quad = (
            _jitter(perturb_quad(points.corners, rng, config), rng)
            if points is not None and np.isfinite(points.corners).all()
            else _decoy_quad(size, rng)
        )
        make_crop = oriented_crop
        if recipe.rectified:
            quad = _jitter_corners(quad, rng)
            make_crop = rectified_crop
        crop = make_crop(quad, TRACK_WIDTH, TRACK_HEIGHT, TRACK_MARGIN_ALONG, TRACK_MARGIN_ACROSS)
        width, height = TRACK_WIDTH, TRACK_HEIGHT
        presence = _tracking(points, crop)
    if augment and rng.random() < recipe.low_res_share:
        image = _lower_resolution(image, rng)
    warped = cv2.warpPerspective(image, crop.to_crop, (width, height), flags=cv2.INTER_LINEAR)
    pixels = np.asarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB), dtype=np.uint8)
    if augment:
        pixels = _augment_strip(pixels, rng)
    if augment and recipe.camera_sim:
        pixels = camera_sim(pixels, rng)
    heat, weight = heatmap_targets(points, crop, (height, width), size, _ignore_mask(frame))
    if frame.real and recipe.mask_real_tops and fixed_path(frame, recipe.fixed_dir) is None:
        # a real frame's black-key tops come from the old pipeline's drawing, a guess at the
        # raise, so we teach the tops from renders alone, where they are the true geometry
        weight[list(BLACK_TOP_CHANNELS)] = 0.0
    if frame.real and not has_back_labels(fixed_path(frame, recipe.fixed_dir)):
        weight[list(BACK_CHANNELS)] = 0.0
    offset, near = offset_targets(points, crop, (height, width))
    # a cell the heatmap leaves out, under a hand or past the frame, teaches no offset either
    return pixels, heat, weight, presence, offset, near * weight


def _lower_resolution(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The frame as a lower-resolution camera would give it, at its own size, so a render
    teaches the model the few pixels a key gets in a small capture."""
    height, width = image.shape[:2]
    target = rng.uniform(*_LOW_RES_WIDTHS)
    if target >= width:
        return image
    scale = target / width
    small = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return np.asarray(cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR))


def camera_sim(pixels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """An RGB crop as another camera would give it. Every step keeps the pixel grid, so the
    targets built from the crop stay aligned; lens distortion is left out because it would move
    the labels."""
    height, width = pixels.shape[:2]
    image = pixels.astype(np.float32)
    image *= rng.uniform(0.85, 1.15, size=3)
    if rng.random() < 0.5:
        image = _defocus(image, rng)
    scale = rng.uniform(*_CAMERA_SCALE)
    small = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA
    )
    image = np.asarray(cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    radius = ((xs / width - 0.5) ** 2 + (ys / height - 0.5) ** 2) / 0.5
    image *= (1.0 - rng.uniform(0.0, 0.5) * radius)[..., None]
    image = 255.0 * (np.clip(image, 0.0, 255.0) / 255.0) ** rng.uniform(0.8, 1.25)
    image = (image - 128.0) * rng.uniform(0.75, 1.25) + 128.0 + rng.uniform(-20.0, 20.0)
    # shot noise grows with the signal, read noise does not
    shot = rng.uniform(0.0, 0.6) * np.sqrt(np.clip(image, 0.0, 255.0))
    image += rng.normal(size=image.shape) * (shot + rng.uniform(0.0, 5.0))
    return _compress(_to_uint8(image), rng)


def _defocus(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    radius = int(rng.integers(1, 4))
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    return np.asarray(cv2.filter2D(image, -1, kernel / kernel.sum()), dtype=np.float32)


def _compress(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """JPEG at a random quality after the chroma blur a 4:2:0 video codec leaves."""
    luma = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    height, width = luma.shape[:2]
    half = (max(1, width // 2), max(1, height // 2))
    for channel in (1, 2):
        luma[..., channel] = cv2.resize(
            cv2.resize(luma[..., channel], half, interpolation=cv2.INTER_AREA),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    bgr = cv2.cvtColor(luma, cv2.COLOR_YCrCb2BGR)
    quality = int(rng.integers(_CAMERA_JPEG_QUALITY[0], _CAMERA_JPEG_QUALITY[1] + 1))
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
    return np.asarray(cv2.cvtColor(decoded if decoded is not None else bgr, cv2.COLOR_BGR2RGB))


class KeyNetDataset(Dataset[tuple[torch.Tensor, ...]]):
    """A fresh draw, perturbation and augmentation of a random frame on every read, all track
    crops or all search frames so a batch shares one input size."""

    def __init__(
        self, sources: Sources, length: int, seed: int, search: bool, recipe: Recipe = BASE_RECIPE
    ) -> None:
        self.sources = sources
        self.length = length
        self.seed = seed
        self.search = search
        self.recipe = recipe
        self.config = PerturbConfig()
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        rng = np.random.default_rng((self.seed, self.epoch, worker_id, index, int(self.search)))
        pixels, heat, weight, presence, offset, offset_weight = sample(
            self.sources.draw(rng), self.search, rng, self.config, True, self.recipe
        )
        return (
            torch.from_numpy(preprocess_crop(pixels)),
            torch.from_numpy(heat),
            torch.from_numpy(weight),
            torch.tensor([presence], dtype=torch.float32),
            # a run without the offset head skips its targets, which are most of a sample's memory
            torch.from_numpy(offset) if self.recipe.offsets else torch.zeros(0),
            torch.from_numpy(offset_weight) if self.recipe.offsets else torch.zeros(0),
        )


def heatmap_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, single_peak: bool = False
) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss, over the cells the weights keep, normalised per
    channel and then averaged: a crop has four keybed corners against dozens of gaps, and
    normalising by every positive at once left the corners, which fix the board's ends, with a
    few percent of the signal."""
    probability = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    if single_peak:
        # each point's own cell is the only positive, as in CenterNet, so the model learns the
        # Gaussian's shape around it and a decoder can read the point's place inside the cell
        grown = nn.functional.max_pool2d(target, 3, stride=1, padding=1)
        peak = (target >= grown) & (target >= _SINGLE_PEAK_LEAST)
    else:
        peak = target >= 0.8
    positive = -((1 - probability) ** 2) * torch.log(probability)
    negative = -((1 - target) ** 4) * probability**2 * torch.log(1 - probability)
    loss = torch.where(peak, positive, negative) * weight
    per_channel = loss.sum(dim=(0, 2, 3)) / (peak.float() * weight).sum(dim=(0, 2, 3)).clamp(
        min=1.0
    )
    return per_channel.mean()


def offset_loss(
    predicted: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """L1 on each cell's sub-cell offset, x and y, weighted by how near the cell is to its point
    and averaged over that weight, as CenterNet trains its offsets."""
    both = weight.repeat_interleave(2, dim=1)
    return (torch.abs(predicted - target) * both).sum() / both.sum().clamp(min=1.0)


@dataclass(frozen=True)
class PeakScore:
    precision: float
    recall: float
    error_px: float


def _peaks(heat: np.ndarray) -> list[np.ndarray]:
    """Each channel's local maxima above the threshold, as (row, col) cells."""
    found: list[np.ndarray] = []
    for channel in heat:
        grown = cv2.dilate(channel, np.ones((3, 3), np.uint8))
        rows, cols = np.nonzero((channel >= grown) & (channel >= _PEAK_THRESHOLD))
        found.append(np.stack([rows, cols], axis=1).astype(np.float64))
    return found


def score_peaks(predicted: np.ndarray, target: np.ndarray, weight: np.ndarray) -> PeakScore:
    """How well the predicted peaks match the target's, over the cells the weights keep."""
    hits = 0
    predictions = 0
    truths = 0
    errors: list[float] = []
    for guess, truth, kept in zip(
        _peaks(predicted), _peaks(target * (target >= 0.8)), weight, strict=True
    ):
        guess = np.array([cell for cell in guess if kept[int(cell[0]), int(cell[1])] > 0])
        counted = [cell for cell in truth if kept[int(cell[0]), int(cell[1])] > 0]
        predictions += len(guess)
        truths += len(counted)
        for cell in counted:
            if len(guess) == 0:
                continue
            distances = np.linalg.norm(guess - cell, axis=1)
            if float(distances.min()) <= _MATCH_CELLS:
                hits += 1
                errors.append(float(distances.min()) * STRIDE)
    return PeakScore(
        precision=hits / predictions if predictions else 1.0,
        recall=hits / truths if truths else 1.0,
        error_px=float(np.mean(errors)) if errors else 0.0,
    )


@dataclass(frozen=True)
class Evaluation:
    loss: float
    peaks: PeakScore
    presence_accuracy: float


def evaluate(
    model: KeyNet,
    frames: list[SynthKeysFrame],
    search: bool,
    device: torch.device,
    batch: int,
    recipe: Recipe = BASE_RECIPE,
) -> Evaluation:
    model.eval()
    config = PerturbConfig()
    losses: list[float] = []
    scores: list[PeakScore] = []
    correct = 0
    with torch.no_grad():
        for start in range(0, len(frames), batch):
            chunk = [
                sample(
                    frame,
                    search,
                    np.random.default_rng((_VAL_SEED, start + i)),
                    config,
                    False,
                    recipe,
                )
                for i, frame in enumerate(frames[start : start + batch])
            ]
            crops = torch.stack([torch.from_numpy(preprocess_crop(c[0])) for c in chunk]).to(device)
            heat = torch.stack([torch.from_numpy(c[1]) for c in chunk]).to(device)
            weight = torch.stack([torch.from_numpy(c[2]) for c in chunk]).to(device)
            logits, presence, _ = model(crops)
            losses.append(float(heatmap_loss(logits, heat, weight)) * len(chunk))
            probabilities = torch.sigmoid(logits).cpu().numpy()
            for i, item in enumerate(chunk):
                scores.append(score_peaks(probabilities[i], item[1], item[2]))
                said = float(torch.sigmoid(presence[i, 0])) >= 0.5
                correct += int(said == (item[3] >= 0.5))
    count = max(len(frames), 1)
    return Evaluation(
        loss=sum(losses) / count,
        peaks=PeakScore(
            precision=float(np.mean([s.precision for s in scores])) if scores else 0.0,
            recall=float(np.mean([s.recall for s in scores])) if scores else 0.0,
            error_px=float(np.mean([s.error_px for s in scores])) if scores else 0.0,
        ),
        presence_accuracy=correct / count,
    )


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    synthetic: Evaluation
    real: Evaluation | None


def _batches(loader: DataLoader[tuple[torch.Tensor, ...]]) -> Iterator[tuple[torch.Tensor, ...]]:
    while True:
        yield from loader


def _split(
    frames: list[SynthKeysFrame], seed: int
) -> tuple[list[SynthKeysFrame], list[SynthKeysFrame]]:
    order = np.random.default_rng(seed).permutation(len(frames))
    count = max(1, round(len(frames) * _VAL_FRACTION)) if len(frames) > 1 else 0
    held = set(order[:count].tolist())
    train = [frame for i, frame in enumerate(frames) if i not in held] or frames
    return train, [frame for i, frame in enumerate(frames) if i in held]


def train_keynet(
    still_dir: Path,
    *,
    motion_dir: Path | None = None,
    real_dir: Path | None = None,
    shares: tuple[float, float, float] = (0.5, 0.3, 0.2),
    hold_out: tuple[str, ...] = (),
    init_onnx: Path | None = None,
    init_pt: Path | None = None,
    recipe: Recipe = BASE_RECIPE,
    epochs: int = _DEFAULT_EPOCHS,
    batch: int = _DEFAULT_BATCH,
    lr: float = _DEFAULT_LR,
    seed: int = 0,
    steps_per_epoch: int = _DEFAULT_STEPS_PER_EPOCH,
    workers: int = 4,
    pretrained: bool = True,
    on_epoch: Callable[[EpochMetrics], None] | None = None,
) -> tuple[KeyNet, list[EpochMetrics]]:
    device = _device()
    print(f"training device {device}", flush=True)
    stills = load_synth_keys(still_dir)
    if not stills:
        raise ValueError(f"no rendered frames found in {still_dir}")
    motion = load_synth_keys(motion_dir) if motion_dir is not None else []
    real = load_synth_keys(real_dir) if real_dir is not None else []
    still_train, still_val = _split(stills, seed)
    motion_train, motion_val = _split(motion, seed)
    real_train = [frame for frame in real if not _held_out(frame, hold_out)]
    real_val = [frame for frame in real if _held_out(frame, hold_out)]
    print(
        f"frames: {len(still_train)} still, {len(motion_train)} motion, {len(real_train)} real "
        f"to train on; {len(real_val)} real held out",
        flush=True,
    )
    sources = Sources(
        ((still_train, shares[0]), (motion_train, shares[1]), (real_train, shares[2]))
    )
    track = KeyNetDataset(sources, steps_per_epoch * batch, seed, False, recipe)
    search = KeyNetDataset(sources, steps_per_epoch * batch // _SEARCH_EVERY, seed, True, recipe)
    track_loader = DataLoader(track, batch_size=batch, num_workers=workers)
    search_loader = DataLoader(search, batch_size=batch, num_workers=workers)
    synthetic_val = still_val + motion_val

    torch.manual_seed(seed)
    if init_pt is not None:
        model = keynet_from_checkpoint(str(init_pt))
    elif init_onnx is not None:
        model = keynet_from_keyseg(str(init_onnx))
    else:
        model = KeyNet(pretrained=pretrained)
    model = model.to(device)
    # the encoder from a keyseg export carries folded statistics in its batch norms, and a batch
    # of crops would overwrite them, so those stay in eval mode while the rest trains
    frozen = (
        [module for module in model.features.modules() if isinstance(module, nn.BatchNorm2d)]
        if init_onnx is not None or init_pt is not None
        else []
    )
    if recipe.heads_only:
        for part in (model.features, model.ups):
            part.requires_grad_(False)
        frozen = [m for m in model.ups.modules() if isinstance(m, nn.BatchNorm2d)] + [
            m for m in model.features.modules() if isinstance(m, nn.BatchNorm2d)
        ]
    presence_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=max(steps_per_epoch * epochs, 1)
    )
    history: list[EpochMetrics] = []
    for epoch in range(epochs):
        track.set_epoch(epoch)
        search.set_epoch(epoch)
        model.train()
        for norm in frozen:
            norm.eval()
        track_batches = _batches(track_loader)
        search_batches = _batches(search_loader)
        running = 0.0
        for step in range(steps_per_epoch):
            crops, heat, weight, presence, offset, offset_weight = (
                tensor.to(device)
                for tensor in (
                    next(search_batches) if step % _SEARCH_EVERY == 0 else next(track_batches)
                )
            )
            optimizer.zero_grad(set_to_none=True)
            logits, said, predicted_offset = model(crops)
            loss = heatmap_loss(
                logits, heat, weight, recipe.single_peak
            ) + _PRESENCE_WEIGHT * presence_loss(said, presence)
            if recipe.offsets:
                loss = loss + _OFFSET_WEIGHT * offset_loss(predicted_offset, offset, offset_weight)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += float(loss.detach())
        synthetic = evaluate(model, synthetic_val, False, device, batch)
        held = (
            evaluate(
                model,
                real_val,
                False,
                device,
                batch,
                Recipe(fixed_dir=recipe.fixed_dir, rectified=recipe.rectified),
            )
            if real_val
            else None
        )
        metrics = EpochMetrics(epoch + 1, running / max(steps_per_epoch, 1), synthetic, held)
        history.append(metrics)
        real_text = (
            f" real p {held.peaks.precision:.3f} r {held.peaks.recall:.3f} "
            f"err {held.peaks.error_px:.2f}px"
            if held is not None
            else ""
        )
        print(
            f"epoch {metrics.epoch}/{epochs} loss {metrics.loss:.4f} synthetic p "
            f"{synthetic.peaks.precision:.3f} r {synthetic.peaks.recall:.3f} err "
            f"{synthetic.peaks.error_px:.2f}px presence {synthetic.presence_accuracy:.3f}"
            f"{real_text}",
            flush=True,
        )
        if on_epoch is not None:
            on_epoch(metrics)
    return model, history


def _require_mlflow() -> ModuleType:
    if mlflow is None:
        raise RuntimeError("mlflow is not installed; add it to the tools dev dependency group")
    return mlflow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="train KeyNet on keyboard keypoints")
    parser.add_argument("--still-dir", type=Path, required=True)
    parser.add_argument("--motion-dir", type=Path)
    parser.add_argument("--real-dir", type=Path)
    parser.add_argument("--shares", default="0.5,0.3,0.2", help="still,motion,real draw shares")
    parser.add_argument("--hold-out", default="", help="comma separated real clips kept out")
    parser.add_argument("--init-onnx", type=Path, help="start the encoder from this keyseg export")
    parser.add_argument(
        "--init-pt", type=Path, help="start from this KeyNet checkpoint, growing its heatmap head"
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=_DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--low-res-share", type=float, default=0.0, help="share of samples shown at low res"
    )
    parser.add_argument(
        "--mask-real-tops", action="store_true", help="leave real frames' guessed tops out"
    )
    parser.add_argument(
        "--single-peak", action="store_true", help="one positive heatmap cell per point"
    )
    parser.add_argument("--offsets", action="store_true", help="train the sub-cell offset head")
    parser.add_argument(
        "--heads-only", action="store_true", help="train only the output layers, encoder frozen"
    )
    parser.add_argument(
        "--rectified",
        action="store_true",
        help="crop tracking samples with the rectifying homography",
    )
    parser.add_argument(
        "--camera-sim",
        action="store_true",
        help="simulate other cameras on every input crop: noise, jpeg, blur, vignette, low res",
    )
    parser.add_argument(
        "--fixed-dir", type=Path, help="hand-corrected labels, one <frame stem>.json per frame"
    )
    parser.add_argument("--steps-per-epoch", type=int, default=_DEFAULT_STEPS_PER_EPOCH)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--mlflow", action="store_true", help="log params, metrics and artifacts")
    parser.add_argument(
        "--mlflow-experiment", default=os.environ.get("MLFLOW_EXPERIMENT_NAME", "keynet")
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    shares = tuple(float(share) for share in args.shares.split(","))
    if len(shares) != 3:
        raise ValueError("--shares takes three numbers: still,motion,real")
    tracker = _require_mlflow() if args.mlflow else None
    if tracker is not None:
        tracker.set_experiment(args.mlflow_experiment)
    run_context: AbstractContextManager[object] = (
        tracker.start_run() if tracker is not None else nullcontext()
    )
    with run_context:
        if tracker is not None:
            tracker.log_params({key: str(value) for key, value in vars(args).items()})

        def log_epoch(metrics: EpochMetrics) -> None:
            if tracker is None:
                return
            logged = {
                "loss": metrics.loss,
                "synthetic_precision": metrics.synthetic.peaks.precision,
                "synthetic_recall": metrics.synthetic.peaks.recall,
                "synthetic_error_px": metrics.synthetic.peaks.error_px,
                "synthetic_presence": metrics.synthetic.presence_accuracy,
            }
            if metrics.real is not None:
                logged |= {
                    "real_precision": metrics.real.peaks.precision,
                    "real_recall": metrics.real.peaks.recall,
                    "real_error_px": metrics.real.peaks.error_px,
                }
            tracker.log_metrics(logged, step=metrics.epoch)

        model, history = train_keynet(
            args.still_dir,
            motion_dir=args.motion_dir,
            real_dir=args.real_dir,
            shares=(shares[0], shares[1], shares[2]),
            hold_out=tuple(clip for clip in args.hold_out.split(",") if clip),
            init_onnx=args.init_onnx,
            init_pt=args.init_pt,
            recipe=Recipe(
                low_res_share=args.low_res_share,
                mask_real_tops=args.mask_real_tops,
                single_peak=args.single_peak,
                offsets=args.offsets,
                fixed_dir=args.fixed_dir,
                rectified=args.rectified,
                camera_sim=args.camera_sim,
                heads_only=args.heads_only,
            ),
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            steps_per_epoch=args.steps_per_epoch,
            workers=args.workers,
            on_epoch=log_epoch,
        )
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pt_path = args.out_dir / "keynet.pt"
        onnx_path = args.out_dir / "keynet.onnx"
        torch.save(model.state_dict(), pt_path)
        export_keynet_onnx(model.cpu(), str(onnx_path))
        last = history[-1]
        for stem, value in (
            ("synthetic_recall", last.synthetic.peaks.recall),
            ("synthetic_error_px", last.synthetic.peaks.error_px),
            ("real_recall", last.real.peaks.recall if last.real is not None else 0.0),
        ):
            (args.out_dir / f"{stem}.txt").write_text(f"{value:.6f}")
        if tracker is not None:
            tracker.log_artifact(str(pt_path))
            tracker.log_artifact(str(onnx_path))
    print(f"saved {pt_path}")


if __name__ == "__main__":
    main()
