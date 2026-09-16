"""Train dense piano-key alignment evidence on Kaggle."""

import json
import random
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.segmentation import lraspp_mobilenet_v3_large

INPUT_SIZE = (320, 240)
BATCH_SIZE = 16
EPOCHS = 60
SEED = 7
_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def dataset_root() -> Path:
    source = next(Path("/kaggle/input").rglob("train.jsonl")).parent
    if (source / "images").is_dir():
        return source
    destination = Path("/kaggle/working/key-instance-data")
    destination.mkdir()
    for manifest in ("train.jsonl", "validation.jsonl"):
        shutil.copy(source / manifest, destination / manifest)
    for archive in source.rglob("*.zip"):
        with zipfile.ZipFile(archive) as contents:
            contents.extractall(destination)
    return destination


def load_rows(root: Path, name: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in (root / name).read_text().splitlines()]


class KeyDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path, entries: list[dict[str, object]], augment: bool) -> None:
        self.root = root
        self.entries = entries
        self.augment = augment

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.entries[index]
        image_name = item["image"]
        if not isinstance(image_name, str):
            raise ValueError("manifest image path must be text")
        image = cv2.imread(str(self.root / image_name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read {image_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        targets = [self._target(item, name) for name in ("visible", "black", "edge", "coordinate")]
        width, height = INPUT_SIZE
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        binary = [
            cv2.resize(target, (width, height), interpolation=cv2.INTER_AREA)
            for target in targets[:3]
        ]
        coordinate = cv2.resize(targets[3], (width, height), interpolation=cv2.INTER_NEAREST)
        floating = image.astype(np.float32) / 255.0
        if self.augment:
            floating = self._augment(floating)
        floating = (floating - _MEAN) / _STD
        stacked = np.stack(
            (*[target / 255.0 for target in binary], coordinate / 65535.0)
        ).astype(np.float32)
        return torch.from_numpy(floating.transpose(2, 0, 1)), torch.from_numpy(stacked)

    def _target(self, item: dict[str, object], name: str) -> np.ndarray:
        value = item[name]
        if not isinstance(value, str):
            raise ValueError(f"manifest {name} path must be text")
        target = cv2.imread(str(self.root / value), cv2.IMREAD_UNCHANGED)
        if target is None:
            raise ValueError(f"cannot read {value}")
        return target

    @staticmethod
    def _augment(image: np.ndarray) -> np.ndarray:
        gain = np.random.uniform(0.65, 1.35)
        bias = np.random.uniform(-0.12, 0.12)
        gamma = np.random.uniform(0.7, 1.4)
        noise = np.random.normal(0.0, 0.025, image.shape)
        return np.clip(np.clip(image * gain + bias, 0.0, 1.0) ** gamma + noise, 0.0, 1.0).astype(
            np.float32
        )


class AlignmentNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = lraspp_mobilenet_v3_large(
            weights=None,
            weights_backbone=MobileNet_V3_Large_Weights.DEFAULT,
            num_classes=4,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image)["out"]


def alignment_loss(logits: torch.Tensor, target: torch.Tensor, device: torch.device) -> torch.Tensor:
    visible = target[:, :1]
    visible_loss = functional.binary_cross_entropy_with_logits(
        logits[:, :1], target[:, :1], pos_weight=torch.tensor([3.0], device=device)
    )
    black_loss = functional.binary_cross_entropy_with_logits(
        logits[:, 1:2], target[:, 1:2], pos_weight=torch.tensor([8.0], device=device)
    )
    visible_probability = torch.sigmoid(logits[:, :1])
    dice = 1 - (2 * (visible_probability * visible).sum() + 1) / (
        visible_probability.sum() + visible.sum() + 1
    )
    edge_weight = torch.tensor([5.0], device=device)
    edges = functional.binary_cross_entropy_with_logits(
        logits[:, 2:3], target[:, 2:3], pos_weight=edge_weight
    )
    coordinate = functional.smooth_l1_loss(
        torch.sigmoid(logits[:, 3:4])[visible > 0.5], target[:, 3:4][visible > 0.5]
    )
    return visible_loss + black_loss + dice + edges + 3 * coordinate


def validation_score(
    model: AlignmentNet, loader: torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]], device: torch.device
) -> tuple[float, float]:
    model.eval()
    intersection = union = coordinate_error = coordinate_count = 0.0
    with torch.no_grad():
        for image, target in loader:
            target = target.to(device)
            logits = model(image.to(device))
            predicted = torch.sigmoid(logits[:, :1]) > 0.5
            truth = target[:, :1] > 0.5
            intersection += float((predicted & truth).sum())
            union += float((predicted | truth).sum())
            coordinate_error += float(
                (torch.sigmoid(logits[:, 3:4])[truth] - target[:, 3:4][truth]).abs().sum()
            )
            coordinate_count += float(truth.sum())
    return intersection / max(union, 1.0), coordinate_error / max(coordinate_count, 1.0)


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    root = dataset_root()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = KeyDataset(root, load_rows(root, "train.jsonl"), augment=True)
    validation = KeyDataset(root, load_rows(root, "validation.jsonl"), augment=False)
    train_loader = torch.utils.data.DataLoader(train, BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation, BATCH_SIZE, num_workers=4, pin_memory=True)
    model = AlignmentNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS * len(train_loader), 1e-6)
    output = Path("/kaggle/working")
    best = -1.0
    print(f"{device=}, train={len(train)}, validation={len(validation)}")
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for image, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = alignment_loss(model(image.to(device)), target.to(device), device)
            loss.backward()
            optimizer.step()
            schedule.step()
            total += float(loss.detach())
        iou, coordinate_error = validation_score(model, validation_loader, device)
        print(
            f"epoch {epoch + 1:02d}/{EPOCHS} loss {total / len(train_loader):.4f} "
            f"visible_iou {iou:.4f} coordinate_mae {coordinate_error:.5f}",
            flush=True,
        )
        if iou > best:
            best = iou
            torch.save(model.state_dict(), output / "piano_key_alignment.pt")
    model.load_state_dict(torch.load(output / "piano_key_alignment.pt", map_location=device))
    model.eval()
    example = torch.zeros(1, 3, INPUT_SIZE[1], INPUT_SIZE[0], device=device)
    torch.onnx.export(
        model,
        example,
        output / "piano_key_alignment.onnx",
        input_names=["image"],
        output_names=["alignment"],
        dynamic_axes={"image": {0: "batch"}, "alignment": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    print(f"best visible IoU: {best:.4f}")


if __name__ == "__main__":
    main()
