from pathlib import Path

import numpy as np
import torch

from pianocv.model import (
    KeybedNet,
    corner_loss,
    decode_heatmaps,
    heatmap_targets,
    load_model,
    predict_corners,
    preprocess,
)
from pianocv.render import render_sample

FRAME_SIZE = (640.0, 480.0)


def test_forward_output_shapes() -> None:
    model = KeybedNet()
    model.eval()
    with torch.no_grad():
        heatmaps, present_logits = model(torch.zeros(2, 1, 288, 288))
        corners = decode_heatmaps(heatmaps)
    assert heatmaps.shape == (2, 4, 72, 72)
    assert present_logits.shape == (2,)
    assert corners.shape == (2, 8)
    assert bool((corners >= 0.0).all())
    assert bool((corners <= 1.0).all())


def test_parameter_count_under_1m() -> None:
    assert sum(p.numel() for p in KeybedNet().parameters()) < 1_000_000


def test_preprocess_resizes_and_normalizes() -> None:
    image = np.full((480, 640, 3), 200.0, dtype=np.float32)
    prepared = preprocess(image)
    assert prepared.shape == (288, 288)
    assert prepared.dtype == np.float32
    assert np.allclose(prepared, 200.0 / 255.0)


def _fixed_batch(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    samples = [render_sample(np.random.default_rng(seed)) for seed in range(count)]
    inputs = np.stack([preprocess(sample.image) for sample in samples])
    corners = np.stack([(sample.quad_px / np.array(FRAME_SIZE)).reshape(8) for sample in samples])
    present = np.array([1.0 if sample.present else 0.0 for sample in samples])
    return inputs, corners, present


def test_heatmap_targets_peak_at_clamped_corner_and_zero_when_absent() -> None:
    corners = torch.tensor([[0.0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.25, 0.75]])
    present = torch.tensor([0.0])
    targets = heatmap_targets(corners, present)
    assert targets.shape == (1, 4, 72, 72)
    assert float(targets.max()) == 0.0
    targets = heatmap_targets(corners, torch.ones(1))
    assert float(targets[0, 0].max()) <= 1.0
    y, x = divmod(int(targets[0, 0].argmax()), 72)
    assert (x, y) == (0, 71)
    y, x = divmod(int(targets[0, 2].argmax()), 72)
    assert (x, y) == (71, 0)


def test_decode_heatmaps_falls_back_to_center_when_map_is_empty() -> None:
    decoded = decode_heatmaps(torch.zeros(2, 4, 72, 72))
    assert torch.allclose(decoded, torch.full((2, 8), 0.5))


def test_decode_heatmaps_peaks_at_logit_maximum() -> None:
    heatmaps = torch.full((1, 4, 72, 72), -1e4)
    heatmaps[:, :, 20, 30] = 1e4
    decoded = decode_heatmaps(heatmaps)
    peak = torch.tensor([30.0 / 71.0, 20.0 / 71.0]).repeat(4)
    assert torch.allclose(decoded[0], peak, atol=1e-5)


def test_corner_loss_decreases_over_training_steps() -> None:
    inputs, corners, present = _fixed_batch(32)
    torch.manual_seed(0)
    model = KeybedNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    tensor = torch.from_numpy(inputs).unsqueeze(1)
    corner_tensor = torch.from_numpy(corners)
    flags = torch.from_numpy(present)
    target = heatmap_targets(corner_tensor, flags)
    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        pred_heatmaps, present_logits = model(tensor)
        loss = corner_loss(pred_heatmaps, present_logits, target, flags, corner_tensor)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_corner_loss_backpropagates_to_head_conv_weights() -> None:
    inputs, corners, _ = _fixed_batch(4)
    torch.manual_seed(0)
    model = KeybedNet()
    corner_tensor = torch.from_numpy(corners)
    pred_heatmaps, present_logits = model(torch.from_numpy(inputs).unsqueeze(1))
    loss = corner_loss(
        pred_heatmaps,
        present_logits,
        heatmap_targets(corner_tensor, torch.ones(4)),
        torch.ones(4),
        corner_tensor,
    )
    loss.backward()
    assert model.heatmap_head[0].weight.grad is not None
    assert model.heatmap_head[-1].weight.grad is not None


def test_predict_corners_scales_back_to_frame() -> None:
    sample = render_sample(np.random.default_rng(2))
    quad, probability = predict_corners(KeybedNet(), sample.image)
    assert quad.shape == (4, 2)
    assert 0.0 <= probability <= 1.0
    assert np.all(quad[:, 0] >= 0.0)
    assert np.all(quad[:, 0] <= FRAME_SIZE[0])
    assert np.all(quad[:, 1] >= 0.0)
    assert np.all(quad[:, 1] <= FRAME_SIZE[1])


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "keybed_net.pt"
    source = KeybedNet()
    torch.save(source.state_dict(), path)
    loaded = load_model(path)
    tensor = torch.zeros(1, 1, 288, 288)
    with torch.no_grad():
        source_heatmaps, source_present = source(tensor)
        loaded_heatmaps, loaded_present = loaded(tensor)
    assert torch.allclose(source_heatmaps, loaded_heatmaps)
    assert torch.allclose(source_present, loaded_present)
    assert not loaded.training
