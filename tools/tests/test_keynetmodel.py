from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from pianocv.keynet import CHANNELS, SEARCH_SIZE, STRIDE, TRACK_HEIGHT, TRACK_WIDTH
from pianocv.keynetmodel import (
    KeyNet,
    export_keynet_onnx,
    keynet_from_checkpoint,
    keynet_from_keyseg,
)
from pianocv.keyseg import KeySegNet, export_keyseg_onnx


def test_heatmaps_come_at_stride_two_in_both_modes() -> None:
    model = KeyNet(pretrained=False).eval()
    with torch.no_grad():
        for height, width in ((TRACK_HEIGHT, TRACK_WIDTH), (SEARCH_SIZE, SEARCH_SIZE)):
            heat, presence, offset = model(torch.zeros(1, 3, height, width))
            assert offset.shape == (1, 2 * CHANNELS, height // STRIDE, width // STRIDE)
            assert heat.shape == (1, CHANNELS, height // STRIDE, width // STRIDE)
            assert presence.shape == (1, 1)


def test_heatmaps_start_out_predicting_almost_nothing() -> None:
    model = KeyNet(pretrained=False).eval()
    with torch.no_grad():
        heat, _, _ = model(torch.zeros(1, 3, TRACK_HEIGHT, TRACK_WIDTH))
    assert float(torch.sigmoid(heat).mean()) < 0.05


def test_one_export_serves_the_track_crop_and_the_search_frame(tmp_path: Path) -> None:
    path = tmp_path / "keynet.onnx"
    export_keynet_onnx(KeyNet(pretrained=False), str(path))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for height, width in ((TRACK_HEIGHT, TRACK_WIDTH), (SEARCH_SIZE, SEARCH_SIZE)):
        heat, presence, offset = session.run(
            None, {"image": np.zeros((1, 3, height, width), np.float32)}
        )
        assert offset.shape == (1, 2 * CHANNELS, height // STRIDE, width // STRIDE)
        assert heat.shape == (1, CHANNELS, height // STRIDE, width // STRIDE)
        assert 0.0 <= float(presence[0, 0]) <= 1.0


def test_the_encoder_starts_from_a_keyseg_export(tmp_path: Path) -> None:
    torch.manual_seed(0)
    keyseg = KeySegNet(pretrained=False)
    path = tmp_path / "keyseg.onnx"
    export_keyseg_onnx(keyseg, str(path))
    model = keynet_from_keyseg(str(path)).eval()
    crop = torch.randn(1, 3, TRACK_HEIGHT, TRACK_WIDTH)
    with torch.no_grad():
        ours = model.features(crop)
        theirs = keyseg.eval().features(crop)
    assert float((ours - theirs).abs().max()) < 1e-3


def test_a_checkpoint_with_fewer_channels_keeps_its_heatmaps_and_grows_the_rest(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    saved = KeyNet(pretrained=False)
    saved.heat = torch.nn.Conv2d(saved.heat.in_channels, 7, 1)
    path = tmp_path / "keynet.pt"
    torch.save(saved.state_dict(), path)
    model = keynet_from_checkpoint(str(path)).eval()
    crop = torch.randn(1, 3, TRACK_HEIGHT, TRACK_WIDTH)
    with torch.no_grad():
        ours, _, offset = model(crop)
        theirs, _, _ = saved.eval()(crop)
    assert float(offset.abs().max()) == 0.0
    assert ours.shape[1] == CHANNELS
    assert float((ours[:, :7] - theirs).abs().max()) < 1e-5
    assert float((ours[:, 7:9] - theirs[:, 5:7]).abs().max()) < 1e-5
    assert float((ours[:, 9] - theirs[:, 4]).abs().max()) < 1e-5
    assert float((ours[:, 10:] - theirs[:, 5:7]).abs().max()) < 1e-5


def test_a_nine_channel_checkpoint_grows_the_back_channels_from_their_front_twins(
    tmp_path: Path,
) -> None:
    torch.manual_seed(0)
    saved = KeyNet(pretrained=False)
    saved.heat = torch.nn.Conv2d(saved.heat.in_channels, 9, 1)
    path = tmp_path / "keynet.pt"
    torch.save(saved.state_dict(), path)
    grown = keynet_from_checkpoint(str(path)).state_dict()
    old = saved.state_dict()
    assert grown["heat.weight"].shape[0] == 12
    for channel, source in ((9, 4), (10, 7), (11, 8)):
        assert torch.equal(grown["heat.weight"][channel], old["heat.weight"][source])
        assert torch.equal(grown["heat.bias"][channel], old["heat.bias"][source])
    assert torch.equal(grown["heat.weight"][:9], old["heat.weight"])
