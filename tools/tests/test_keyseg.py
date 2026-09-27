from pathlib import Path

import numpy as np
import torch

from pianocv.keymatch import SidecarKey, SynthKeysFrame
from pianocv.keyseg import (
    BLACK,
    BOUNDARY,
    CROP_HEIGHT,
    CROP_WIDTH,
    IGNORE,
    WHITE,
    KeySegNet,
    black_silhouette,
    class_iou,
    crop_for,
    label_map,
    preprocess_crop,
)

QUAD = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 300.0), (100.0, 300.0)])


def _face(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _frame(image_size: tuple[int, int] = (640, 480)) -> SynthKeysFrame:
    whites = [
        SidecarKey(
            pitch=60 + 2 * i,
            black=False,
            top=_face(100 + 100 * i, 200 + 100 * i, 200, 300),
            front=None,
        )
        for i in range(4)
    ]
    top = _face(180, 220, 200, 260)
    front = np.array([top[3], top[2], (220.0, 266.0), (180.0, 266.0)])
    black = SidecarKey(pitch=61, black=True, top=top, front=front)
    return SynthKeysFrame(Path("unused.png"), QUAD, [*whites, black], image_size)


def test_crop_puts_the_keys_left_to_right_across_the_crop_centre() -> None:
    crop = crop_for(QUAD)
    corners = crop.points(QUAD)
    assert np.allclose(corners.mean(axis=0), (CROP_WIDTH / 2, CROP_HEIGHT / 2))
    assert corners[1, 0] > corners[0, 0]
    assert np.isclose(corners[0, 1], corners[1, 1])


def test_black_silhouette_covers_the_top_and_the_drop_to_the_white_keys() -> None:
    outline = black_silhouette(_frame().keys[-1])
    ys = outline[:, 1]
    assert ys.min() == 200 and ys.max() == 266


def test_labels_mark_white_keys_black_keys_and_the_lines_between_white_keys() -> None:
    frame = _frame()
    crop = crop_for(QUAD)
    labels = label_map(frame, crop)

    def at(x: float, y: float) -> int:
        cx, cy = np.round(crop.points(np.array([[x, y]]))[0]).astype(int)
        return int(labels[cy, cx])

    assert at(150, 290) == WHITE
    assert at(200, 230) == BLACK
    assert at(300, 290) == BOUNDARY


def test_crop_pixels_outside_the_camera_frame_are_ignored() -> None:
    frame = _frame(image_size=(300, 480))
    crop = crop_for(QUAD)
    labels = label_map(frame, crop)
    x, y = np.round(crop.points(np.array([[450.0, 290.0]]))[0]).astype(int)
    assert labels[y, x] == IGNORE
    assert class_iou(labels, labels) == [1.0, 1.0, 1.0, 1.0]


def test_model_keeps_the_crop_resolution() -> None:
    model = KeySegNet(pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, CROP_HEIGHT, CROP_WIDTH))
    assert out.shape == (1, 4, CROP_HEIGHT, CROP_WIDTH)


def test_preprocess_gives_the_model_a_channels_first_crop() -> None:
    crop = np.zeros((CROP_HEIGHT, CROP_WIDTH, 3), dtype=np.uint8)
    assert preprocess_crop(crop).shape == (3, CROP_HEIGHT, CROP_WIDTH)
