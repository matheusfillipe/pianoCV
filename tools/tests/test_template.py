import cv2
import numpy as np

from pianocv.template import STRIP_HEIGHT, STRIP_WIDTH, TEMPLATE, build_template, score

WIDTH = 640
HEIGHT = 480
CANONICAL_QUAD = np.array([[80.0, 120.0], [560.0, 100.0], [600.0, 380.0], [60.0, 400.0]])
STRIP_SOURCE = np.array([[0.0, 0.0], [519.0, 0.0], [519.0, 63.0], [0.0, 63.0]], dtype=np.float32)


def _canonical_image() -> np.ndarray:
    sheet = (TEMPLATE * 255.0).astype(np.uint8)
    matrix = cv2.getPerspectiveTransform(STRIP_SOURCE, CANONICAL_QUAD.astype(np.float32))
    warped = cv2.warpPerspective(
        cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR),
        matrix,
        (WIDTH, HEIGHT),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 40, 40),
    )
    return cv2.GaussianBlur(warped, (0, 0), 1.0)


def test_template_has_black_keys_only_on_the_back_half() -> None:
    back = TEMPLATE[: STRIP_HEIGHT // 3]
    front = TEMPLATE[-STRIP_HEIGHT // 3 :]
    assert float(back.min()) < 0.3
    assert float(front.min()) >= 0.5


def test_build_template_scales_the_key_count_to_any_strip_width() -> None:
    for width in (260, STRIP_WIDTH, 1040):
        template = build_template(width, STRIP_HEIGHT)
        assert template.shape == (STRIP_HEIGHT, width)
        seams = int((template[0] == 0.5).sum())
        assert seams > 0


def test_score_peaks_at_the_true_quad() -> None:
    image = _canonical_image()
    truth = score(image, CANONICAL_QUAD)
    assert truth > 0.8
    for shift in (8.0, -8.0):
        assert score(image, CANONICAL_QUAD + shift) < truth


def test_score_is_low_on_a_flat_image() -> None:
    flat = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
    assert abs(score(flat, CANONICAL_QUAD)) < 0.1


def test_score_accepts_a_custom_strip_size() -> None:
    image = _canonical_image()
    assert score(image, CANONICAL_QUAD, 260, 32) > 0.5
