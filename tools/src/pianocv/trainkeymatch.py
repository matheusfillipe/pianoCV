"""Strip augmentation shared by the KeyNet trainer."""

import cv2
import numpy as np


def _draw_blob(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
    axes = (int(rng.integers(6, 40)), int(rng.integers(4, 16)))
    angle = float(rng.uniform(0.0, 180.0))
    dark = rng.random() < 0.5
    shade = int(rng.integers(0, 40))
    color = (
        (shade, shade, shade)
        if dark
        else (
            int(rng.integers(180, 230)),
            int(rng.integers(120, 180)),
            int(rng.integers(100, 150)),
        )
    )
    overlay = image.copy()
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
    alpha = rng.uniform(0.4, 0.85)
    cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0, dst=image)


def _draw_glare(image: np.ndarray, rng: np.random.Generator) -> None:
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)
    x0, x1 = int(rng.integers(0, width)), int(rng.integers(0, width))
    thickness = int(rng.integers(2, 8))
    cv2.line(overlay, (x0, 0), (x1, height), (255, 255, 255), thickness)
    alpha = rng.uniform(0.2, 0.5)
    cv2.addWeighted(overlay, alpha, image, 1.0, 0.0, dst=image)


def _to_uint8(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(values, 0.0, 255.0), dtype=np.uint8)


def _augment_strip(strip_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    working = strip_rgb.astype(np.float32) * rng.uniform(0.7, 1.3) + rng.uniform(-20.0, 20.0)
    gamma = rng.uniform(0.7, 1.4)
    working = 255.0 * (np.clip(working, 0.0, 255.0) / 255.0) ** gamma
    working = working + rng.uniform(-12.0, 12.0, size=3)
    image = _to_uint8(working)

    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 5]))
        image = np.asarray(cv2.GaussianBlur(image, (ksize, ksize), 0), dtype=np.uint8)
    if rng.random() < 0.3:
        length = int(rng.integers(4, 12))
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0
        matrix = cv2.getRotationMatrix2D(
            (length / 2, length / 2), float(rng.uniform(0.0, 180.0)), 1.0
        )
        kernel = np.asarray(cv2.warpAffine(kernel, matrix, (length, length)), dtype=np.float32)
        kernel /= max(float(kernel.sum()), 1e-6)
        image = np.asarray(cv2.filter2D(image, -1, kernel), dtype=np.uint8)

    noise = rng.normal(scale=rng.uniform(2.0, 10.0), size=image.shape)
    image = _to_uint8(image.astype(np.float32) + noise)

    if rng.random() < 0.5:
        quality = int(rng.integers(35, 90))
        ok, encoded = cv2.imencode(
            ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else None
        if decoded is not None:
            image = np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)

    for _ in range(int(rng.integers(0, 3))):
        _draw_blob(image, rng)
    if rng.random() < 0.3:
        _draw_glare(image, rng)
    return image
