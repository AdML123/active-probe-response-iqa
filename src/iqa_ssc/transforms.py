"""Deterministic image transformations for the IQA-SSC protocol."""

from __future__ import annotations

import cv2
import numpy as np

BILATERAL_LEVELS = ((10, 0.10), (25, 0.15), (50, 0.20), (75, 0.25), (100, 0.30))
JPEG_QUALITIES = (10, 20, 30, 40, 50)
BLUR_SIGMAS = (1, 2, 3, 4, 5)


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if np.issubdtype(values.dtype, np.floating):
        values = np.rint(np.clip(values, 0.0, 1.0) * 255.0 if values.max() <= 1.0 else np.clip(values, 0.0, 255.0))
    return np.clip(values, 0, 255).astype(np.uint8)


def global_bilateral(image: np.ndarray, *, sigma_space: float, sigma_color: float, diameter: int = 9) -> np.ndarray:
    rgb = _as_uint8_rgb(image)
    normalized = rgb.astype(np.float32) / 255.0
    filtered = cv2.bilateralFilter(normalized, diameter, sigma_color, sigma_space)
    return np.rint(np.clip(filtered, 0.0, 1.0) * 255.0).astype(np.uint8)


def jpeg_roundtrip(image: np.ndarray, *, quality: int) -> np.ndarray:
    if quality not in JPEG_QUALITIES:
        raise ValueError(f"quality must be one of {JPEG_QUALITIES}")
    rgb = _as_uint8_rgb(image)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("JPEG decoding failed")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def gaussian_blur(image: np.ndarray, *, sigma: int) -> np.ndarray:
    if sigma not in BLUR_SIGMAS:
        raise ValueError(f"sigma must be one of {BLUR_SIGMAS}")
    rgb = _as_uint8_rgb(image)
    return cv2.GaussianBlur(rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)


def apply_condition(image: np.ndarray, condition: str, index: int) -> np.ndarray:
    if condition == "original":
        return _as_uint8_rgb(image).copy()
    if condition == "bilateral":
        sigma_space, sigma_color = BILATERAL_LEVELS[index]
        return global_bilateral(image, sigma_space=sigma_space, sigma_color=sigma_color)
    if condition == "jpeg":
        return jpeg_roundtrip(image, quality=JPEG_QUALITIES[index])
    if condition == "gaussian_blur":
        return gaussian_blur(image, sigma=BLUR_SIGMAS[index])
    raise ValueError(f"unknown condition: {condition}")

