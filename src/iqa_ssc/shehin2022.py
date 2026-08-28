"""A disclosed Shehin--Sankar frequency-residual ABF adaptation.

The source paper studies an adaptive bilateral filter.  The IQA-SSC corpus
uses a fixed OpenCV bilateral operator, so this module applies the published
frequency-ratio construction to that operator and records the mismatch.
"""

from __future__ import annotations

import cv2
import numpy as np

from .transforms import global_bilateral


def _gray_float(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 3:
        values = cv2.cvtColor(values.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if values.ndim != 2:
        raise ValueError("image must be HxW or HxWx3")
    return values.astype(np.float32) / 255.0


def dct_magnitude(image: np.ndarray) -> float:
    """Return the sum of absolute 2-D DCT coefficients (Eq. (8))."""

    gray = _gray_float(image)
    value = float(np.abs(cv2.dct(gray)).sum())
    if not np.isfinite(value) or value <= 0:
        raise ValueError("DCT magnitude must be finite and positive")
    return value


def successive_abf_images(image: np.ndarray, levels: int = 2) -> list[np.ndarray]:
    """Generate M_0 ... M_levels with the frozen bilateral adaptation."""

    if levels < 2:
        raise ValueError("at least two successive images are required")
    current = np.asarray(image)
    images = [current.astype(np.uint8, copy=True)]
    for _ in range(levels):
        current = global_bilateral(current, sigma_space=10.0, sigma_color=0.10, diameter=9)
        images.append(current)
    return images


def successive_abf_residuals(image: np.ndarray, levels: int = 2) -> np.ndarray:
    """Return d_i values from repeated fixed bilateral filtering."""

    values = np.asarray([dct_magnitude(item) for item in successive_abf_images(image, levels)])
    if not np.all(np.isfinite(values)):
        raise ValueError("non-finite residual sequence")
    return values


def ratio_features(residuals: np.ndarray) -> tuple[float, float]:
    """Return alpha=d1/d0 and beta=d2/d1 from Eqs. (9)--(10)."""

    values = np.asarray(residuals, dtype=float).reshape(-1)
    if values.size < 3 or not np.all(np.isfinite(values)) or np.any(values[:3] <= 0):
        raise ValueError("three positive residuals are required")
    return float(values[1] / values[0]), float(values[2] / values[1])


def feature_d(image: np.ndarray) -> float:
    """Return the signed D feature from Eq. (11)."""

    alpha, beta = ratio_features(successive_abf_residuals(image, levels=2))
    return float(1000.0 * np.log(alpha / beta))


def feature_dm(image: np.ndarray) -> float:
    """Return the modified D_m feature from Eq. (14)."""

    images = successive_abf_images(image, levels=2)
    d0, d1, d2 = [dct_magnitude(item) for item in images]
    blurred_m1 = cv2.GaussianBlur(images[1], (0, 0), sigmaX=1.0, sigmaY=1.0)
    d1b = dct_magnitude(blurred_m1)
    if min(d0, d1, d2, d1b) <= 0:
        raise ValueError("invalid modified residual")
    return float(1000.0 * np.log((d1 / d0) / (d2 / d1b)))


def abf_evidence(value: float) -> float:
    """Orient the published detector polarity as larger-is-filtered evidence."""

    value = float(value)
    if not np.isfinite(value):
        raise ValueError("feature must be finite")
    return -value
