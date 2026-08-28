"""Wavelet patch features used by the spatial-selectivity protocol."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pywt


@dataclass(frozen=True)
class FeatureBatch:
    vectors: np.ndarray
    patch_origins: tuple[tuple[int, int], ...]
    energies: np.ndarray


def _to_gray(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    if values.ndim == 2:
        return values
    if values.ndim == 3 and values.shape[2] in (3, 4):
        return values[..., :3] @ np.array([0.299, 0.587, 0.114])
    raise ValueError("image must be a grayscale or RGB/RGBA array")


def _patch_features(patch: np.ndarray, wavelet: str, level: int, mode: str) -> tuple[np.ndarray, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Level value .* boundary effects")
        coeffs = pywt.wavedec2(patch, wavelet=wavelet, level=level, mode=mode)
    details = [band for level_details in coeffs[1:] for band in level_details]
    high_frequency = np.concatenate([band.reshape(-1) for band in details])
    vector = np.array([np.mean(high_frequency), np.std(high_frequency)], dtype=np.float64)
    energy = float(np.mean(np.square(high_frequency)))
    return vector, energy


def extract_patch_features(
    image: np.ndarray,
    region_mask: np.ndarray,
    *,
    patch_size: int = 16,
    wavelet: str = "db4",
    level: int = 2,
    mode: str = "symmetric",
    feature_indices: tuple[int, ...] = (0, 1),
) -> FeatureBatch:
    """Extract features from patches fully contained in ``region_mask``."""

    gray = _to_gray(image)
    mask = np.asarray(region_mask, dtype=bool)
    if gray.shape != mask.shape:
        raise ValueError(f"image and region_mask shapes differ: {gray.shape} != {mask.shape}")
    if patch_size <= 0 or level <= 0:
        raise ValueError("patch_size and level must be positive")
    if not feature_indices or any(index not in (0, 1) for index in feature_indices):
        raise ValueError("feature_indices must be a non-empty subset of (0, 1)")
    vectors: list[np.ndarray] = []
    energies: list[float] = []
    origins: list[tuple[int, int]] = []
    height, width = gray.shape
    for top in range(0, height - patch_size + 1, patch_size):
        for left in range(0, width - patch_size + 1, patch_size):
            if np.all(mask[top : top + patch_size, left : left + patch_size]):
                vector, energy = _patch_features(gray[top : top + patch_size, left : left + patch_size], wavelet, level, mode)
                if np.all(np.isfinite(vector)) and np.isfinite(energy) and energy >= 0:
                    vectors.append(vector[list(feature_indices)])
                    energies.append(energy)
                    origins.append((top, left))
    matrix = np.vstack(vectors) if vectors else np.empty((0, len(feature_indices)), dtype=np.float64)
    return FeatureBatch(matrix, tuple(origins), np.asarray(energies, dtype=np.float64))


def all_pixel_mask(shape: tuple[int, int]) -> np.ndarray:
    """Return an explicit all-pixel region mask for the global condition."""

    return np.ones(shape, dtype=bool)

