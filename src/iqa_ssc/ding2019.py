"""A disclosed adaptation of Ding et al.'s edge-texture smoothing feature.

The paper leaves several geometric choices implementation-dependent.  This
module freezes those choices so the feature can be audited on the IQA-SSC
protocol; it is not presented as the authors' original source code.
"""

from __future__ import annotations

import cv2
import numpy as np


def _gray_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim == 3 and values.shape[2] == 3:
        values = cv2.cvtColor(values.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if values.ndim != 2:
        raise ValueError("image must be HxW or HxWx3")
    return np.clip(values, 0, 255).astype(np.uint8)


def canny_edges(gray: np.ndarray, *, low: int = 100, high: int = 200) -> np.ndarray:
    """Return the frozen Canny edge map used by the adapted feature."""

    values = _gray_uint8(gray)
    if not 0 <= low < high:
        raise ValueError("Canny thresholds must satisfy 0 <= low < high")
    return cv2.Canny(values, low, high, apertureSize=3, L2gradient=True) > 0


def edge_centered_patches(gray: np.ndarray, edges: np.ndarray) -> list[np.ndarray]:
    """Extract non-overlapping 5x5 patches centered on edge pixels.

    The source paper specifies a central edge column but does not specify how
    curved edge geometry is linearized.  We use a deterministic vertical
    central column and scan in row-major order, skipping overlapping patches.
    """

    values = _gray_uint8(gray)
    edge_map = np.asarray(edges, dtype=bool)
    if edge_map.shape != values.shape:
        raise ValueError("edges and gray image must have the same shape")
    height, width = values.shape
    occupied = np.zeros_like(edge_map, dtype=bool)
    patches: list[np.ndarray] = []
    for y, x in zip(*np.nonzero(edge_map)):
        if y < 2 or y + 2 >= height or x < 2 or x + 2 >= width:
            continue
        window = np.s_[y - 2 : y + 3, x - 2 : x + 3]
        if occupied[window].any():
            continue
        patches.append(values[window].copy())
        occupied[window] = True
    return patches


def quantize_patch(patch: np.ndarray) -> np.ndarray:
    """Apply the four-level quantizer in Ding et al.'s Eq. (9)."""

    values = np.asarray(patch)
    if values.ndim != 2 or values.shape != (5, 5):
        raise ValueError("patch must have shape 5x5")
    if not np.all(np.isfinite(values)):
        raise ValueError("patch must be finite")
    return np.floor((values.astype(np.float64) + 1.0) / 4.0).astype(np.int16)


def derivative_histogram(patches: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized first- and second-derivative histograms.

    Histogram support is clipped to the 64 bins retained by the source paper;
    derivative signs are represented by absolute magnitudes in this adapter.
    """

    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for patch in patches:
        q = quantize_patch(patch)
        first.append(np.abs(q[1:, :] - q[:-1, :]).reshape(-1))
        second.append(np.abs(q[:, 2:] + q[:, :-2] - 2 * q[:, 1:-1]).reshape(-1))
    if not first:
        return np.zeros(64, dtype=float), np.zeros(64, dtype=float)
    h_first = np.bincount(np.clip(np.concatenate(first), 0, 63), minlength=64).astype(float)
    h_second = np.bincount(np.clip(np.concatenate(second), 0, 63), minlength=64).astype(float)
    h_first /= h_first.sum()
    h_second /= h_second.sum()
    return h_first, h_second


def texture_features(image: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    """Return Ding's 15-D texture vector and extraction diagnostics."""

    gray = _gray_uint8(image)
    edges = canny_edges(gray)
    patches = edge_centered_patches(gray, edges)
    h_first, h_second = derivative_histogram(patches)
    values = np.concatenate([h_first[:10], h_second[:5]]).astype(float)
    if values.shape != (15,) or not np.all(np.isfinite(values)):
        raise ValueError("non-finite Ding feature")
    return values, {
        "edge_pixel_count": int(edges.sum()),
        "valid_patch_count": len(patches),
        "edge_density": float(edges.mean()),
    }
