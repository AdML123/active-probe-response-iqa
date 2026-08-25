"""Locked baseline metric adapters; no-reference and paired APIs are separate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


NO_REFERENCE_METRICS = ("brisque", "niqe", "piqe")
PAIRED_METRICS = ("lpips",)


def _validate_rgb_uint8(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")
    if values.dtype != np.uint8:
        raise ValueError("image must have dtype uint8")
    return values


def dct_high_frequency_ratio(image: np.ndarray, *, block_size: int = 8) -> float:
    """Return a deterministic DCT high-frequency energy ratio.

    This is a transparent adapter for DCT-domain smoothing-forensics methods;
    it is not presented as an exact reimplementation of any cited paper.
    """

    import cv2

    values = _validate_rgb_uint8(image)
    if block_size != 8:
        raise ValueError("block_size is frozen at 8")
    gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height = (gray.shape[0] // block_size) * block_size
    width = (gray.shape[1] // block_size) * block_size
    if height < block_size or width < block_size:
        raise ValueError("image is smaller than one DCT block")
    cropped = gray[:height, :width]
    blocks = cropped.reshape(height // block_size, block_size, width // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    coordinate = np.arange(block_size, dtype=float)
    basis = np.cos(np.pi * (2.0 * coordinate[:, None] + 1.0) * coordinate[None, :] / (2.0 * block_size))
    basis[:, 0] *= np.sqrt(1.0 / block_size)
    basis[:, 1:] *= np.sqrt(2.0 / block_size)
    coeff = np.einsum("ij,bjk,lk->bil", basis, blocks, basis, optimize=True)
    energy = coeff * coeff
    total = np.sum(energy[:, 1:, :], axis=(1, 2))
    high_mask = np.add.outer(np.arange(block_size), np.arange(block_size)) >= 4
    high = np.sum(energy[:, high_mask], axis=1)
    ratios = high[total > 1e-12] / total[total > 1e-12]
    if ratios.size == 0:
        raise ValueError("no valid DCT blocks")
    value = float(np.median(np.asarray(ratios, dtype=float)))
    if not np.isfinite(value):
        raise ValueError("non-finite DCT ratio")
    return value


def local_texture_statistics(image: np.ndarray, *, block_size: int = 8) -> np.ndarray:
    """Return fixed local texture statistics for a passive texture baseline."""

    import cv2

    values = _validate_rgb_uint8(image)
    if block_size != 8:
        raise ValueError("block_size is frozen at 8")
    gray = cv2.cvtColor(values, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    height = (gray.shape[0] // block_size) * block_size
    width = (gray.shape[1] // block_size) * block_size
    if height < block_size or width < block_size:
        raise ValueError("image is smaller than one texture block")
    blocks = gray[:height, :width].reshape(height // block_size, block_size, width // block_size, block_size)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, block_size, block_size)
    local_std = blocks.std(axis=(1, 2), ddof=0)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    magnitude = np.abs(laplacian[:height, :width]).reshape(-1, block_size, block_size).mean(axis=(1, 2))
    result = np.asarray([np.mean(local_std), np.std(local_std), np.mean(magnitude)], dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError("non-finite texture statistics")
    return result


@dataclass(frozen=True)
class Direction:
    metric: str
    higher_is_worse: bool
    check_n: int
    original_mean: float
    severe_mean: float


def normalize_score(raw_score: float, *, higher_is_worse: bool) -> float:
    value = float(raw_score)
    if not np.isfinite(value):
        raise ValueError("metric score must be finite")
    return value if higher_is_worse else -value


@lru_cache(maxsize=None)
def _metric_instance(metric_name: str, device: str):
    import pyiqa

    return pyiqa.create_metric(metric_name, device=device)


def score_metric(metric_name: str, image: np.ndarray, *, reference: np.ndarray | None = None, device: str = "cuda") -> float:
    """Score an RGB uint8 image with pyiqa, enforcing paired LPIPS semantics."""

    metric_name = metric_name.lower()
    if metric_name not in NO_REFERENCE_METRICS + PAIRED_METRICS:
        raise ValueError(f"unknown metric: {metric_name}")
    if metric_name in PAIRED_METRICS and reference is None:
        raise ValueError("LPIPS requires a paired reference image")
    if metric_name in NO_REFERENCE_METRICS and reference is not None:
        raise ValueError("no-reference metrics do not accept a reference image")
    import torch
    import pyiqa

    def tensor(values: np.ndarray):
        array = np.asarray(values)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("images must have shape HxWx3")
        return torch.from_numpy(array.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)

    metric = _metric_instance(metric_name, device)
    with torch.inference_mode():
        if reference is None:
            value = metric(tensor(image))
        else:
            value = metric(tensor(image), tensor(reference))
    score = float(value.detach().cpu().reshape(-1)[0])
    if not np.isfinite(score):
        raise ValueError(f"{metric_name} returned a non-finite score")
    return score


def score_metric_batch(metric_name: str, images: list[np.ndarray], *, device: str = "cuda") -> list[float]:
    """Score a batch of RGB uint8 images for no-reference metrics."""

    metric_name = metric_name.lower()
    if metric_name not in NO_REFERENCE_METRICS:
        raise ValueError("batch scoring is only defined for no-reference metrics")
    if not images:
        return []
    import torch

    arrays = [_validate_rgb_uint8(image) for image in images]
    shapes = {tuple(array.shape) for array in arrays}
    if len(shapes) != 1:
        raise ValueError("all batch images must have the same shape")
    batch = torch.from_numpy(np.stack([array.astype(np.float32).transpose(2, 0, 1) / 255.0 for array in arrays], axis=0))
    metric = _metric_instance(metric_name, device)
    with torch.inference_mode():
        values = metric(batch)
    scores = [float(value) for value in values.detach().cpu().reshape(-1)]
    if len(scores) != len(images) or not np.all(np.isfinite(scores)):
        raise ValueError(f"{metric_name} returned invalid batch scores")
    return scores
