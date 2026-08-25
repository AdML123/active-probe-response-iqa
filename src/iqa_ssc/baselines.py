"""Locked baseline metric adapters; no-reference and paired APIs are separate."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


NO_REFERENCE_METRICS = ("brisque", "niqe", "piqe")
PAIRED_METRICS = ("lpips",)


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
