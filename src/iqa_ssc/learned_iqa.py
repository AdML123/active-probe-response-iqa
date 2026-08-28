"""Optional IQA-PyTorch adapter used by the conformance and audit scripts."""

from __future__ import annotations

from typing import Iterable


def import_pyiqa():
    """Import pyiqa while bridging its legacy CLIP packaging assumption."""
    import packaging
    from packaging import version
    import pkg_resources

    if not hasattr(packaging, "version"):
        packaging.version = version
    if not hasattr(pkg_resources, "packaging"):
        pkg_resources.packaging = packaging

    import pyiqa

    return pyiqa


def load_metric(name: str, *, device: str = "cuda"):
    """Construct a named pyiqa metric and return it with its score direction."""
    metric = import_pyiqa().create_metric(name, device=device)
    return metric, bool(getattr(metric, "lower_better"))


def score_in_batches(metric, images, *, device: str = "cuda", batch_size: int = 1) -> list[float]:
    """Score tensors in bounded batches to avoid high-resolution memory spikes."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    import torch

    scores: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            scores.extend(metric(batch).reshape(-1).detach().cpu().double().tolist())
            del batch
    return scores
