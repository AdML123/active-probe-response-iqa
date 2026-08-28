"""Frozen calibration, ranking, AUC, and bootstrap utilities for Q_SSC."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

LAMBDA_GRID = tuple(round(index * 0.05, 2) for index in range(41))
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20260824


@dataclass(frozen=True)
class ScoreMoments:
    mean: float
    std: float


def reversal_indicator(selective: float, control: float) -> float:
    if selective > control:
        return 0.0
    if selective < control:
        return 1.0
    return 0.5


def _auc_fast(selective: np.ndarray, control: np.ndarray) -> float:
    positive = np.asarray(selective, dtype=float)
    negative = np.asarray(control, dtype=float)
    if positive.size == 0 or positive.size != negative.size:
        raise ValueError("AUC requires equally sized non-empty paired arrays")
    ordered = np.sort(negative)
    left = np.searchsorted(ordered, positive, side="left")
    right = np.searchsorted(ordered, positive, side="right")
    return float(np.mean((left + right) / (2.0 * negative.size)))


def pair_auc(selective: np.ndarray, control: np.ndarray) -> float:
    return _auc_fast(selective, control)


def fit_moments(values: list[float] | np.ndarray) -> ScoreMoments:
    array = np.asarray(values, dtype=float)
    if array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("calibration values must contain at least two finite scores")
    std = float(np.std(array, ddof=1))
    if std <= 0 or not np.isfinite(std):
        raise ValueError("calibration score standard deviation must be positive")
    return ScoreMoments(float(np.mean(array)), std)


def z_score(value: float, moments: ScoreMoments) -> float:
    return (float(value) - moments.mean) / moments.std


def directional_score(value: float, *, higher_is_worse: bool) -> float:
    score = float(value)
    if not np.isfinite(score):
        raise ValueError("metric score must be finite")
    return score if higher_is_worse else -score


def corrected_score(raw: float, sc: float, *, moments: ScoreMoments, sc_moments: ScoreMoments, lam: float) -> float:
    return z_score(raw, moments) + float(lam) * z_score(sc, sc_moments)


def bootstrap_mean_ci(values: np.ndarray, *, n_resamples: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("bootstrap values must be finite and non-empty")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_auc_difference(
    selective: np.ndarray,
    control: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    positive = np.asarray(selective, dtype=float)
    negative = np.asarray(control, dtype=float)
    if positive.size != negative.size or positive.size == 0:
        raise ValueError("bootstrap AUC arrays must be equally sized and non-empty")
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sample = rng.integers(0, positive.size, size=positive.size)
        differences[index] = _auc_fast(positive[sample], negative[sample]) - _auc_fast(positive, negative)
    return float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))


def select_lambda(
    rows: list[dict[str, object]],
    metric: str,
    *,
    higher_is_worse: bool,
) -> tuple[float, dict[str, float], ScoreMoments, ScoreMoments]:
    raw_values = []
    sc_values = []
    for row in rows:
        raw_values.extend([
            directional_score(float(row["selective"]["scores"][metric]), higher_is_worse=higher_is_worse),
            directional_score(float(row["control"]["scores"][metric]), higher_is_worse=higher_is_worse),
        ])
        sc_values.extend([float(row["selective_sc"]), float(row["control_sc"])])
    moments = fit_moments(raw_values)
    sc_moments = fit_moments(sc_values)
    objectives: dict[str, float] = {}
    for lam in LAMBDA_GRID:
        indicators = []
        for row in rows:
            selective = corrected_score(directional_score(float(row["selective"]["scores"][metric]), higher_is_worse=higher_is_worse), float(row["selective_sc"]), moments=moments, sc_moments=sc_moments, lam=lam)
            control = corrected_score(directional_score(float(row["control"]["scores"][metric]), higher_is_worse=higher_is_worse), float(row["control_sc"]), moments=moments, sc_moments=sc_moments, lam=lam)
            indicators.append(reversal_indicator(selective, control))
        objectives[f"{lam:.2f}"] = float(np.mean(indicators))
    minimum = min(objectives.values())
    selected = min(float(key) for key, value in objectives.items() if value == minimum)
    return selected, objectives, moments, sc_moments


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
