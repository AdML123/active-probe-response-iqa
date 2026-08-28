"""Pre-registered orthogonalization for the v2 SSC score."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .ssc import LAMBDA_GRID, ScoreMoments, directional_score, fit_moments, reversal_indicator, z_score


def fit_uniform_beta(records: Iterable[dict[str, object]], *, min_records: int = 100) -> dict[str, float | int]:
    """Fit no-intercept SC ~ beta * Delta_global on uniform controls only."""
    values: list[tuple[float, float]] = []
    for row in records:
        if str(row.get("condition")) not in {"jpeg", "gaussian_blur"}:
            continue
        if row.get("invalid_reason") is not None:
            continue
        sc = row.get("sc")
        delta_global = row.get("delta_global")
        if not isinstance(sc, (int, float)) or not isinstance(delta_global, (int, float)):
            continue
        sc_value = float(sc)
        delta_value = float(delta_global)
        if not math.isfinite(sc_value) or not math.isfinite(delta_value):
            continue
        values.append((sc_value, delta_value))
    if len(values) < min_records:
        raise ValueError(f"uniform calibration records {len(values)} < required {min_records}")
    numerator = sum(sc * delta for sc, delta in values)
    denominator = sum(delta * delta for _, delta in values)
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("uniform calibration denominator is zero or non-finite")
    beta = numerator / denominator
    residuals = [sc - beta * delta for sc, delta in values]
    residual_mean = sum(residuals) / len(residuals)
    residual_var = sum((residual - residual_mean) ** 2 for residual in residuals) / len(residuals)
    delta_mean = sum(delta for _, delta in values) / len(values)
    sc_mean = sum(sc for sc, _ in values) / len(values)
    covariance = sum((sc - sc_mean) * (delta - delta_mean) for sc, delta in values)
    sc_var = sum((sc - sc_mean) ** 2 for sc, _ in values)
    delta_var = sum((delta - delta_mean) ** 2 for _, delta in values)
    correlation = covariance / math.sqrt(sc_var * delta_var) if sc_var > 0 and delta_var > 0 else float("nan")
    return {
        "uniform_record_count": len(values),
        "numerator": float(numerator),
        "denominator": float(denominator),
        "beta": float(beta),
        "residual_mean": float(residual_mean),
        "residual_std": float(math.sqrt(residual_var)),
        "residual_max_abs": float(max(abs(value) for value in residuals)),
        "sc_delta_global_pearson_r": float(correlation),
    }


def orthogonalize_sc(sc: float, delta_global: float, beta: float) -> float:
    """Remove the locked uniform-severity component from the spatial contrast."""
    return float(sc) - float(beta) * float(delta_global)


def select_lambda_v2(rows: list[dict[str, object]], metric: str, *, higher_is_worse: bool) -> dict[str, object]:
    """Select the frozen-grid lambda using calibration bilateral/JPEG pairs only."""
    raw_values: list[float] = []
    sc_values: list[float] = []
    for row in rows:
        raw_values.extend([
            directional_score(float(row["selective"]["scores"][metric]), higher_is_worse=higher_is_worse),
            directional_score(float(row["control"]["scores"][metric]), higher_is_worse=higher_is_worse),
        ])
        sc_values.extend([float(row["selective_sc_orth"]), float(row["control_sc_orth"])])
    moments = fit_moments(raw_values)
    sc_moments = fit_moments(sc_values)
    objectives: dict[str, float] = {}
    for lam in LAMBDA_GRID:
        indicators = []
        for row in rows:
            selective = z_score(float(row["selective"]["scores"][metric]), moments) if higher_is_worse else z_score(-float(row["selective"]["scores"][metric]), moments)
            control = z_score(float(row["control"]["scores"][metric]), moments) if higher_is_worse else z_score(-float(row["control"]["scores"][metric]), moments)
            selective += lam * z_score(float(row["selective_sc_orth"]), sc_moments)
            control += lam * z_score(float(row["control_sc_orth"]), sc_moments)
            indicators.append(reversal_indicator(selective, control))
        objectives[f"{lam:.2f}"] = float(np.mean(indicators))
    minimum = min(objectives.values())
    selected = min(float(key) for key, value in objectives.items() if value == minimum)
    return {
        "lambda": selected,
        "objective_min_reversal_rate": minimum,
        "objectives": objectives,
        "score_moments": moments.__dict__,
        "sc_orth_moments": sc_moments.__dict__,
        "pair_count": len(rows),
        "higher_is_worse": higher_is_worse,
    }
