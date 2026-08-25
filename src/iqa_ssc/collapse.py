"""Locked covariance and paired regional collapse calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class InvalidCollapse(ValueError):
    """A region cannot produce a valid collapse score."""


@dataclass(frozen=True)
class LockedCovariance:
    covariance: np.ndarray
    inverse: np.ndarray
    mean: np.ndarray
    condition_number: float
    regularization: float


def _validate(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 20 or not np.all(np.isfinite(array)):
        raise InvalidCollapse("features must be finite with K >= 20")
    return array


def lock_pre_covariance(pre_features: np.ndarray, *, condition_limit: float = 1000.0) -> LockedCovariance:
    values = _validate(pre_features)
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if covariance.shape != (values.shape[1], values.shape[1]) or not np.all(np.isfinite(covariance)):
        raise InvalidCollapse("pre covariance is not finite or has the wrong shape")
    raw_condition = float(np.linalg.cond(covariance))
    regularization = 0.0
    if not np.isfinite(raw_condition) or raw_condition > condition_limit:
        regularization = 0.01 * float(np.trace(covariance)) / values.shape[1]
        covariance = covariance + regularization * np.eye(values.shape[1])
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError as exc:
        raise InvalidCollapse("regularized pre covariance remains singular") from exc
    return LockedCovariance(
        covariance=covariance,
        inverse=inverse,
        mean=values.mean(axis=0),
        condition_number=float(np.linalg.cond(covariance)),
        regularization=regularization,
    )


def lths(features: np.ndarray, locked: LockedCovariance) -> float:
    values = _validate(features)
    if values.shape[1] != locked.inverse.shape[0]:
        raise InvalidCollapse("feature dimension differs from locked covariance")
    centered = values - values.mean(axis=0)
    distances = np.einsum("ki,ij,kj->k", centered, locked.inverse, centered)
    score = float(np.mean(distances))
    if not np.isfinite(score) or score <= 0:
        raise InvalidCollapse("LTHS is non-positive or non-finite")
    return score


def delta(pre_lths: float, post_lths: float) -> float:
    if not np.isfinite(pre_lths) or not np.isfinite(post_lths) or pre_lths <= 0:
        raise InvalidCollapse("pre LTHS must be finite and positive")
    value = float(1.0 - post_lths / pre_lths)
    if not np.isfinite(value):
        raise InvalidCollapse("Delta is non-finite")
    return value


def selectivity_contrast(delta_skin: float, delta_global: float) -> float:
    value = float(delta_skin - delta_global)
    if not np.isfinite(value):
        raise InvalidCollapse("SC is non-finite")
    return value

