from __future__ import annotations

import numpy as np
import pytest

from iqa_ssc.collapse import InvalidCollapse, delta, lock_pre_covariance, lths, selectivity_contrast
from iqa_ssc.features import all_pixel_mask, extract_patch_features


def test_extracts_two_d_wavelet_features_from_skin_and_global_regions() -> None:
    image = np.arange(64 * 64, dtype=np.float64).reshape(64, 64)
    skin = np.ones((64, 64), dtype=bool)
    batch = extract_patch_features(image, skin)
    global_batch = extract_patch_features(image, all_pixel_mask(image.shape))
    assert batch.vectors.shape[1] == 2
    assert batch.vectors.shape[0] == global_batch.vectors.shape[0] == 16


def test_lock_covariance_uses_exact_regularization_for_ill_conditioned_data() -> None:
    base = np.linspace(0.0, 1.0, 20)
    features = np.column_stack([base, base + 1e-8 * np.arange(20)])
    locked = lock_pre_covariance(features)
    raw = np.atleast_2d(np.cov(features, rowvar=False, ddof=1))
    expected = 0.01 * np.trace(raw) / 2
    assert locked.regularization == pytest.approx(expected)
    assert np.isfinite(locked.condition_number)
    assert np.all(np.isfinite(locked.inverse))


def test_k_less_than_twenty_is_invalid_and_sc_is_difference() -> None:
    with pytest.raises(InvalidCollapse, match="K >= 20"):
        lock_pre_covariance(np.ones((19, 2)))
    assert delta(2.0, 1.0) == pytest.approx(0.5)
    assert selectivity_contrast(0.5, 0.1) == pytest.approx(0.4)


def test_lths_uses_locked_pre_inverse_not_post_covariance() -> None:
    rng = np.random.default_rng(4)
    pre = rng.normal(size=(40, 2))
    post = pre * np.array([0.5, 0.5])
    locked = lock_pre_covariance(pre)
    score = lths(post, locked)
    assert score > 0

