from __future__ import annotations

import numpy as np
import pytest

from iqa_ssc.baselines import dct_high_frequency_ratio, local_texture_statistics, normalize_score, score_metric


def test_normalize_score_locks_higher_is_worse_direction() -> None:
    assert normalize_score(3.0, higher_is_worse=True) == 3.0
    assert normalize_score(3.0, higher_is_worse=False) == -3.0


def test_normalize_rejects_nonfinite() -> None:
    with pytest.raises(ValueError, match="finite"):
        normalize_score(float("nan"), higher_is_worse=True)


def test_lpips_requires_reference_and_no_reference_rejects_reference() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="paired reference"):
        score_metric("lpips", image, device="cpu")
    with pytest.raises(ValueError, match="do not accept"):
        score_metric("brisque", image, reference=image, device="cpu")


def test_dct_and_texture_adapters_are_finite_and_deterministic() -> None:
    image = np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)
    ratio = dct_high_frequency_ratio(image)
    texture = local_texture_statistics(image)
    assert np.isfinite(ratio)
    assert texture.shape == (3,)
    assert np.all(np.isfinite(texture))
    assert ratio == dct_high_frequency_ratio(image)
    np.testing.assert_array_equal(texture, local_texture_statistics(image))


def test_adapters_reject_malformed_images() -> None:
    with pytest.raises(ValueError, match="shape"):
        dct_high_frequency_ratio(np.zeros((8, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="dtype"):
        local_texture_statistics(np.zeros((16, 16, 3), dtype=np.float32))

