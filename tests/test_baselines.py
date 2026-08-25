from __future__ import annotations

import numpy as np
import pytest

from iqa_ssc.baselines import normalize_score, score_metric


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

