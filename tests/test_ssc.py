from __future__ import annotations

import numpy as np

from iqa_ssc.ssc import LAMBDA_GRID, corrected_score, fit_moments, pair_auc, reversal_indicator, select_lambda


def _pair(image_id: str, selective: float, control: float, sc_s: float, sc_c: float) -> dict[str, object]:
    return {"image_id": image_id, "selective_sc": sc_s, "control_sc": sc_c, "selective": {"scores": {"brisque": selective}}, "control": {"scores": {"brisque": control}}}


def test_lambda_grid_is_frozen_and_reversal_direction_is_explicit() -> None:
    assert LAMBDA_GRID[0] == 0.0 and LAMBDA_GRID[-1] == 2.0 and len(LAMBDA_GRID) == 41
    assert reversal_indicator(2.0, 1.0) == 0.0
    assert reversal_indicator(1.0, 2.0) == 1.0
    assert reversal_indicator(1.0, 1.0) == 0.5


def test_select_lambda_uses_calibration_rows_and_smallest_tie_break() -> None:
    rows = [_pair(str(i), 1.0 + i * 0.01, 2.0 + i * 0.01, i * 0.01, i * 0.01) for i in range(20)]
    selected, objectives, moments, sc_moments = select_lambda(rows, "brisque", higher_is_worse=True)
    assert selected == 0.0
    assert len(objectives) == 41
    assert moments.std > 0 and sc_moments.std > 0
    assert corrected_score(2.0, 0.0, moments=moments, sc_moments=fit_moments(np.arange(20)), lam=0.0) == (2.0 - moments.mean) / moments.std


def test_pair_auc_is_larger_when_selective_scores_are_worse() -> None:
    assert pair_auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
