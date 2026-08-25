import pytest

from iqa_ssc.ssc_v2 import fit_uniform_beta, orthogonalize_sc, select_lambda_v2


def test_beta_is_no_intercept_and_ignores_non_uniform_rows():
    rows = [
        {"condition": "jpeg", "sc": 2.0, "delta_global": 1.0, "invalid_reason": None},
        {"condition": "gaussian_blur", "sc": 4.0, "delta_global": 2.0, "invalid_reason": None},
        {"condition": "bilateral", "sc": 100.0, "delta_global": 1.0, "invalid_reason": None},
    ]
    result = fit_uniform_beta(rows, min_records=2)
    assert result["beta"] == pytest.approx(2.0)
    assert result["uniform_record_count"] == 2
    assert orthogonalize_sc(5.0, 2.0, 2.0) == pytest.approx(1.0)


def test_beta_rejects_insufficient_records():
    with pytest.raises(ValueError, match="< required"):
        fit_uniform_beta([], min_records=1)


def test_beta_rejects_zero_denominator():
    rows = [{"condition": "jpeg", "sc": 1.0, "delta_global": 0.0, "invalid_reason": None}]
    with pytest.raises(ValueError, match="denominator"):
        fit_uniform_beta(rows, min_records=1)


def test_v2_lambda_uses_smallest_tie_break():
    rows = [
        {"selective": {"scores": {"brisque": 1.0}}, "control": {"scores": {"brisque": 2.0}}, "selective_sc_orth": 0.0, "control_sc_orth": 1.0},
        {"selective": {"scores": {"brisque": 2.0}}, "control": {"scores": {"brisque": 1.0}}, "selective_sc_orth": 1.0, "control_sc_orth": 0.0},
    ]
    result = select_lambda_v2(rows, "brisque", higher_is_worse=True)
    assert result["lambda"] == 0.0
