import numpy as np
import pytest

from scripts.evaluate_iqa_ssc_trajectory_detector import (
    FEATURE_SUBSETS,
    auc,
    bootstrap_auc,
    feature_values,
    evaluate_rows,
    fit_lda,
    score,
    single_point_features,
    stratum_matches,
    validate_split,
)


def test_rank_auc_handles_ties_deterministically():
    assert auc(np.asarray([2.0, 2.0]), np.asarray([1.0, 2.0])) == pytest.approx(0.75)


def test_lda_uses_calibration_statistics_and_fixed_ridge():
    pos = np.asarray([[2.0, 0.0], [3.0, 0.0]])
    neg = np.asarray([[0.0, 2.0], [0.0, 3.0]])
    model = fit_lda(pos, neg)
    assert model["mean"].tolist() == [1.25, 1.25]
    assert model["std"].shape == (2,)
    assert model["regularization"] > 0
    assert score(model, np.asarray([[3.0, 0.0], [0.0, 3.0]])).shape == (2,)


def test_strata_are_pre_registered():
    assert stratum_matches("bilateral", 4, "severe")
    assert stratum_matches("jpeg", 2, "severe")
    assert stratum_matches("gaussian_blur", 5, "severe")
    assert stratum_matches("bilateral", 2, "mild")
    assert stratum_matches("jpeg", 5, "mild")
    assert stratum_matches("gaussian_blur", 1, "mild")
    assert not stratum_matches("bilateral", 3, "mild")


def test_bootstrap_auc_is_reproducible_with_explicit_resample_count():
    pos = np.asarray([0.8, 0.9, 1.0])
    neg = np.asarray([0.0, 0.1, 0.2])
    assert bootstrap_auc(pos, neg, seed=20260824, n_resamples=50) == bootstrap_auc(pos, neg, seed=20260824, n_resamples=50)


def test_validate_split_rejects_overlap():
    with pytest.raises(ValueError, match="overlap"):
        validate_split({"1.jpg", "2.jpg"}, {"2.jpg", "3.jpg"})


def test_ablation_feature_sets_are_fixed_and_single_point_uses_q30_sigma3():
    row = {"fixed_features_v1": list(range(12)), "goc_trajectory": list(range(10)), "s_grid_trajectory": list(range(10, 20))}
    assert set(FEATURE_SUBSETS) == {"jpeg_probe_6", "blur_probe_6", "goc_only_6", "s_grid_only_6"}
    assert feature_values(row, "jpeg_probe_6").tolist() == [0, 1, 2, 6, 7, 8]
    assert feature_values(row, "blur_probe_6").tolist() == [3, 4, 5, 9, 10, 11]
    assert single_point_features(row) == [2.0, 7.0, 12.0, 17.0]


def test_requesting_ablations_does_not_change_primary_report():
    rows = []
    for image_id, split in (("1.jpg", "calibration"), ("2.jpg", "calibration"), ("3.jpg", "evaluation"), ("4.jpg", "evaluation")):
        for family, shift in (("bilateral", 2.0), ("jpeg", 0.0), ("gaussian_blur", -1.0)):
            for index in range(1, 6):
                values = [shift + float(index)] * 12
                rows.append({
                    "image_id": image_id,
                    "base_family": family,
                    "base_index": index,
                    "invalid_reason": None,
                    "fixed_features_v1": values,
                    "goc_trajectory": [*values[:5], *values[:5]],
                    "s_grid_trajectory": [*values[:5], *values[:5]],
                })
    manifest = {"calibration_ids": ["1.jpg", "2.jpg"], "evaluation_ids": ["3.jpg", "4.jpg"]}
    primary = evaluate_rows(rows, manifest, n_resamples=5)
    with_ablations = evaluate_rows(rows, manifest, n_resamples=5, include_ablations=True)
    assert with_ablations["comparisons"] == primary["comparisons"]
