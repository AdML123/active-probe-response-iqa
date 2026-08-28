import numpy as np

from iqa_ssc.shehin2022 import abf_evidence, ratio_features


def test_ratio_features_follow_equations_9_and_10() -> None:
    alpha, beta = ratio_features(np.array([1.0, 2.0, 4.0]))
    assert alpha == 2.0
    assert beta == 2.0


def test_abf_evidence_negates_published_polarity() -> None:
    assert abf_evidence(-3.0) == 3.0
    assert abf_evidence(2.0) == -2.0
