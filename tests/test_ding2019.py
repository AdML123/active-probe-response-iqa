import numpy as np

from iqa_ssc.ding2019 import derivative_histogram, quantize_patch, texture_features


def test_quantization_matches_paper_equation() -> None:
    patch = np.array([[0, 3, 4, 7, 8]], dtype=np.uint8).repeat(5, axis=0)
    assert quantize_patch(patch)[0].tolist() == [0, 1, 1, 2, 2]


def test_texture_feature_dimension_and_diagnostics() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 32:] = 255
    values, diagnostics = texture_features(image)
    assert values.shape == (15,)
    assert np.all(np.isfinite(values))
    assert diagnostics["valid_patch_count"] > 0


def test_empty_edges_return_zero_histograms() -> None:
    first, second = derivative_histogram([])
    assert first.shape == (64,)
    assert second.shape == (64,)
    assert np.all(first == 0) and np.all(second == 0)
