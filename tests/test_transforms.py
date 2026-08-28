from __future__ import annotations

import numpy as np
import pytest

from iqa_ssc.transforms import BILATERAL_LEVELS, BLUR_SIGMAS, JPEG_QUALITIES, apply_condition, jpeg_roundtrip


def test_protocol_grids_are_exact() -> None:
    assert BILATERAL_LEVELS == ((10, 0.10), (25, 0.15), (50, 0.20), (75, 0.25), (100, 0.30))
    assert JPEG_QUALITIES == (10, 20, 30, 40, 50)
    assert BLUR_SIGMAS == (1, 2, 3, 4, 5)


def test_transform_shapes_are_stable_and_original_is_not_mutated() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = [120, 80, 40]
    before = image.copy()
    for condition in ("bilateral", "jpeg", "gaussian_blur"):
        for index in range(5):
            output = apply_condition(image, condition, index)
            assert output.shape == image.shape
            assert output.dtype == np.uint8
    assert np.array_equal(image, before)


def test_jpeg_rejects_unlocked_quality() -> None:
    with pytest.raises(ValueError, match="quality"):
        jpeg_roundtrip(np.zeros((16, 16, 3), dtype=np.uint8), quality=25)

