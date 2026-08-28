from pathlib import Path

import cv2
import numpy as np
import pytest

from scripts.run_iqa_ssc_trajectory_detector import (
    BASE_GRID,
    PROBE_GRID,
    curve_stats,
    fixed_features,
    process_image,
)


def test_probe_grid_and_feature_order_are_locked():
    assert PROBE_GRID == (
        ("jpeg", 0),
        ("jpeg", 1),
        ("jpeg", 2),
        ("jpeg", 3),
        ("jpeg", 4),
        ("gaussian_blur", 0),
        ("gaussian_blur", 1),
        ("gaussian_blur", 2),
        ("gaussian_blur", 3),
        ("gaussian_blur", 4),
    )
    assert len(BASE_GRID) == 15
    assert curve_stats([0, 1, 3, 6, 10]) == [1.5, 3.5, 15.0]
    row = {"goc_trajectory": list(range(10)), "s_grid_trajectory": list(range(10, 20))}
    assert fixed_features(row) == [1.0, 1.0, 8.0, 1.0, 1.0, 28.0, 1.0, 1.0, 48.0, 1.0, 1.0, 68.0]


def test_fixed_features_reject_incomplete_trajectory():
    with pytest.raises(ValueError, match="five JPEG and five blur"):
        fixed_features({"goc_trajectory": [1.0], "s_grid_trajectory": list(range(10))})


def test_each_base_reuses_one_edge_mask_for_all_probe_values(tmp_path):
    image = np.random.default_rng(7).integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
    path = tmp_path / "1.jpg"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    rows = process_image(path.name, tmp_path)

    assert len(rows) == 15
    assert all(len(row["goc_trajectory"]) == 10 for row in rows)
    assert all(len(row["s_grid_trajectory"]) == 10 for row in rows)
    assert all(len(row["fixed_features_v1"]) == 12 for row in rows)
    assert all(row["edge_pixels"] >= 100 for row in rows)
