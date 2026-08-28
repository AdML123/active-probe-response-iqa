import json
from pathlib import Path


def _lock():
    return json.loads(
        Path("results/strong-baselines/protocol-lock.json").read_text(
            encoding="utf-8"
        )
    )


def test_spatial_selectivity_uses_fixed_family_labels():
    lock = _lock()
    assert lock["selective_label_source"] == "bilateral_transform_family"
    assert lock["relabel_by_kappa"] is False
    assert lock["selectivity"]["skin_label"] == 1


def test_group_unit_and_matching_tolerance_are_locked():
    lock = _lock()
    assert lock["independence_unit"] == "image_id"
    assert lock["global_loss_tolerance"] == 0.05
    assert lock["statistics"]["bootstrap_unit"] == "image_id"
    assert lock["statistics"]["bootstrap_resamples"] == 2000


def test_transform_grid_has_five_base_levels_and_ten_probes():
    lock = _lock()
    grid = lock["transform_grid"]
    assert len(grid["bilateral"]["sigma_space"]) == 5
    assert len(grid["jpeg_quality"]) == 5
    assert len(grid["gaussian_blur_sigma"]) == 5
    assert grid["probe_count_per_image"] == 10
