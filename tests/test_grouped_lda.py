from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_grouped_lda import aggregate_condition_vectors, grouped_image_ids


class GroupedLdaTests(unittest.TestCase):
    def test_aggregate_conditions_returns_one_vector_per_image(self) -> None:
        rows = [
            {
                "image_id": "1.jpg",
                "base_family": "bilateral",
                "base_index": index,
                "fixed_features_v1": [float(index)] * 12,
            }
            for index in range(5)
        ]
        result = aggregate_condition_vectors(rows)
        self.assertEqual(list(result), [("1.jpg", "bilateral")])
        self.assertEqual(result[("1.jpg", "bilateral")].shape, (12,))
        self.assertEqual(result[("1.jpg", "bilateral")][0], 2.0)

    def test_grouped_image_ids_are_disjoint(self) -> None:
        rows = [
            {"image_id": "1.jpg", "base_family": "bilateral", "base_index": 1, "fixed_features_v1": [1.0] * 12},
            {"image_id": "2.jpg", "base_family": "bilateral", "base_index": 1, "fixed_features_v1": [2.0] * 12},
        ]
        grouped = aggregate_condition_vectors(rows)
        calibration, evaluation = grouped_image_ids(grouped, {"1.jpg"}, {"2.jpg"})
        self.assertEqual(calibration, {"1.jpg"})
        self.assertEqual(evaluation, {"2.jpg"})

    def test_aggregate_rejects_nonfinite_condition_vectors(self) -> None:
        rows = [
            {"image_id": "1.jpg", "base_family": "bilateral", "base_index": 1, "fixed_features_v1": [np.nan] * 12},
        ]
        self.assertEqual(aggregate_condition_vectors(rows), {})
