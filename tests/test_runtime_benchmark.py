import math
import unittest

from analysis.benchmark_detector_runtime import summarize_timings


class RuntimeBenchmarkTests(unittest.TestCase):
    def test_summary_contains_runtime_contract(self):
        summary = summarize_timings(
        [
            {"transform_ms": 10.0, "feature_ms": 5.0, "classification_ms": 1.0, "total_ms": 16.0},
            {"transform_ms": 12.0, "feature_ms": 6.0, "classification_ms": 2.0, "total_ms": 20.0},
        ],
        image_shape=(512, 512, 3),
        device="CPU",
        python_version="3.test",
        package_commit="deadbeef",
    )
        for key in ("transform_ms", "feature_ms", "classification_ms", "total_ms"):
            self.assertIn(key, summary["median_ms"])
            self.assertIn(key, summary["iqr_ms"])
            self.assertTrue(math.isfinite(summary["median_ms"][key]))
            self.assertGreaterEqual(summary["median_ms"][key], 0.0)
            self.assertGreaterEqual(summary["iqr_ms"][key], 0.0)
        self.assertEqual(summary["image_shape"], [512, 512, 3])
        self.assertEqual(summary["device"], "CPU")
        self.assertEqual(summary["python_version"], "3.test")
        self.assertEqual(summary["package_commit"], "deadbeef")


    def test_summary_rejects_negative_timings(self):
        with self.assertRaises(ValueError):
            summarize_timings(
            [{"transform_ms": -1.0, "feature_ms": 1.0, "classification_ms": 1.0, "total_ms": 1.0}],
            image_shape=(1, 1, 3),
            device="CPU",
            python_version="3.test",
            package_commit="deadbeef",
        )
if __name__ == "__main__":
    unittest.main()
