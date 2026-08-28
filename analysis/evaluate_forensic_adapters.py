"""Evaluate disclosed Ding-2019 and Shehin-2022 feature adaptations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iqa_ssc.ding2019 import texture_features
from iqa_ssc.shehin2022 import abf_evidence, feature_d, feature_dm
from iqa_ssc.transforms import apply_condition
from evaluate_passive_baselines import (
    FAMILIES,
    SEED_BY_STRATUM,
    apply_direction,
    auc,
    bootstrap_auc,
    fit_direction,
    stratum_matches,
)


METRICS = ("ding2019_texture", "shehin_d", "shehin_dm")


def load_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_vector(metric: str, image: np.ndarray) -> np.ndarray:
    if metric == "ding2019_texture":
        return texture_features(image)[0]
    if metric == "shehin_d":
        return np.asarray([abf_evidence(feature_d(image))], dtype=float)
    if metric == "shehin_dm":
        return np.asarray([abf_evidence(feature_dm(image))], dtype=float)
    raise ValueError(metric)


def make_rows(records: list[dict], image_root: Path, metrics: tuple[str, ...]) -> dict[str, list[dict]]:
    rows = {metric: [] for metric in metrics}
    for record in records:
        image_id = str(record["image_id"])
        original = load_rgb(image_root / image_id)
        for family in FAMILIES:
            for index in range(5):
                transformed = apply_condition(original, family, index)
                for metric in metrics:
                    values = feature_vector(metric, transformed)
                    rows[metric].append(
                        {
                            "image_id": image_id,
                            "base_family": family,
                            "base_index": index + 1,
                            "features": values.tolist(),
                        }
                    )
    return rows


def evaluate_metric(rows: list[dict], calibration_ids: set[str], evaluation_ids: set[str], bootstrap: int) -> dict:
    result = {}
    for control in ("jpeg", "gaussian_blur"):
        key = "bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"
        train_pos = np.asarray([row["features"] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == "bilateral"], dtype=float)
        train_neg = np.asarray([row["features"] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == control], dtype=float)
        model = fit_direction(train_pos, train_neg)
        result[key] = {}
        for stratum in ("full", "severe", "mild"):
            pos_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == "bilateral" and stratum_matches("bilateral", int(row["base_index"]), stratum)]
            neg_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == control and stratum_matches(control, int(row["base_index"]), stratum)]
            pos_scores = apply_direction(model, np.asarray([row["features"] for row in pos_rows], dtype=float))
            neg_scores = apply_direction(model, np.asarray([row["features"] for row in neg_rows], dtype=float))
            result[key][stratum] = {
                "positive_count": len(pos_rows),
                "negative_count": len(neg_rows),
                "auc": auc(pos_scores, neg_scores),
                "auc_ci95": list(bootstrap_auc(pos_scores, neg_scores, SEED_BY_STRATUM[stratum], bootstrap)),
                "model": model,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--limit", type=int, default=None, help="Limit each split for a coverage preflight.")
    args = parser.parse_args()
    calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    if args.limit is not None:
        calibration["records"] = calibration["records"][: args.limit]
        evaluation["records"] = evaluation["records"][: args.limit]
    calibration_ids = {str(record["image_id"]) for record in calibration["records"]}
    evaluation_ids = {str(record["image_id"]) for record in evaluation["records"]}
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration/evaluation image IDs overlap")
    rows = make_rows(calibration["records"] + evaluation["records"], args.image_root, tuple(args.metrics))
    report = {
        "schema_version": "iqa_ssc_forensic_adapter_audit_v1",
        "status": "computed",
        "calibration_images": len(calibration_ids),
        "evaluation_images": len(evaluation_ids),
        "bootstrap": args.bootstrap,
        "bootstrap_unit": "image_id",
        "metrics": {metric: evaluate_metric(rows[metric], calibration_ids, evaluation_ids, args.bootstrap) for metric in args.metrics},
        "protocol": {
            "calibration_manifest_sha256": sha256(args.calibration_manifest),
            "evaluation_manifest_sha256": sha256(args.evaluation_manifest),
            "metrics": list(args.metrics),
            "row_count_per_metric": {metric: len(rows[metric]) for metric in args.metrics},
            "ding_geometry": "5x5 vertical central-column edge patches, Canny 100/200, non-overlap scan",
            "shehin_filter": "fixed OpenCV bilateral sigma_space=10, sigma_color=0.10, diameter=9 in place of adaptive bilateral",
        },
        "interpretation": "Tier-B adapted forensic features; values are not claimed as exact reproductions of the published pipelines.",
    }
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_out.open("w", encoding="utf-8") as handle:
        for metric in args.metrics:
            for row in rows[metric]:
                handle.write(json.dumps({"metric": metric, **row}, separators=(",", ":")) + "\n")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({metric: report["metrics"][metric] for metric in args.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
