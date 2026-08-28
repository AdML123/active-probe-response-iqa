"""Image-level sensitivity analysis for the active trajectory detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.evaluate_iqa_ssc_trajectory_detector import auc, bootstrap_auc, fit_lda, score


FEATURE_DIM = 12
BOOTSTRAP_SEEDS = {"full": 20260824}


def aggregate_condition_vectors(rows: list[dict[str, Any]]) -> dict[tuple[str, str], np.ndarray]:
    """Average valid base-condition vectors, giving every image one weight."""

    grouped: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for row in rows:
        values = row.get("fixed_features_v1")
        if row.get("invalid_reason") is not None or not isinstance(values, list):
            continue
        vector = np.asarray(values, dtype=float)
        if vector.shape != (FEATURE_DIM,) or not np.all(np.isfinite(vector)):
            continue
        key = (str(row["image_id"]), str(row["base_family"]))
        grouped.setdefault(key, []).append((int(row["base_index"]), vector))
    result: dict[tuple[str, str], np.ndarray] = {}
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda item: item[0])
        result[key] = np.mean(np.stack([vector for _, vector in ordered], axis=0), axis=0)
    return result


def grouped_image_ids(
    grouped: dict[tuple[str, str], np.ndarray], calibration_ids: set[str], evaluation_ids: set[str]
) -> tuple[set[str], set[str]]:
    available = {image_id for image_id, _ in grouped}
    calibration = available & set(calibration_ids)
    evaluation = available & set(evaluation_ids)
    if calibration & evaluation:
        raise ValueError("calibration/evaluation image IDs overlap")
    return calibration, evaluation


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _image_matrix(grouped: dict[tuple[str, str], np.ndarray], family: str, image_ids: set[str]) -> tuple[list[str], np.ndarray]:
    selected = sorted(image_id for image_id in image_ids if (image_id, family) in grouped)
    if not selected:
        raise ValueError(f"no grouped vectors for family {family}")
    return selected, np.stack([grouped[(image_id, family)] for image_id in selected], axis=0)


def evaluate_grouped(
    calibration_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]], calibration_ids: set[str], evaluation_ids: set[str], *, n_resamples: int = 2000
) -> dict[str, Any]:
    grouped = aggregate_condition_vectors(calibration_rows + evaluation_rows)
    calibration_available, evaluation_available = grouped_image_ids(grouped, calibration_ids, evaluation_ids)
    cal_bilateral_ids, cal_bilateral = _image_matrix(grouped, "bilateral", calibration_available)
    eval_bilateral_ids, eval_bilateral = _image_matrix(grouped, "bilateral", evaluation_available)
    comparisons: dict[str, Any] = {}
    for control in ("jpeg", "gaussian_blur"):
        _, cal_control = _image_matrix(grouped, control, calibration_available)
        eval_control_ids, eval_control = _image_matrix(grouped, control, evaluation_available)
        model = fit_lda(cal_bilateral, cal_control)
        positive = score(model, eval_bilateral)
        negative = score(model, eval_control)
        interval = bootstrap_auc(positive, negative, BOOTSTRAP_SEEDS["full"], n_resamples)
        comparisons["bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"] = {
            "feature_set": "image_mean_of_five_base_conditions",
            "calibration_images_per_class": min(len(cal_bilateral_ids), len(calibration_ids)),
            "evaluation_images_per_class": min(len(eval_bilateral_ids), len(eval_control_ids)),
            "auc": auc(positive, negative),
            "auc_ci95": list(interval),
            "regularization": float(model["regularization"]),
            "mild": None,
            "severe": None,
            "stratum_note": "Severity strata are not estimable after averaging the five base-condition vectors per image.",
        }
    return {
        "schema_version": "iqa_ssc_grouped_lda_sensitivity_v1",
        "status": "computed",
        "calibration_images": len(calibration_ids),
        "evaluation_images": len(evaluation_ids),
        "grouped_vectors": len(grouped),
        "bootstrap_resamples": n_resamples,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "comparisons": comparisons,
        "row_level_reference": "trajectory_detector.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-rows", type=Path, required=True)
    parser.add_argument("--evaluation-rows", type=Path, required=True)
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    calibration_manifest = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    evaluation_manifest = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    calibration_ids = {str(record["image_id"]) for record in calibration_manifest["records"]}
    evaluation_ids = {str(record["image_id"]) for record in evaluation_manifest["records"]}
    report = evaluate_grouped(_load_rows(args.calibration_rows), _load_rows(args.evaluation_rows), calibration_ids, evaluation_ids, n_resamples=args.bootstrap)
    report["inputs"] = {
        "calibration_rows_sha256": _hash(args.calibration_rows),
        "evaluation_rows_sha256": _hash(args.evaluation_rows),
        "calibration_manifest_sha256": _hash(args.calibration_manifest),
        "evaluation_manifest_sha256": _hash(args.evaluation_manifest),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
