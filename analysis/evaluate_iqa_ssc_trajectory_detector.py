"""Evaluate the active trajectory detector with calibration-only LDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_DIM = 12
BOOTSTRAP_SEEDS = {"full": 20260824, "severe": 20260825, "mild": 20260826}
FEATURE_SUBSETS = {
    "jpeg_probe_6": (0, 1, 2, 6, 7, 8),
    "blur_probe_6": (3, 4, 5, 9, 10, 11),
    "goc_only_6": (0, 1, 2, 3, 4, 5),
    "s_grid_only_6": (6, 7, 8, 9, 10, 11),
}


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    if scores_pos.size == 0 or scores_neg.size == 0:
        raise ValueError("AUC requires positive and negative scores")
    values = np.concatenate([scores_pos, scores_neg])
    labels = np.concatenate([np.ones(scores_pos.size, dtype=int), np.zeros(scores_neg.size, dtype=int)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - scores_pos.size * (scores_pos.size + 1) / 2.0) / (scores_pos.size * scores_neg.size))


def bootstrap_auc(scores_pos: np.ndarray, scores_neg: np.ndarray, seed: int, n_resamples: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        pos = scores_pos[rng.integers(0, scores_pos.size, scores_pos.size)]
        neg = scores_neg[rng.integers(0, scores_neg.size, scores_neg.size)]
        values[index] = auc(pos, neg)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratum_matches(family: str, index: int, name: str) -> bool:
    if name == "full":
        return True
    if name == "severe":
        return (family == "bilateral" and index >= 4) or (family == "jpeg" and index <= 2) or (family == "gaussian_blur" and index >= 4)
    if name == "mild":
        return (family == "bilateral" and index <= 2) or (family == "jpeg" and index >= 4) or (family == "gaussian_blur" and index <= 2)
    raise ValueError(f"unknown stratum: {name}")


def validate_split(calibration_ids: set[str], evaluation_ids: set[str]) -> None:
    overlap = sorted(calibration_ids & evaluation_ids)
    if overlap:
        raise ValueError(f"calibration/evaluation overlap: {overlap[:5]}")


def invert_matrix(matrix: np.ndarray) -> np.ndarray:
    values = [[float(value) for value in row] for row in np.asarray(matrix, dtype=float)]
    size = len(values)
    augmented = [row + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(values)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        if abs(augmented[pivot][col]) < 1e-12:
            raise ValueError("singular LDA covariance")
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(size):
            if row == col:
                continue
            factor = augmented[row][col]
            if factor:
                augmented[row] = [left - factor * right for left, right in zip(augmented[row], augmented[col])]
    return np.asarray([row[size:] for row in augmented], dtype=float)


def fit_lda(train_pos: np.ndarray, train_neg: np.ndarray) -> dict[str, np.ndarray | float]:
    train_pos = np.asarray(train_pos, dtype=float)
    train_neg = np.asarray(train_neg, dtype=float)
    if train_pos.ndim != 2 or train_neg.ndim != 2 or train_pos.shape[1] != train_neg.shape[1]:
        raise ValueError("LDA inputs must be two matrices with equal feature dimensions")
    if train_pos.shape[1] != FEATURE_DIM and train_pos.shape[1] == 0:
        raise ValueError("LDA requires at least one feature")
    train = np.vstack([train_pos, train_neg])
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    pos = (train_pos - mean) / std
    neg = (train_neg - mean) / std
    combined = np.vstack([pos, neg])
    centered = combined - combined.mean(axis=0)
    covariance = np.asarray([[float(np.sum(centered[:, i] * centered[:, j])) / max(1, combined.shape[0] - 1) for j in range(combined.shape[1])] for i in range(combined.shape[1])], dtype=float)
    regularization = 1e-3 * float(np.trace(covariance)) / covariance.shape[0]
    inverse = invert_matrix(covariance + regularization * np.eye(covariance.shape[0]))
    difference = pos.mean(axis=0) - neg.mean(axis=0)
    direction = np.asarray([sum(row[j] * float(difference[j]) for j in range(len(difference))) for row in inverse], dtype=float)
    return {"mean": mean, "std": std, "direction": direction, "regularization": regularization}


def score(model: dict[str, np.ndarray | float], values: np.ndarray) -> np.ndarray:
    normalized = (np.asarray(values, dtype=float) - model["mean"]) / model["std"]
    return np.asarray([sum(float(value) * float(weight) for value, weight in zip(row, model["direction"])) for row in normalized], dtype=float)


def single_point_features(row: dict[str, Any]) -> list[float]:
    goc_values = list(row["goc_trajectory"])
    grid_values = list(row["s_grid_trajectory"])
    if len(goc_values) != 10 or len(grid_values) != 10:
        raise ValueError("trajectory must contain ten GOC and ten S_grid values")
    return [float(goc_values[2]), float(goc_values[7]), float(grid_values[2]), float(grid_values[7])]


def feature_values(row: dict[str, Any], feature_set: str) -> np.ndarray:
    if feature_set == "full_12":
        values = row.get("fixed_features_v1")
    elif feature_set == "single_point_q30_sigma3":
        values = single_point_features(row)
    else:
        full = row.get("fixed_features_v1")
        if not isinstance(full, list):
            return np.asarray([], dtype=float)
        values = [full[index] for index in FEATURE_SUBSETS[feature_set]]
    return np.asarray(values, dtype=float)


def valid_for_feature_set(row: dict[str, Any], feature_set: str) -> bool:
    if row.get("invalid_reason") is not None:
        return False
    try:
        values = feature_values(row, feature_set)
    except (KeyError, TypeError, ValueError):
        return False
    return values.size > 0 and bool(np.all(np.isfinite(values)))


def _valid(row: dict[str, Any], feature_dim: int = FEATURE_DIM) -> bool:
    values = row.get("fixed_features_v1")
    return row.get("invalid_reason") is None and isinstance(values, list) and len(values) == feature_dim and bool(np.all(np.isfinite(np.asarray(values, dtype=float))))


def _row_ids(manifest: dict[str, Any]) -> tuple[set[str], set[str]]:
    calibration_ids = set(manifest["calibration_ids"])
    evaluation_ids = set(manifest["evaluation_ids"])
    validate_split(calibration_ids, evaluation_ids)
    return calibration_ids, evaluation_ids


def _coverage(rows: list[dict[str, Any]]) -> dict[str, int | float]:
    valid = sum(_valid(row) for row in rows)
    total = len(rows)
    return {"valid": valid, "total": total, "coverage": valid / total if total else 0.0}


def _evaluate_comparison(rows: list[dict[str, Any]], calibration_ids: set[str], evaluation_ids: set[str], control: str, feature_set: str, *, n_resamples: int) -> dict[str, Any]:
    usable = [row for row in rows if valid_for_feature_set(row, feature_set)]
    train_pos = np.asarray([feature_values(row, feature_set) for row in usable if row["image_id"] in calibration_ids and row["base_family"] == "bilateral"], dtype=float)
    train_neg = np.asarray([feature_values(row, feature_set) for row in usable if row["image_id"] in calibration_ids and row["base_family"] == control], dtype=float)
    model = fit_lda(train_pos, train_neg)
    strata: dict[str, Any] = {}
    for stratum in ("full", "severe", "mild"):
        pos_rows = [row for row in usable if row["image_id"] in evaluation_ids and row["base_family"] == "bilateral" and stratum_matches("bilateral", int(row["base_index"]), stratum)]
        neg_rows = [row for row in usable if row["image_id"] in evaluation_ids and row["base_family"] == control and stratum_matches(control, int(row["base_index"]), stratum)]
        pos_scores = score(model, np.asarray([feature_values(row, feature_set) for row in pos_rows], dtype=float))
        neg_scores = score(model, np.asarray([feature_values(row, feature_set) for row in neg_rows], dtype=float))
        current_auc = auc(pos_scores, neg_scores)
        ci = bootstrap_auc(pos_scores, neg_scores, BOOTSTRAP_SEEDS[stratum], n_resamples=n_resamples)
        strata[stratum] = {
            "positive_count": len(pos_rows),
            "negative_count": len(neg_rows),
            "auc": current_auc,
            "auc_ci95": list(ci),
            "full_gate": current_auc >= 0.80 and ci[0] >= 0.75,
        }
    return {"feature_set": feature_set, "regularization": float(model["regularization"]), "strata": strata}


def evaluate_rows(rows: list[dict[str, Any]], manifest: dict[str, Any], *, n_resamples: int = 2000, include_ablations: bool = False) -> dict[str, Any]:
    calibration_ids, evaluation_ids = _row_ids(manifest)
    usable = [row for row in rows if _valid(row)]
    evaluation_rows = [row for row in rows if row["image_id"] in evaluation_ids]
    report: dict[str, Any] = {
        "schema_version": "iqa_ssc_trajectory_detector_report_v1",
        "status": "computed",
        "row_count": len(rows),
        "usable_rows": len(usable),
        "feature_dim": FEATURE_DIM,
        "calibration_images": len(calibration_ids),
        "evaluation_images": len(evaluation_ids),
        "bootstrap_resamples": n_resamples,
        "bootstrap_seeds": BOOTSTRAP_SEEDS,
        "feature_order": [
            "GOC_JPEG_first", "GOC_JPEG_last", "GOC_JPEG_area",
            "GOC_blur_first", "GOC_blur_last", "GOC_blur_area",
            "S_grid_JPEG_first", "S_grid_JPEG_last", "S_grid_JPEG_area",
            "S_grid_blur_first", "S_grid_blur_last", "S_grid_blur_area",
        ],
        "validity": {family: _coverage([row for row in rows if row["base_family"] == family]) for family in ("bilateral", "jpeg", "gaussian_blur")},
        "evaluation_validity": {family: _coverage([row for row in evaluation_rows if row["base_family"] == family]) for family in ("bilateral", "jpeg", "gaussian_blur")},
        "comparisons": {},
    }
    for control in ("jpeg", "gaussian_blur"):
        key = "bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"
        comparison = _evaluate_comparison(rows, calibration_ids, evaluation_ids, control, "full_12", n_resamples=n_resamples)
        comparison["primary_gate"] = bool(comparison["strata"]["full"]["full_gate"] and report["evaluation_validity"]["bilateral"]["coverage"] >= 0.95 and report["evaluation_validity"][control]["coverage"] >= 0.95)
        report["comparisons"][key] = comparison
    if include_ablations:
        report["ablations"] = {}
        for feature_set in (*FEATURE_SUBSETS.keys(), "single_point_q30_sigma3"):
            report["ablations"][feature_set] = {}
            for control in ("jpeg", "gaussian_blur"):
                key = "bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"
                report["ablations"][feature_set][key] = _evaluate_comparison(rows, calibration_ids, evaluation_ids, control, feature_set, n_resamples=n_resamples)
    report["split_disjoint"] = True
    report["primary_gate_all"] = bool(all(value["primary_gate"] for value in report["comparisons"].values()))
    return report


def markdown_report(report: dict[str, Any]) -> str:
    lines = ["# IQA-SSC Active Trajectory Detector", "", f"Status: `{report['status']}`", "", "## Validity", "", "| Family | Valid | Total | Coverage |", "|---|---:|---:|---:|"]
    for family, values in report["validity"].items():
        lines.append(f"| {family} | {values['valid']} | {values['total']} | {values['coverage']:.4f} |")
    lines.extend(["", "## Comparisons", "", "| Comparison | Stratum | N positive | N negative | AUC | CI lower | CI upper | Gate |", "|---|---|---:|---:|---:|---:|---:|:---:|"])
    for comparison, values in report["comparisons"].items():
        for stratum, result in values["strata"].items():
            lines.append(f"| {comparison} | {stratum} | {result['positive_count']} | {result['negative_count']} | {result['auc']:.6f} | {result['auc_ci95'][0]:.6f} | {result['auc_ci95'][1]:.6f} | {'PASS' if result['full_gate'] else 'FAIL'} |")
    if report.get("ablations"):
        lines.extend(["", "## Ablations (full stratum)", "", "| Feature set | Comparison | AUC | CI lower | CI upper |", "|---|---|---:|---:|---:|"])
        for feature_set, comparisons in report["ablations"].items():
            for comparison, values in comparisons.items():
                result = values["strata"]["full"]
                lines.append(f"| {feature_set} | {comparison} | {result['auc']:.6f} | {result['auc_ci95'][0]:.6f} | {result['auc_ci95'][1]:.6f} |")
    lines.extend(["", f"Primary detector gate: **{'PASS' if report['primary_gate_all'] else 'FAIL'}**", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.rows.read_text(encoding="utf-8").splitlines() if line.strip()]
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = evaluate_rows(rows, manifest, n_resamples=args.bootstrap, include_ablations=args.ablations)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_out.write_text(markdown_report(report) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
