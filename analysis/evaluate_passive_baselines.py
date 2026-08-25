"""Evaluate passive single-image baselines on the frozen detector split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from iqa_ssc.baselines import NO_REFERENCE_METRICS, dct_high_frequency_ratio, local_texture_statistics, score_metric_batch
from iqa_ssc.transforms import apply_condition


METRICS = ("brisque", "niqe", "piqe", "dct_ratio", "texture_stats")
FAMILIES = ("bilateral", "jpeg", "gaussian_blur")
SEED_BY_STRATUM = {"full": 20260824, "severe": 20260825, "mild": 20260826}


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


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.size == 0 or neg.size == 0 or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(neg)):
        raise ValueError("AUC requires finite non-empty scores")
    values = np.concatenate((pos, neg))
    labels = np.concatenate((np.ones(pos.size, dtype=int), np.zeros(neg.size, dtype=int)))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_ranks = ranks[labels == 1]
    return float((pos_ranks.sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def bootstrap_auc(pos: np.ndarray, neg: np.ndarray, seed: int, n_resamples: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        values[index] = auc(pos[rng.integers(0, pos.size, pos.size)], neg[rng.integers(0, neg.size, neg.size)])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def stratum_matches(family: str, index: int, name: str) -> bool:
    if name == "full":
        return True
    if name == "severe":
        return (family == "bilateral" and index >= 4) or (family == "jpeg" and index <= 2) or (family == "gaussian_blur" and index >= 4)
    if name == "mild":
        return (family == "bilateral" and index <= 2) or (family == "jpeg" and index >= 4) or (family == "gaussian_blur" and index <= 2)
    raise ValueError(name)


def fit_direction(pos: np.ndarray, neg: np.ndarray) -> dict[str, Any]:
    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    if pos.ndim != 2 or neg.ndim != 2 or pos.shape[1] != neg.shape[1] or pos.shape[0] < 2 or neg.shape[0] < 2:
        raise ValueError("training arrays must be non-empty matrices with equal dimensions")
    train = np.vstack((pos, neg))
    mean = train.mean(axis=0)
    std = train.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    normalized = (train - mean) / std
    centered = normalized - normalized.mean(axis=0)
    covariance = centered.T @ centered / max(1, normalized.shape[0] - 1)
    ridge = 1e-3 * float(np.trace(covariance)) / covariance.shape[0]
    inverse = np.linalg.pinv(covariance + ridge * np.eye(covariance.shape[0]))
    difference = ((pos - mean) / std).mean(axis=0) - ((neg - mean) / std).mean(axis=0)
    direction = inverse @ difference
    return {"mean": mean.tolist(), "std": std.tolist(), "direction": direction.tolist(), "ridge": ridge}


def apply_direction(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return ((values - np.asarray(model["mean"])) / np.asarray(model["std"])) @ np.asarray(model["direction"])


def feature_vector(metric: str, image: np.ndarray, device: str) -> np.ndarray:
    if metric in ("brisque", "niqe", "piqe"):
        return np.asarray([score_metric(metric, image, device=device)], dtype=float)
    if metric == "dct_ratio":
        return np.asarray([dct_high_frequency_ratio(image)], dtype=float)
    if metric == "texture_stats":
        return local_texture_statistics(image)
    raise ValueError(metric)


def make_rows(manifest: dict[str, Any], image_root: Path, metrics: tuple[str, ...], device: str, batch_size: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    records = manifest["records"]
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        originals = [(str(record["image_id"]), load_rgb(image_root / str(record["image_id"]))) for record in batch_records]
        for family in FAMILIES:
            for index in range(5):
                transformed = [apply_condition(original, family, index) for _, original in originals]
                batch_scores: dict[str, list[float]] = {}
                for metric in metrics:
                    if metric in NO_REFERENCE_METRICS:
                        batch_scores[metric] = score_metric_batch(metric, transformed, device=device)
                    else:
                        batch_scores[metric] = [feature_vector(metric, image, device)[0] if metric == "dct_ratio" else feature_vector(metric, image, device).tolist() for image in transformed]
                for position, (image_id, _) in enumerate(originals):
                    scores = {metric: ([batch_scores[metric][position]] if metric == "dct_ratio" else batch_scores[metric][position]) for metric in metrics}
                    rows.append({"image_id": image_id, "base_family": family, "base_index": index + 1, "scores": scores})
    return rows


def evaluate(rows: list[dict[str, Any]], calibration_ids: set[str], evaluation_ids: set[str], metrics: tuple[str, ...], bootstrap: int) -> dict[str, Any]:
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration/evaluation image IDs overlap")
    report: dict[str, Any] = {"metrics": {}, "calibration_images": len(calibration_ids), "evaluation_images": len(evaluation_ids), "bootstrap": bootstrap, "bootstrap_seeds": SEED_BY_STRATUM}
    for metric in metrics:
        report["metrics"][metric] = {}
        for control in ("jpeg", "gaussian_blur"):
            key = "bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"
            report["metrics"][metric][key] = {}
            for stratum in ("full", "severe", "mild"):
                train_pos = np.asarray([row["scores"][metric] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == "bilateral"], dtype=float)
                train_neg = np.asarray([row["scores"][metric] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == control], dtype=float)
                model = fit_direction(train_pos, train_neg)
                pos_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == "bilateral" and stratum_matches("bilateral", int(row["base_index"]), stratum)]
                neg_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == control and stratum_matches(control, int(row["base_index"]), stratum)]
                pos_scores = apply_direction(model, np.asarray([row["scores"][metric] for row in pos_rows], dtype=float))
                neg_scores = apply_direction(model, np.asarray([row["scores"][metric] for row in neg_rows], dtype=float))
                interval = bootstrap_auc(pos_scores, neg_scores, SEED_BY_STRATUM[stratum], bootstrap)
                report["metrics"][metric][key][stratum] = {"positive_count": len(pos_rows), "negative_count": len(neg_rows), "auc": auc(pos_scores, neg_scores), "auc_ci95": list(interval), "model": model}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    calibration_ids = {str(record["image_id"]) for record in calibration["records"]}
    evaluation_ids = {str(record["image_id"]) for record in evaluation["records"]}
    rows = make_rows({"records": calibration["records"] + evaluation["records"]}, args.image_root, tuple(args.metrics), args.device, args.batch_size)
    report = evaluate(rows, calibration_ids, evaluation_ids, tuple(args.metrics), args.bootstrap)
    report["status"] = "computed"
    report["protocol"] = {"calibration_manifest_sha256": sha256(args.calibration_manifest), "evaluation_manifest_sha256": sha256(args.evaluation_manifest), "image_root": str(args.image_root), "metrics": list(args.metrics), "row_count": len(rows), "coverage": {metric: 1.0 for metric in args.metrics}}
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    with args.rows_out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "rows": len(rows), "metrics": list(args.metrics)}, indent=2))


if __name__ == "__main__":
    main()
