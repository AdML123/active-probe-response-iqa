"""Evaluate frozen learned IQA scores on energy-matched JPEG pairs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iqa_ssc.learned_iqa import import_pyiqa, score_in_batches
from iqa_ssc.pair_audit import (
    bootstrap_reversal_rate,
    file_sha256,
    hash_matches,
    orient_quality_score,
    read_runtime_version,
    reversal_flag,
    validate_image_manifest,
    validate_runtime_contract,
    validate_unique_pairs,
)
from iqa_ssc.transforms import apply_condition


METRICS = ("musiq", "maniqa", "topiq_nr", "arniqa", "liqe", "clipiqa")
PAIR_HASH = "2FD382A86E9FDB675992E545FE2BDA8F79079F35A579C458FDE75616F4A706EE"
PAIR_SOURCE_ID = "v2_evaluation/bilateral_jpeg.json"
IMAGE_MANIFEST_HASH = "D27604669CF493B12CAB1A63F58723DCDE35419683856FF608DDA0E809BADE59"
IMAGE_MANIFEST_SOURCE_ID = "v2_evaluation-image-manifest.json"
BOOTSTRAP_SEED = 20260827


def load_rgb(path: Path) -> np.ndarray:
    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def score_images(metric, images: list[np.ndarray], *, device: str, batch_size: int) -> list[float]:
    import torch

    arrays = np.stack([image.astype(np.float32).transpose(2, 0, 1) / 255.0 for image in images], axis=0)
    return score_in_batches(metric, torch.from_numpy(arrays), device=device, batch_size=batch_size)


def load_pairs(path: Path, limit: int | None = None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pairs = payload.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("pair JSON must contain a pairs list")
    selected = pairs[:limit] if limit is not None else pairs
    validate_unique_pairs(selected)
    return selected


def validate_lock(
    lock_path: Path,
    artifact_lock_path: Path,
    repository_root: Path,
    pair_path: Path,
    pair_hash: str,
    image_manifest_hash: str,
    metrics: list[str],
    *,
    batch_size: int,
    bootstrap: int,
    maniqa_test_sample: int,
    device: str,
    actual_backend_version: str,
    limit: int | None,
) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "frozen":
        raise ValueError("modern pair audit lock is not frozen")
    if lock.get("pair_source_id") != PAIR_SOURCE_ID:
        raise ValueError("pair source id does not match lock")
    if not hash_matches(pair_hash, str(lock.get("pair_file_sha256", ""))):
        raise ValueError("pair file hash does not match lock")
    if not hash_matches(image_manifest_hash, str(lock.get("image_manifest_sha256", ""))):
        raise ValueError("image manifest hash does not match lock")
    locked_metrics = list(lock.get("metrics", []))
    if metrics != locked_metrics:
        raise ValueError(f"metric list does not match lock: expected {locked_metrics}")
    expected = lock.get("runtime_parameters", {})
    observed = {
        "device": device,
        "batch_size": batch_size,
        "maniqa_test_sample": maniqa_test_sample,
        "bootstrap_resamples": bootstrap,
    }
    if observed != expected:
        raise ValueError(f"runtime parameters do not match lock: {observed}")
    frozen_configuration = {
        "pair_source_id": PAIR_SOURCE_ID,
        "pair_file_sha256": pair_hash,
        "image_manifest_source_id": IMAGE_MANIFEST_SOURCE_ID,
        "image_manifest_sha256": image_manifest_hash,
        "control_family": "jpeg",
        "global_loss_tolerance": 0.05,
        "metrics": metrics,
        "backend_package": "pyiqa",
        "backend_version": actual_backend_version,
        "score_orientation": "higher quality after sign correction",
        "reversal_rule": "oriented bilateral score > oriented JPEG score",
        "tie_rule": "ties are not reversals",
        "invalid_row_policy": "fail closed for malformed pairs, missing images, nonfinite scores, or unsupported condition indices; no imputation",
        "runtime_parameters": observed,
        "bootstrap": {
            "unit": "image_id",
            "resamples": bootstrap,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile 95%",
        },
        "blur_modern_audit": "deferred",
    }
    validate_runtime_contract(
        lock,
        actual_backend_version=actual_backend_version,
        artifact_lock_path=artifact_lock_path,
        repository_root=repository_root,
        observed_configuration=frozen_configuration,
    )
    if limit is None and int(lock.get("pair_count", -1)) < 1:
        raise ValueError("lock has invalid pair count")
    if not pair_path.is_file():
        raise FileNotFoundError(pair_path)
    return lock


def validate_image_files(image_root: Path, image_records: dict[str, dict], required_ids: set[str]) -> None:
    for image_id in sorted(required_ids):
        record = image_records[image_id]
        image_path = image_root / str(record["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"missing restricted input for {image_id}")
        if not hash_matches(file_sha256(image_path), str(record["sha256"])):
            raise ValueError(f"image hash mismatch for {image_id}")


def score_metric_pairs(
    pyiqa,
    backend_label: str,
    metric_name: str,
    pairs: list[dict],
    image_root: Path,
    image_records: dict[str, dict],
    *,
    device: str,
    batch_size: int,
    maniqa_test_sample: int | None,
    frozen_configuration_sha256: str,
) -> tuple[list[dict], dict]:
    metric_kwargs = {}
    if metric_name == "maniqa" and maniqa_test_sample is not None:
        metric_kwargs["test_sample"] = maniqa_test_sample
    model = pyiqa.create_metric(metric_name, device=device, **metric_kwargs)
    lower_better = bool(getattr(model, "lower_better"))
    rows: list[dict] = []
    try:
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            originals = [load_rgb(image_root / str(image_records[str(pair["image_id"])]["relative_path"])) for pair in chunk]
            bilateral_images = [
                apply_condition(original, "bilateral", int(pair["selective_index"]) - 1)
                for original, pair in zip(originals, chunk)
            ]
            control_images = [
                apply_condition(original, "jpeg", int(pair["control_index"]) - 1)
                for original, pair in zip(originals, chunk)
            ]
            raw_bilateral = score_images(model, bilateral_images, device=device, batch_size=batch_size)
            raw_control = score_images(model, control_images, device=device, batch_size=batch_size)
            if len(raw_bilateral) != len(chunk) or len(raw_control) != len(chunk):
                raise ValueError("metric output count does not match pair batch")
            for pair, bilateral_score, control_score in zip(chunk, raw_bilateral, raw_control):
                oriented_bilateral = orient_quality_score(bilateral_score, lower_better)
                oriented_control = orient_quality_score(control_score, lower_better)
                rows.append({
                    "image_id": str(pair["image_id"]),
                    "split": "v2_evaluation",
                    "control_family": "jpeg",
                    "bilateral_index": int(pair["selective_index"]),
                    "control_index": int(pair["control_index"]),
                    "metric": metric_name,
                    "backend": backend_label,
                    "configuration": metric_kwargs,
                    "frozen_configuration_sha256": frozen_configuration_sha256,
                    "pair_file_sha256": PAIR_HASH,
                    "image_manifest_sha256": IMAGE_MANIFEST_HASH,
                    "bilateral_raw": float(bilateral_score),
                    "control_raw": float(control_score),
                    "bilateral_oriented": oriented_bilateral,
                    "control_oriented": oriented_control,
                    "reversal": reversal_flag(oriented_bilateral, oriented_control),
                    "valid": True,
                    "invalid_reason": None,
                    "lower_better": lower_better,
                })
    finally:
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    return rows, {
        "metric": metric_name,
        "lower_better": lower_better,
        "configuration": metric_kwargs,
        "frozen_configuration_sha256": frozen_configuration_sha256,
        "backend": backend_label,
    }


def summarize(rows: list[dict], *, seed: int, bootstrap: int) -> dict:
    valid = [row for row in rows if row.get("valid")]
    if not valid:
        raise ValueError("no valid pair rows")
    rate = float(np.mean([row["reversal"] for row in valid]))
    ci = bootstrap_reversal_rate(valid, seed=seed, n_resamples=bootstrap)
    differences = np.asarray(
        [row["bilateral_oriented"] - row["control_oriented"] for row in valid],
        dtype=float,
    )
    return {
        "pair_count": len(rows),
        "valid_count": len(valid),
        "reversal_count": int(sum(row["reversal"] for row in valid)),
        "reversal_rate": rate,
        "ci95": list(ci),
        "median_oriented_difference": float(np.median(differences)),
        "bootstrap_seed": seed,
        "bootstrap_resamples": bootstrap,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--lock-json", type=Path, required=True)
    parser.add_argument("--artifact-lock-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--maniqa-test-sample", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    args = parser.parse_args()
    if args.batch_size < 1 or args.bootstrap < 1:
        raise ValueError("batch-size and bootstrap must be positive")
    pair_hash = file_sha256(args.pair_json)
    image_manifest_hash = file_sha256(args.image_manifest)
    if args.limit is None and not hash_matches(pair_hash, PAIR_HASH):
        raise ValueError(f"unexpected frozen pair hash: {pair_hash}")
    pyiqa = import_pyiqa()
    backend_version = read_runtime_version(pyiqa, "pyiqa")
    backend_label = f"pyiqa {backend_version}"
    artifact_lock_path = args.artifact_lock_json or args.lock_json.with_name("artifact-lock.json")
    repository_root = Path(__file__).resolve().parents[2]
    lock = validate_lock(
        args.lock_json,
        artifact_lock_path,
        repository_root,
        args.pair_json,
        pair_hash,
        image_manifest_hash,
        args.metrics,
        batch_size=args.batch_size,
        bootstrap=args.bootstrap,
        maniqa_test_sample=args.maniqa_test_sample,
        device=args.device,
        actual_backend_version=backend_version,
        limit=args.limit,
    )
    pairs = load_pairs(args.pair_json, args.limit)
    if args.limit is None and len(pairs) != int(lock["pair_count"]):
        raise ValueError("pair count does not match lock")
    image_manifest = json.loads(args.image_manifest.read_text(encoding="utf-8"))
    required_ids = {str(pair["image_id"]) for pair in pairs}
    image_records = validate_image_manifest(image_manifest, required_ids)
    validate_image_files(args.image_root, image_records, required_ids)
    all_rows: list[dict] = []
    metric_reports: dict[str, dict] = {}
    for metric_name in args.metrics:
        try:
            rows, config = score_metric_pairs(
                pyiqa,
                backend_label,
                metric_name,
                pairs,
                args.image_root,
                image_records,
                device=args.device,
                batch_size=args.batch_size,
                maniqa_test_sample=args.maniqa_test_sample,
                frozen_configuration_sha256=str(lock["frozen_configuration_sha256"]),
            )
            all_rows.extend(rows)
            metric_reports[metric_name] = {
                **config,
                "status": "computed",
                **summarize(rows, seed=BOOTSTRAP_SEED, bootstrap=args.bootstrap),
            }
        except Exception as error:
            metric_reports[metric_name] = {
                "status": "modern_audit_unavailable",
                "reason": type(error).__name__,
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "modern-pair-audit.jsonl"
    with rows_path.open("w", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "iqa_ssc_modern_pair_audit_v1",
        "status": "computed" if all(item.get("status") == "computed" for item in metric_reports.values()) else "modern_audit_unavailable",
        "pair_source_id": PAIR_SOURCE_ID,
        "pair_file_sha256": pair_hash,
        "pair_count": len(pairs),
        "source_manifest_id": IMAGE_MANIFEST_SOURCE_ID,
        "source_manifest_sha256": image_manifest_hash,
        "source_manifest_images": int(image_manifest["count"]),
        "contributing_images": len(required_ids),
        "frozen_configuration_sha256": str(lock["frozen_configuration_sha256"]),
        "metrics": metric_reports,
        "blur_modern_audit": "deferred",
        "protocol": {
            "backend": backend_label,
            "device": args.device,
            "input_range": "float32 RGB [0,1]",
            "batch_size": args.batch_size,
            "maniqa_test_sample": args.maniqa_test_sample,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": args.bootstrap,
            "global_loss_tolerance": 0.05,
            "independence_unit": "image_id",
        },
    }
    (args.output_dir / "modern-pair-audit-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
