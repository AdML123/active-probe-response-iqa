"""Evaluate pyiqa scalar baselines on the frozen detector split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iqa_ssc.learned_iqa import import_pyiqa, score_in_batches
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


METRICS = ("brisque", "niqe", "piqe", "musiq", "maniqa", "topiq_nr", "liqe", "arniqa", "clipiqa")


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


def _score_images(metric, images: list[np.ndarray], *, device: str, batch_size: int) -> list[float]:
    import torch

    arrays = np.stack([image.astype(np.float32).transpose(2, 0, 1) / 255.0 for image in images], axis=0)
    tensor = torch.from_numpy(arrays)
    return score_in_batches(metric, tensor, device=device, batch_size=batch_size)


def make_rows(
    records: list[dict],
    image_root: Path,
    metric_names: tuple[str, ...],
    *,
    device: str,
    batch_size: int,
    maniqa_test_sample: int | None = None,
) -> dict[str, list[dict]]:
    rows = {name: [] for name in metric_names}
    pyiqa = import_pyiqa()
    for name in metric_names:
        metric_kwargs = {}
        if name == "maniqa" and maniqa_test_sample is not None:
            metric_kwargs["test_sample"] = maniqa_test_sample
        model = pyiqa.create_metric(name, device=device, **metric_kwargs)
        try:
            for start in range(0, len(records), batch_size):
                chunk = records[start : start + batch_size]
                originals = [load_rgb(image_root / str(record["image_id"])) for record in chunk]
                for family in FAMILIES:
                    for index in range(5):
                        transformed = [apply_condition(image, family, index) for image in originals]
                        scores = _score_images(model, transformed, device=device, batch_size=batch_size)
                        for position, record in enumerate(chunk):
                            rows[name].append(
                                {
                                    "image_id": str(record["image_id"]),
                                    "base_family": family,
                                    "base_index": index + 1,
                                    "score": float(scores[position]),
                                }
                            )
        finally:
            import gc
            import torch

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def evaluate_metric(rows: list[dict], calibration_ids: set[str], evaluation_ids: set[str], bootstrap: int) -> dict:
    result = {}
    for control in ("jpeg", "gaussian_blur"):
        key = "bilateral_vs_jpeg" if control == "jpeg" else "bilateral_vs_blur"
        result[key] = {}
        train_pos = np.asarray([row["score"] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == "bilateral"], dtype=float)[:, None]
        train_neg = np.asarray([row["score"] for row in rows if row["image_id"] in calibration_ids and row["base_family"] == control], dtype=float)[:, None]
        model = fit_direction(train_pos, train_neg)
        for stratum in ("full", "severe", "mild"):
            pos_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == "bilateral" and stratum_matches("bilateral", int(row["base_index"]), stratum)]
            neg_rows = [row for row in rows if row["image_id"] in evaluation_ids and row["base_family"] == control and stratum_matches(control, int(row["base_index"]), stratum)]
            pos_scores = apply_direction(model, np.asarray([[row["score"]] for row in pos_rows], dtype=float))
            neg_scores = apply_direction(model, np.asarray([[row["score"]] for row in neg_rows], dtype=float))
            ci = bootstrap_auc(pos_scores, neg_scores, SEED_BY_STRATUM[stratum], bootstrap)
            result[key][stratum] = {
                "positive_count": len(pos_rows),
                "negative_count": len(neg_rows),
                "auc": auc(pos_scores, neg_scores),
                "auc_ci95": list(ci),
                "model": model,
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--maniqa-test-sample",
        type=int,
        default=None,
        help="Override MANIQA internal crop count; record this as an adapted configuration.",
    )
    args = parser.parse_args()
    calibration = json.loads(args.calibration_manifest.read_text(encoding="utf-8"))
    evaluation = json.loads(args.evaluation_manifest.read_text(encoding="utf-8"))
    calibration_ids = {str(record["image_id"]) for record in calibration["records"]}
    evaluation_ids = {str(record["image_id"]) for record in evaluation["records"]}
    if calibration_ids & evaluation_ids:
        raise ValueError("calibration/evaluation image IDs overlap")
    records = calibration["records"] + evaluation["records"]
    rows = make_rows(
        records,
        args.image_root,
        tuple(args.metrics),
        device=args.device,
        batch_size=args.batch_size,
        maniqa_test_sample=args.maniqa_test_sample,
    )
    report = {
        "status": "computed",
        "calibration_images": len(calibration_ids),
        "evaluation_images": len(evaluation_ids),
        "bootstrap": args.bootstrap,
        "bootstrap_seeds": SEED_BY_STRATUM,
        "metrics": {name: evaluate_metric(rows[name], calibration_ids, evaluation_ids, args.bootstrap) for name in args.metrics},
        "protocol": {
            "calibration_manifest_sha256": sha256(args.calibration_manifest),
            "evaluation_manifest_sha256": sha256(args.evaluation_manifest),
            "metrics": list(args.metrics),
            "batch_size": args.batch_size,
            "maniqa_test_sample": args.maniqa_test_sample,
            "row_count_per_metric": {name: len(rows[name]) for name in args.metrics},
        },
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: report["metrics"][name] for name in args.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
