"""Measure the active probe-response detector on one authorized local image."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

TIMING_KEYS = ("transform_ms", "feature_ms", "classification_ms", "total_ms")


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"median": float(np.median(array)), "q1": float(np.quantile(array, 0.25)), "q3": float(np.quantile(array, 0.75))}


def summarize_timings(
    records: list[dict[str, float]],
    *,
    image_shape: tuple[int, ...],
    device: str,
    python_version: str,
    package_commit: str,
    hardware: str = "unspecified",
) -> dict[str, Any]:
    if not records:
        raise ValueError("at least one timing record is required")
    values: dict[str, list[float]] = {key: [] for key in TIMING_KEYS}
    for record in records:
        for key in TIMING_KEYS:
            value = float(record[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"invalid {key}: {value}")
            values[key].append(value)
    medians = {key: _percentiles(values[key])["median"] for key in TIMING_KEYS}
    iqrs = {key: _percentiles(values[key])["q3"] - _percentiles(values[key])["q1"] for key in TIMING_KEYS}
    return {
        "iterations": len(records),
        "median_ms": medians,
        "iqr_ms": iqrs,
        "image_shape": [int(value) for value in image_shape],
        "device": device,
        "python_version": python_version,
        "package_commit": package_commit,
        "hardware": hardware,
    }


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _calibration_model(rows_path: Path, control: str) -> dict[str, np.ndarray | float]:
    from scripts.evaluate_iqa_ssc_trajectory_detector import fit_lda

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bilateral = [row["fixed_features_v1"] for row in rows if row.get("base_family") == "bilateral" and row.get("invalid_reason") is None]
    controls = [row["fixed_features_v1"] for row in rows if row.get("base_family") == control and row.get("invalid_reason") is None]
    if not bilateral or not controls:
        raise ValueError(f"no finite calibration rows for bilateral vs {control}")
    return fit_lda(np.asarray(bilateral, dtype=float), np.asarray(controls, dtype=float))


def measure_once(image_path: Path, model: dict[str, np.ndarray | float]) -> dict[str, float]:
    from scripts.evaluate_iqa_ssc_trajectory_detector import score
    from scripts.run_iqa_ssc_trajectory_detector import (
        BASE_GRID,
        PROBE_GRID,
        apply_condition,
        fixed_features,
        fixed_masks,
        goc,
        gray_gradient,
        load_rgb,
        s_grid,
    )

    original = load_rgb(image_path)
    transform_seconds = 0.0
    feature_seconds = 0.0
    feature_rows: list[list[float]] = []
    total_start = time.perf_counter()
    for base_condition, base_index in BASE_GRID:
        start = time.perf_counter()
        base = apply_condition(original, base_condition, base_index)
        transform_seconds += time.perf_counter() - start

        start = time.perf_counter()
        base_gray, base_gx, base_gy, base_magnitude = gray_gradient(base)
        masks = fixed_masks(base_magnitude)
        feature_seconds += time.perf_counter() - start
        goc_values: list[float] = []
        grid_values: list[float] = []
        for probe_condition, probe_index in PROBE_GRID:
            start = time.perf_counter()
            probe = apply_condition(base, probe_condition, probe_index)
            transform_seconds += time.perf_counter() - start
            start = time.perf_counter()
            probe_gray, probe_gx, probe_gy, _ = gray_gradient(probe)
            goc_values.append(goc(base_gx, base_gy, probe_gx, probe_gy, masks["edge"]))
            grid_values.append(s_grid(base_gray, probe_gray))
            feature_seconds += time.perf_counter() - start
        feature_rows.append(fixed_features({"goc_trajectory": goc_values, "s_grid_trajectory": grid_values}))

    start = time.perf_counter()
    score(model, np.asarray(feature_rows, dtype=float))
    classification_seconds = time.perf_counter() - start
    total_seconds = time.perf_counter() - total_start
    return {
        "transform_ms": transform_seconds * 1000.0,
        "feature_ms": feature_seconds * 1000.0,
        "classification_ms": classification_seconds * 1000.0,
        "total_ms": total_seconds * 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True, help="authorized local input image")
    parser.add_argument("--calibration-rows", type=Path, required=True, help="frozen calibration trajectory JSONL")
    parser.add_argument("--control", choices=("jpeg", "gaussian_blur"), default="jpeg")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1:
        raise ValueError("iterations must be positive")
    from scripts.run_iqa_ssc_trajectory_detector import BASE_GRID, PROBE_GRID, load_rgb

    model = _calibration_model(args.calibration_rows, args.control)
    measure_once(args.image, model)
    records = [measure_once(args.image, model) for _ in range(args.iterations)]
    image = load_rgb(args.image)
    report = summarize_timings(
        records,
        image_shape=image.shape,
        device="CPU",
        python_version=platform.python_version(),
        package_commit=_git_commit(),
        hardware=platform.processor() or platform.platform(),
    )
    report.update({"control": args.control, "probe_count": len(PROBE_GRID), "base_condition_count": len(BASE_GRID)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
