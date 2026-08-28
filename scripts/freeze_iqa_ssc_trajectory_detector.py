"""Freeze the independent calibration/evaluation boundary for the trajectory detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_image_key(image_id: str) -> tuple[int, str]:
    name = Path(image_id).name
    stem = Path(name).stem
    try:
        number = int(stem)
    except ValueError as exc:
        raise ValueError(f"image id has no integer stem: {image_id}") from exc
    return number, name


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("records"), list):
        raise ValueError(f"invalid manifest: {path}")
    return value


def _record_path(manifest: dict[str, Any], record: dict[str, Any]) -> Path:
    root = manifest.get("asset_root")
    relative = record.get("relative_path") or record.get("image_id")
    if not root or not relative:
        raise ValueError("manifest record lacks asset_root or relative_path")
    return Path(root) / relative


def _verify_records(manifest: dict[str, Any]) -> None:
    for record in manifest["records"]:
        source = _record_path(manifest, record)
        if not source.is_file():
            raise FileNotFoundError(source)
        expected = record.get("sha256")
        if expected and sha256_file(source) != expected:
            raise ValueError(f"source hash mismatch: {source}")


def _child_manifest(parent: dict[str, Any], records: list[dict[str, Any]], role: str, parent_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "iqa_ssc_trajectory_detector_child_v1",
        "status": "frozen",
        "role": role,
        "parent_manifest": "results/provenance/v3_fresh/evaluation.json",
        "parent_sha256": parent_hash,
        "sort": "numeric_stem_then_filename",
        "asset_root": parent.get("asset_root"),
        "count": len(records),
        "records": records,
    }


def freeze_manifests(parent_path: Path, pilot_path: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load(parent_path)
    pilot = _load(pilot_path)
    if len(parent["records"]) != 500:
        raise ValueError(f"trajectory detector parent must contain 500 records, got {len(parent['records'])}")
    _verify_records(parent)
    _verify_records(pilot)
    records = sorted(parent["records"], key=lambda row: numeric_image_key(row["image_id"]))
    pilot_ids = {row["image_id"] for row in pilot["records"]}
    overlap = sorted({row["image_id"] for row in records} & pilot_ids, key=numeric_image_key)
    if overlap:
        raise ValueError(f"trajectory pilot overlap: {overlap[:5]}")

    parent_hash = sha256_file(parent_path)
    calibration = _child_manifest(parent, records[:250], "calibration", parent_hash)
    evaluation = _child_manifest(parent, records[250:], "evaluation", parent_hash)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "detector_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    (output_dir / "detector_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    audit = {
        "schema_version": "iqa_ssc_trajectory_detector_manifest_v1",
        "status": "frozen",
        "parent_manifest": "results/provenance/v3_fresh/evaluation.json",
        "parent_sha256": parent_hash,
        "calibration_manifest": "results/provenance/trajectory_detector/detector_calibration.json",
        "evaluation_manifest": "results/provenance/trajectory_detector/detector_evaluation.json",
        "calibration_records": len(calibration["records"]),
        "evaluation_records": len(evaluation["records"]),
        "calibration_ids": [row["image_id"] for row in calibration["records"]],
        "evaluation_ids": [row["image_id"] for row in evaluation["records"]],
        "pilot_overlap_count": len(overlap),
        "pilot_overlap_ids": overlap,
        "calibration_sha256": sha256_file(output_dir / "detector_calibration.json"),
        "evaluation_sha256": sha256_file(output_dir / "detector_evaluation.json"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=Path("results/provenance/v3_fresh/evaluation.json"))
    parser.add_argument("--pilot", type=Path, default=Path("results/provenance/v3_fresh/calibration.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/provenance/trajectory_detector"))
    args = parser.parse_args()
    print(json.dumps(freeze_manifests(args.parent, args.pilot, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
