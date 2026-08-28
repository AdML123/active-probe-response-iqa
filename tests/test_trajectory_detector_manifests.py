import hashlib
import json
from pathlib import Path

import pytest

from scripts.freeze_iqa_ssc_trajectory_detector import freeze_manifests, numeric_image_key


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, root: Path, ids: list[str]) -> None:
    records = []
    for image_id in ids:
        source = root / image_id
        source.write_bytes(image_id.encode())
        records.append({"image_id": image_id, "relative_path": image_id, "sha256": _sha256(source)})
    path.write_text(json.dumps({"asset_root": str(root), "count": len(records), "records": records}), encoding="utf-8")


def test_numeric_image_order_uses_integer_stem_then_filename():
    assert sorted(["10.jpg", "2.jpg", "2.png", "1.jpg"], key=numeric_image_key) == ["1.jpg", "2.jpg", "2.png", "10.jpg"]


def test_freeze_manifests_writes_disjoint_250_record_children(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    parent = tmp_path / "evaluation.json"
    pilot = tmp_path / "pilot.json"
    ids = [f"{index:04d}.jpg" for index in range(500)]
    _write_manifest(parent, root, ids)
    _write_manifest(pilot, root, [f"{index:04d}.jpg" for index in range(500, 600)])

    result = freeze_manifests(parent, pilot, tmp_path / "out")

    calibration = json.loads((tmp_path / "out" / "detector_calibration.json").read_text(encoding="utf-8"))
    evaluation = json.loads((tmp_path / "out" / "detector_evaluation.json").read_text(encoding="utf-8"))
    assert len(calibration["records"]) == 250
    assert len(evaluation["records"]) == 250
    assert {row["image_id"] for row in calibration["records"]}.isdisjoint(row["image_id"] for row in evaluation["records"])
    assert result["pilot_overlap_count"] == 0
    assert result["calibration_records"] == 250
    assert result["evaluation_records"] == 250


def test_freeze_manifests_rejects_source_hash_mismatch(tmp_path):
    root = tmp_path / "images"
    root.mkdir()
    parent = tmp_path / "evaluation.json"
    pilot = tmp_path / "pilot.json"
    _write_manifest(parent, root, [f"{index:04d}.jpg" for index in range(500)])
    _write_manifest(pilot, root, [f"{index:04d}.jpg" for index in range(500, 501)])
    data = json.loads(parent.read_text(encoding="utf-8"))
    data["records"][0]["sha256"] = "0" * 64
    parent.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="source hash mismatch"):
        freeze_manifests(parent, pilot, tmp_path / "out")
