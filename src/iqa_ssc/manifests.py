"""Freeze immutable IQA-SSC image manifests and provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class FrozenManifests:
    calibration: Path
    evaluation: Path
    pilot: Path


def _numeric_key(image_id: str) -> tuple[int, str]:
    stem = Path(image_id).stem
    try:
        return int(stem), image_id
    except ValueError as exc:
        raise ValueError(f"image ID is not numerically sortable: {image_id}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload.get("ids") if isinstance(payload, dict) else payload
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
        raise ValueError(f"manifest must contain unique non-empty ids: {path}")
    return [str(image_id) for image_id in ids]


def _write_manifest(path: Path, *, role: str, ids: list[str], source: Path, seed: int) -> None:
    records = []
    for image_id in ids:
        image_path = source / image_id
        if not image_path.is_file():
            raise FileNotFoundError(f"missing declared image: {image_path}")
        records.append({
            "image_id": image_id,
            "relative_path": image_id,
            "file_size": image_path.stat().st_size,
            "sha256": _sha256(image_path),
        })
    payload = {
        "status": "frozen",
        "dataset": "CelebA-HQ",
        "role": role,
        "seed": seed,
        "asset_root": source.resolve().as_posix(),
        "count": len(records),
        "records": records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def freeze_manifests(
    *,
    asset_root: Path,
    test_manifest: Path,
    output_dir: Path,
    historical_manifest: Path | None = None,
    seed: int = 124,
    pilot_count: int = 20,
    calibration_count: int = 500,
    evaluation_count: int = 500,
) -> FrozenManifests:
    """Freeze split manifests without network access or adaptive selection."""

    asset_root = asset_root.resolve()
    if not asset_root.is_dir():
        raise NotADirectoryError(f"asset root does not exist: {asset_root}")
    test_ids = sorted(_load_ids(test_manifest), key=_numeric_key)
    if len(test_ids) != calibration_count + evaluation_count:
        raise ValueError("test manifest size must equal calibration_count + evaluation_count")
    if historical_manifest is None:
        historical_ids: set[str] = set()
    else:
        historical_ids = set(_load_ids(historical_manifest))
    all_ids = sorted((path.name for path in asset_root.iterdir() if path.is_file()), key=_numeric_key)
    if len(all_ids) < len(test_ids) + pilot_count:
        raise ValueError("asset root does not contain enough images for disjoint pilot")
    test_set = set(test_ids)
    pilot_pool = [image_id for image_id in all_ids if image_id not in test_set and image_id not in historical_ids]
    if len(pilot_pool) < pilot_count:
        raise ValueError("asset root does not contain enough eligible pilot images")
    pilot_ids = sorted(np.random.default_rng(seed).choice(np.asarray(pilot_pool), size=pilot_count, replace=False).tolist(), key=_numeric_key)
    calibration_ids = test_ids[:calibration_count]
    evaluation_ids = test_ids[calibration_count:]
    if set(calibration_ids) & set(evaluation_ids) or set(pilot_ids) & test_set:
        raise AssertionError("frozen manifests are not disjoint")
    output_dir = output_dir.resolve()
    calibration_path = output_dir / "calibration.json"
    evaluation_path = output_dir / "evaluation.json"
    pilot_path = output_dir / "pilot.json"
    _write_manifest(calibration_path, role="calibration", ids=calibration_ids, source=asset_root, seed=seed)
    _write_manifest(evaluation_path, role="evaluation", ids=evaluation_ids, source=asset_root, seed=seed)
    _write_manifest(pilot_path, role="pilot", ids=pilot_ids, source=asset_root, seed=seed)
    return FrozenManifests(calibration_path, evaluation_path, pilot_path)

