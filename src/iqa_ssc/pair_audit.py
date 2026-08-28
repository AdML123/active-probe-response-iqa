"""Pure operations for the energy-matched scalar-score audit."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_matches(actual: str, expected: str) -> bool:
    return actual.casefold() == expected.casefold()


def read_runtime_version(module: object, package_name: str) -> str:
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version.strip():
        raise ValueError(f"runtime {package_name} module does not expose __version__")
    return version.strip()


def validate_runtime_contract(
    lock: dict,
    *,
    actual_backend_version: str,
    artifact_lock_path: Path,
    repository_root: Path,
    observed_configuration: dict,
) -> None:
    if lock.get("backend_package") != "pyiqa":
        raise ValueError("audit lock does not name the pyiqa backend")
    locked_version = str(lock.get("backend_version", ""))
    if actual_backend_version != locked_version:
        raise ValueError(
            f"runtime pyiqa version {actual_backend_version!r} does not match lock {locked_version!r}"
        )

    root = repository_root.resolve()
    artifact_record = lock.get("artifact_lock")
    if not isinstance(artifact_record, dict):
        raise ValueError("audit lock does not pin the artifact lock")
    expected_artifact_path = (root / str(artifact_record.get("path", ""))).resolve()
    if expected_artifact_path != artifact_lock_path.resolve():
        raise ValueError("artifact lock path does not match audit lock")
    if not artifact_lock_path.is_file():
        raise FileNotFoundError(artifact_lock_path)
    artifact_payload = json.loads(artifact_lock_path.read_text(encoding="utf-8"))
    if not hash_matches(
        canonical_json_sha256(artifact_payload),
        str(artifact_record.get("content_sha256", "")),
    ):
        raise ValueError("artifact lock hash does not match audit lock")
    artifact_version = str(
        artifact_payload.get("dependencies", {}).get("pyiqa", {}).get("version_target", "")
    )
    if artifact_version != actual_backend_version:
        raise ValueError("artifact lock pyiqa version does not match runtime")

    frozen_configuration = lock.get("frozen_configuration")
    if not isinstance(frozen_configuration, dict):
        raise ValueError("audit lock has no frozen configuration")
    frozen_fingerprint = str(lock.get("frozen_configuration_sha256", ""))
    if not hash_matches(canonical_json_sha256(frozen_configuration), frozen_fingerprint):
        raise ValueError("frozen configuration fingerprint does not match lock contents")
    if observed_configuration != frozen_configuration:
        raise ValueError("observed configuration does not match frozen configuration")

    implementation = lock.get("implementation", {})
    files = implementation.get("files") if isinstance(implementation, dict) else None
    if not isinstance(files, list) or not files:
        raise ValueError("audit lock has no implementation file fingerprints")
    for record in files:
        if not isinstance(record, dict):
            raise ValueError("invalid implementation fingerprint record")
        path = (root / str(record.get("path", ""))).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("implementation path escapes repository root") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        if not hash_matches(canonical_text_sha256(path), str(record.get("sha256", ""))):
            raise ValueError(f"implementation hash mismatch: {record.get('path', '')}")


def orient_quality_score(score: float, lower_better: bool) -> float:
    value = float(score)
    if not np.isfinite(value):
        raise ValueError("quality score must be finite")
    return -value if lower_better else value


def reversal_flag(oriented_bilateral: float, oriented_control: float) -> bool:
    return bool(float(oriented_bilateral) > float(oriented_control))


def validate_pair_record(record: dict) -> None:
    required = {
        "image_id", "selective_condition", "selective_index",
        "control_condition", "control_index", "energy_residual",
    }
    missing = required.difference(record)
    if missing:
        raise ValueError(f"missing pair fields: {sorted(missing)}")
    if record["selective_condition"] != "bilateral":
        raise ValueError("selective condition must be bilateral")
    if record["control_condition"] != "jpeg":
        raise ValueError("control condition must be jpeg")
    if int(record["selective_index"]) not in range(1, 6):
        raise ValueError("bilateral index must be one through five")
    if int(record["control_index"]) not in range(1, 6):
        raise ValueError("JPEG index must be one through five")
    residual = float(record["energy_residual"])
    if not np.isfinite(residual) or not 0.0 <= residual <= 0.05:
        raise ValueError("energy residual exceeds the frozen tolerance")


def validate_unique_pairs(records: Iterable[dict]) -> None:
    seen: set[tuple[str, int, int]] = set()
    used_controls: set[tuple[str, int]] = set()
    for record in records:
        validate_pair_record(record)
        key = (str(record["image_id"]), int(record["selective_index"]), int(record["control_index"]))
        if key in seen:
            raise ValueError(f"duplicate pair: {key}")
        control_key = (str(record["image_id"]), int(record["control_index"]))
        if control_key in used_controls:
            raise ValueError(f"control condition reused: {control_key}")
        seen.add(key)
        used_controls.add(control_key)


def validate_image_manifest(payload: dict, required_ids: set[str]) -> dict[str, dict]:
    if payload.get("status") != "frozen":
        raise ValueError("image manifest is not frozen")
    records = payload.get("records")
    if not isinstance(records, list) or int(payload.get("count", -1)) != len(records):
        raise ValueError("image manifest count is invalid")
    by_id: dict[str, dict] = {}
    for record in records:
        image_id = str(record.get("image_id", ""))
        if not image_id or image_id in by_id:
            raise ValueError(f"duplicate or missing image id: {image_id}")
        if not record.get("relative_path") or not record.get("sha256"):
            raise ValueError(f"incomplete image record: {image_id}")
        by_id[image_id] = record
    missing = required_ids.difference(by_id)
    if missing:
        raise ValueError(f"missing required images: {sorted(missing)}")
    return by_id


def bootstrap_reversal_rate(rows: list[dict], seed: int, n_resamples: int) -> tuple[float, float]:
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    groups: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        if not row.get("valid", True):
            continue
        groups[str(row["image_id"])].append(bool(row["reversal"]))
    image_ids = sorted(groups)
    if not image_ids:
        raise ValueError("at least one valid image group is required")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        sampled = rng.integers(0, len(image_ids), size=len(image_ids))
        flags = [flag for item in sampled for flag in groups[image_ids[item]]]
        estimates[index] = float(np.mean(flags))
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)
