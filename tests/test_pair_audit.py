from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from iqa_ssc import pair_audit as pair_audit_module
from iqa_ssc.pair_audit import (
    bootstrap_reversal_rate,
    canonical_json_sha256,
    canonical_text_sha256,
    read_runtime_version,
    orient_quality_score,
    reversal_flag,
    validate_runtime_contract,
    validate_image_manifest,
    validate_pair_record,
    validate_unique_pairs,
)


def runtime_contract(tmp_path):
    artifact_lock_path = tmp_path / "artifact-lock.json"
    artifact_lock_path.write_text(
        json.dumps({"dependencies": {"pyiqa": {"version_target": "0.1.13"}}}),
        encoding="utf-8",
    )
    evaluator_path = tmp_path / "evaluator.py"
    evaluator_path.write_text("print('frozen')\n", encoding="utf-8")
    configuration = {
        "metrics": ["musiq", "maniqa"],
        "runtime_parameters": {"device": "cuda", "batch_size": 4},
    }
    lock = {
        "backend_package": "pyiqa",
        "backend_version": "0.1.13",
        "artifact_lock": {
            "path": "artifact-lock.json",
            "content_sha256": canonical_json_sha256(
                json.loads(artifact_lock_path.read_text(encoding="utf-8"))
            ),
        },
        "implementation": {
            "files": [
                {"path": "evaluator.py", "sha256": canonical_text_sha256(evaluator_path)},
            ],
        },
        "frozen_configuration": configuration,
        "frozen_configuration_sha256": canonical_json_sha256(configuration),
    }
    return lock, artifact_lock_path, configuration


def pair(**overrides):
    record = {
        "image_id": "00001.jpg",
        "selective_condition": "bilateral",
        "selective_index": 1,
        "control_condition": "jpeg",
        "control_index": 2,
        "energy_residual": 0.02,
    }
    record.update(overrides)
    return record


def test_score_direction_and_ties():
    assert orient_quality_score(0.3, False) == pytest.approx(0.3)
    assert orient_quality_score(0.3, True) == pytest.approx(-0.3)
    assert reversal_flag(0.3, 0.2)
    assert not reversal_flag(0.3, 0.3)
    with pytest.raises(ValueError):
        orient_quality_score(float("nan"), False)


def test_hash_comparison_is_case_insensitive():
    assert pair_audit_module.hash_matches("AB12", "ab12")


def test_canonical_json_fingerprint_is_order_independent():
    assert canonical_json_sha256({"a": 1, "b": [2, 3]}) == canonical_json_sha256(
        {"b": [2, 3], "a": 1}
    )
    assert canonical_json_sha256({"a": 1}) != canonical_json_sha256({"a": 2})


def test_source_fingerprint_normalizes_line_endings(tmp_path):
    source = tmp_path / "source.py"
    source.write_bytes(b"one\r\ntwo\r\n")
    windows_hash = canonical_text_sha256(source)
    source.write_bytes(b"one\ntwo\n")
    assert canonical_text_sha256(source) == windows_hash


def test_runtime_version_is_read_from_imported_backend():
    assert read_runtime_version(SimpleNamespace(__version__="0.1.13"), "pyiqa") == "0.1.13"
    with pytest.raises(ValueError, match="does not expose"):
        read_runtime_version(SimpleNamespace(), "pyiqa")


def test_runtime_contract_accepts_matching_backend_locks_and_implementation(tmp_path):
    lock, artifact_lock_path, configuration = runtime_contract(tmp_path)
    validate_runtime_contract(
        lock,
        actual_backend_version="0.1.13",
        artifact_lock_path=artifact_lock_path,
        repository_root=tmp_path,
        observed_configuration=configuration,
    )


def test_runtime_contract_rejects_backend_version_mismatch(tmp_path):
    lock, artifact_lock_path, configuration = runtime_contract(tmp_path)
    with pytest.raises(ValueError, match="runtime pyiqa version"):
        validate_runtime_contract(
            lock,
            actual_backend_version="0.2.0",
            artifact_lock_path=artifact_lock_path,
            repository_root=tmp_path,
            observed_configuration=configuration,
        )


def test_runtime_contract_rejects_artifact_lock_or_implementation_change(tmp_path):
    lock, artifact_lock_path, configuration = runtime_contract(tmp_path)
    artifact_lock_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact lock hash"):
        validate_runtime_contract(
            lock,
            actual_backend_version="0.1.13",
            artifact_lock_path=artifact_lock_path,
            repository_root=tmp_path,
            observed_configuration=configuration,
        )

    lock, artifact_lock_path, configuration = runtime_contract(tmp_path)
    (tmp_path / "evaluator.py").write_text("print('changed')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="implementation hash"):
        validate_runtime_contract(
            lock,
            actual_backend_version="0.1.13",
            artifact_lock_path=artifact_lock_path,
            repository_root=tmp_path,
            observed_configuration=configuration,
        )


def test_runtime_contract_rejects_configuration_change(tmp_path):
    lock, artifact_lock_path, configuration = runtime_contract(tmp_path)
    changed = {**configuration, "runtime_parameters": {"device": "cpu", "batch_size": 4}}
    with pytest.raises(ValueError, match="configuration"):
        validate_runtime_contract(
            lock,
            actual_backend_version="0.1.13",
            artifact_lock_path=artifact_lock_path,
            repository_root=tmp_path,
            observed_configuration=changed,
        )


def test_pair_validation_and_duplicate_rejection():
    validate_pair_record(pair())
    with pytest.raises(ValueError):
        validate_pair_record(pair(energy_residual=0.051))
    with pytest.raises(ValueError):
        validate_unique_pairs([pair(), pair()])
    with pytest.raises(ValueError, match="control condition reused"):
        validate_unique_pairs([pair(control_index=2), pair(image_id="00001.jpg", selective_index=2, control_index=2)])


def test_bootstrap_is_deterministic_and_image_grouped():
    rows = [
        {"image_id": "a", "reversal": True, "valid": True},
        {"image_id": "a", "reversal": False, "valid": True},
        {"image_id": "b", "reversal": True, "valid": True},
        {"image_id": "c", "reversal": False, "valid": True},
    ]
    first = bootstrap_reversal_rate(rows, seed=20260827, n_resamples=100)
    second = bootstrap_reversal_rate(rows, seed=20260827, n_resamples=100)
    assert first == second
    assert 0.0 <= first[0] <= first[1] <= 1.0


def test_image_manifest_covers_required_ids_without_duplicates():
    payload = {
        "status": "frozen",
        "count": 2,
        "records": [
            {"image_id": "a.jpg", "relative_path": "a.jpg", "sha256": "aa"},
            {"image_id": "b.jpg", "relative_path": "b.jpg", "sha256": "bb"},
        ],
    }
    records = validate_image_manifest(payload, {"a.jpg"})
    assert records["a.jpg"]["sha256"] == "aa"
    with pytest.raises(ValueError, match="missing required images"):
        validate_image_manifest(payload, {"c.jpg"})
