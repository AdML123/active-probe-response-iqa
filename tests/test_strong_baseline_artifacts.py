import json
from pathlib import Path


REQUIRED = {
    "ding2019",
    "shehin2022",
    "brisque",
    "niqe",
    "piqe",
    "musiq",
    "maniqa",
    "topiq_nr",
    "liqe",
    "arniqa",
    "clipiqa",
}


def _lock():
    return json.loads(
        Path("results/strong-baselines/artifact-lock.json").read_text(
            encoding="utf-8"
        )
    )


def test_artifact_lock_has_required_methods():
    lock = _lock()
    assert REQUIRED <= set(lock["methods"])
    for item in lock["artifacts"]:
        assert len(item["sha256"]) == 64
        assert item["license_status"] in {"paper-only", "redistributable", "local-use-only"}


def test_artifact_lock_keeps_restricted_payloads_out_of_public_tree():
    lock = _lock()
    assert all(not item["public_payload"] for item in lock["artifacts"])
    assert all("local_path" not in item for item in lock["artifacts"])
