import json
from pathlib import Path


CORE = ("brisque", "niqe", "piqe", "musiq", "maniqa")
EXTENSIONS = ("topiq_nr", "liqe", "arniqa", "clipiqa")


def _lock():
    return json.loads(
        Path("results/strong-baselines/artifact-lock.json").read_text(
            encoding="utf-8"
        )
    )


def test_candidate_registry_has_core_and_extension_tiers():
    lock = _lock()
    assert set(CORE) | set(EXTENSIONS) <= set(lock["candidate_registry"])
    for name in CORE:
        assert lock["candidate_registry"][name]["tier"] == "core"
    for name in EXTENSIONS:
        assert lock["candidate_registry"][name]["tier"] == "extension"


def test_external_dependency_and_weights_stay_outside_public_package():
    lock = _lock()
    assert lock["dependencies"]["pyiqa"]["public_payload"] is False
    for method in CORE + EXTENSIONS:
        item = lock["methods"][method]
        assert item["weights_in_public_package"] is False
        assert "public-repro-package" not in json.dumps(item).replace("\\", "/")


def test_backend_status_is_explicit_when_runtime_is_missing():
    lock = _lock()
    for method in CORE + EXTENSIONS:
        assert lock["methods"][method]["implementation_status"] in {
            "blocked_iqa_library_backend",
            "pyiqa_validated",
            "pyiqa_validated_20_image_repeat",
            "baseline_unavailable_detail",
        }


def test_conformance_output_is_not_paper_result():
    summary_path = Path("results/strong-baselines/conformance-summary-paper20.json")
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary.get("results_written") is False
