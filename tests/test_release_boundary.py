from __future__ import annotations

import json
import os
from pathlib import Path

from scripts.generate_synthetic_demo import generate_demo


PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# These files are local forensic preflight scratch outputs. They are ignored by
# the release tree and must not be allowed to make the working-tree audit differ
# from the clean archive exported from Git.
LOCAL_ONLY_FILES = {
    "results/strong-baselines/forensic-adapters-preflight-20.json",
    "results/strong-baselines/forensic-adapters-preflight-20.jsonl",
    "results/strong-baselines/forensic-adapters-rows.jsonl",
}


def _release_files() -> list[Path]:
    files: list[Path] = []

    def onerror(_error: OSError) -> None:
        return None

    for root, dirs, names in os.walk(PACKAGE_ROOT, onerror=onerror):
        dirs[:] = [
            name
            for name in dirs
            if not name.startswith(".pytest-")
            and not name.endswith(".egg-info")
            and name not in {".git", ".pytest_cache", "__pycache__"}
        ]
        for name in names:
            path = Path(root, name)
            if path.relative_to(PACKAGE_ROOT).as_posix() not in LOCAL_ONLY_FILES:
                files.append(path)
    return files


def test_release_excludes_manuscript_data_and_weights() -> None:
    forbidden = {".tex", ".bib", ".bst", ".sty", ".cls", ".pdf", ".pt", ".pth", ".ckpt", ".safetensors"}
    offenders = [path.relative_to(PACKAGE_ROOT).as_posix() for path in _release_files() if path.suffix.lower() in forbidden]
    assert offenders == []
    assert not (PACKAGE_ROOT / "submission").exists()
    assert not (PACKAGE_ROOT / "manuscript").exists()


def test_lineage_declares_staged_boundary() -> None:
    lineage = json.loads((PACKAGE_ROOT / "results" / "release-lineage-v0.2.0.json").read_text(encoding="utf-8"))
    assert lineage["release_version"] == "v0.2.0"
    assert lineage["release_status"] == "staged_not_published"
    assert lineage["version_doi"] is None
    assert lineage["concept_doi"] == "10.5281/zenodo.22098907"
    assert lineage["synthetic_demo"]["generated_outputs_tracked"] is False


def test_release_versions_and_concept_doi_agree() -> None:
    pyproject = (PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    citation = (PACKAGE_ROOT / "CITATION.cff").read_text(encoding="utf-8")
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    assert 'version = "0.2.0"' in pyproject
    assert "version: 0.2.0" in citation
    assert "v0.2.0" in readme
    assert "10.5281/zenodo.22098907" in citation
    assert "\ndoi:" not in citation


def test_manifest_matches_release_files() -> None:
    manifest_path = PACKAGE_ROOT / "MANIFEST.sha256"
    listed = {line.split("  ", 1)[1] for line in manifest_path.read_text(encoding="ascii").splitlines() if line.strip()}
    actual = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for path in _release_files()
        if path != manifest_path
    }
    assert listed == actual


def test_synthetic_demo_is_deterministic_and_weight_free(tmp_path: Path) -> None:
    first = generate_demo(tmp_path / "first", count=2)
    second = generate_demo(tmp_path / "second", count=2)
    assert first == second
    assert first["weights_required"] is False
    assert [record["sha256"] for record in first["records"]] == [record["sha256"] for record in second["records"]]
    manifest = json.loads((tmp_path / "first" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "iqa_ssc_synthetic_demo_v1"
