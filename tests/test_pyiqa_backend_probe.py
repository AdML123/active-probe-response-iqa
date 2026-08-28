from analysis.validate_pyiqa_backend import classify_backend


def test_missing_modules_produce_explicit_blocking_status():
    summary = classify_backend(
        available={"pyiqa": False, "torch": False, "sklearn": False}
    )
    assert summary["status"] == "blocked_iqa_library_backend"
    assert set(summary["missing_modules"]) == {"pyiqa", "torch", "sklearn"}
    assert summary["results_written"] is False


def test_present_modules_allow_construction_stage():
    summary = classify_backend(
        available={"pyiqa": True, "torch": True, "sklearn": True}
    )
    assert summary["status"] == "ready_for_metric_construction"
    assert summary["missing_modules"] == []
