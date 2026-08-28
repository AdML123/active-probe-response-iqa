"""Record whether the external IQA-PyTorch backend can be constructed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iqa_ssc.learned_iqa import import_pyiqa


CANDIDATES = (
    "brisque",
    "niqe",
    "piqe",
    "musiq",
    "maniqa",
    "topiq_nr",
    "liqe",
    "arniqa",
    "clipiqa",
)


def classify_backend(*, available: dict[str, bool]) -> dict[str, object]:
    missing = sorted(name for name, present in available.items() if not present)
    return {
        "status": "blocked_iqa_library_backend" if missing else "ready_for_metric_construction",
        "missing_modules": missing,
        "candidate_metrics": list(CANDIDATES),
        "results_written": False,
        "reason": (
            "Required external runtime modules are not installed; no learned-score result was produced."
            if missing
            else "Runtime imports are available; metric construction and 20-image repeat checks remain required."
        ),
    }


def probe() -> dict[str, object]:
    available = {
        name: importlib.util.find_spec(name) is not None
        for name in ("pyiqa", "torch", "sklearn")
    }
    summary = classify_backend(available=available)
    summary["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["available_modules"] = available
    return summary


def construct_metrics(*, device: str = "cuda") -> dict[str, object]:
    """Construct every registered metric and retain per-model failures."""
    import gc
    import torch

    pyiqa = import_pyiqa()

    results: dict[str, object] = {}
    for name in CANDIDATES:
        try:
            metric = pyiqa.create_metric(name, device=device)
            results[name] = {
                "status": "constructed",
                "class": metric.net.__class__.__name__,
                "lower_better": bool(getattr(metric, "lower_better")),
            }
        except Exception as exc:  # model-specific downloads/configs may fail independently
            results[name] = {
                "status": "baseline_unavailable_detail",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        finally:
            if "metric" in locals():
                del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/strong-baselines/conformance-summary.json"),
    )
    parser.add_argument("--construct", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    summary = probe()
    if args.construct and summary["status"] != "blocked_iqa_library_backend":
        summary["metric_construction"] = construct_metrics(device=args.device)
        failed = [
            item for item in summary["metric_construction"].values()
            if item["status"] != "constructed"
        ]
        summary["construction_status"] = "all_constructed" if not failed else "partial"
        summary["results_written"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "ready_for_metric_construction" else 2


if __name__ == "__main__":
    raise SystemExit(main())
