"""Run a small deterministic pyiqa conformance panel on a fixed manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iqa_ssc.learned_iqa import import_pyiqa, score_in_batches


METRICS = (
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


def _load_images(image_root: Path, manifest: Path, limit: int):
    import numpy as np
    import torch
    from PIL import Image

    records = json.loads(manifest.read_text(encoding="utf-8"))["records"][:limit]
    tensors = []
    ids = []
    for record in records:
        image_id = record["image_id"]
        image = np.array(Image.open(image_root / record["relative_path"]).convert("RGB"), copy=True)
        tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        tensors.append(tensor)
        ids.append(image_id)
    return torch.stack(tensors), ids


def _digest(values):
    payload = json.dumps(values, separators=(",", ":"), sort_keys=False).encode()
    return hashlib.sha256(payload).hexdigest()


def run(
    *,
    image_root: Path,
    manifest: Path,
    device: str,
    limit: int = 20,
    batch_size: int = 1,
    metrics: tuple[str, ...] = METRICS,
):
    import gc
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pyiqa = import_pyiqa()

    images, image_ids = _load_images(image_root, manifest, limit)
    results = {"image_ids": image_ids, "device": device, "metrics": {}}
    for name in metrics:
        item = {"pyiqa_name": name}
        try:
            metric = pyiqa.create_metric(name, device=device)
            item["class"] = metric.net.__class__.__name__
            item["lower_better"] = bool(getattr(metric, "lower_better"))
            first = score_in_batches(metric, images, device=device, batch_size=batch_size)
            second = score_in_batches(metric, images, device=device, batch_size=batch_size)
            item["scores"] = first
            item["repeat_scores"] = second
            item["finite"] = all(__import__("math").isfinite(value) for value in first + second)
            item["repeat_max_abs_diff"] = max(abs(a - b) for a, b in zip(first, second))
            item["output_sha256"] = _digest(first)
            item["repeat_output_sha256"] = _digest(second)
            item["status"] = "pyiqa_validated" if item["finite"] and item["repeat_max_abs_diff"] <= 1e-6 else "baseline_unavailable_detail"
        except Exception as exc:
            item.update(
                {
                    "status": "baseline_unavailable_detail",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "finite": False,
                }
            )
        finally:
            if "metric" in locals():
                del metric
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results["metrics"][name] = item
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--metrics", nargs="*", default=list(METRICS), choices=METRICS)
    args = parser.parse_args()
    result = run(
        image_root=args.image_root,
        manifest=args.manifest,
        device=args.device,
        limit=args.limit,
        batch_size=args.batch_size,
        metrics=tuple(args.metrics),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        name: {
            key: item.get(key)
            for key in ("status", "class", "lower_better", "finite", "repeat_max_abs_diff", "error_type")
            if key in item
        }
        for name, item in result["metrics"].items()
    }
    print(json.dumps(summary, indent=2))
    return 0 if all(item["status"] == "pyiqa_validated" for item in result["metrics"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
