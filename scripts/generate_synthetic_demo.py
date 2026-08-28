"""Generate a tiny deterministic image manifest for a weight-free smoke test.

The demo is deliberately independent of CelebA-HQ, parsing masks, IQA model
weights, and network access.  Its images are only test inputs; they are not
part of the scientific evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_SEED = 20260828
DEFAULT_SIZE = (128, 128)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image(index: int, *, size: tuple[int, int], seed: int) -> np.ndarray:
    height, width = size
    rng = np.random.default_rng(seed + index)
    y, x = np.mgrid[:height, :width]
    phase = 0.17 * index
    smooth = 0.5 + 0.25 * np.sin((x + 3 * y) / 11.0 + phase)
    smooth += 0.15 * np.cos((2 * x - y) / 17.0 - phase)
    checker = ((x // 8 + y // 8 + index) % 2).astype(float)
    radius = np.hypot(x - width * (0.35 + 0.2 * index), y - height * 0.52)
    circle = np.clip(1.0 - radius / (height * 0.34), 0.0, 1.0)
    noise = rng.normal(0.0, 0.018, size=(height, width))
    channels = np.stack(
        [smooth + 0.16 * checker, smooth * 0.82 + 0.18 * circle, smooth * 0.65 + 0.25 * (1.0 - checker)],
        axis=-1,
    )
    return np.rint(np.clip(channels + noise[..., None], 0.0, 1.0) * 255.0).astype(np.uint8)


def generate_demo(output_dir: Path, *, count: int = 2, seed: int = DEFAULT_SEED, size: tuple[int, int] = DEFAULT_SIZE) -> dict[str, object]:
    """Write deterministic PNG inputs and a detector-compatible manifest."""

    if count < 1:
        raise ValueError("count must be positive")
    if min(size) < 32:
        raise ValueError("demo images must be at least 32 pixels per side")
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index in range(count):
        image_id = f"synthetic_{index:03d}.png"
        path = output_dir / image_id
        if not cv2.imwrite(str(path), cv2.cvtColor(_image(index, size=size, seed=seed), cv2.COLOR_RGB2BGR)):
            raise OSError(f"failed to write {path}")
        records.append({"image_id": image_id, "sha256": _sha256(path), "height": size[0], "width": size[1]})
    manifest: dict[str, object] = {
        "schema_version": "iqa_ssc_synthetic_demo_v1",
        "seed": seed,
        "image_size": list(size),
        "records": records,
        "scientific_data": False,
        "weights_required": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(generate_demo(args.output_dir, count=args.count, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
