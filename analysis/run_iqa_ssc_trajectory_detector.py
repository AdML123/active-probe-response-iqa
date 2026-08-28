"""Generate fixed active challenge-response trajectories for the formal detector split."""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from iqa_ssc.transforms import BLUR_SIGMAS, JPEG_QUALITIES, apply_condition


PROBE_GRID = (
    ("jpeg", 0),
    ("jpeg", 1),
    ("jpeg", 2),
    ("jpeg", 3),
    ("jpeg", 4),
    ("gaussian_blur", 0),
    ("gaussian_blur", 1),
    ("gaussian_blur", 2),
    ("gaussian_blur", 3),
    ("gaussian_blur", 4),
)
BASE_GRID = tuple(("bilateral", index) for index in range(5)) + tuple(("jpeg", index) for index in range(5)) + tuple(("gaussian_blur", index) for index in range(5))
MIN_REGION_PIXELS = 100


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def gray_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gray = image[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    gray /= 255.0
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return gray, gx, gy, np.hypot(gx, gy)


def fixed_masks(magnitude: np.ndarray) -> dict[str, np.ndarray]:
    interior = np.ones_like(magnitude, dtype=bool)
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    values = magnitude[interior]
    return {
        "edge": interior & (magnitude >= np.quantile(values, 0.90)),
        "texture": interior & (magnitude <= np.quantile(values, 0.50)),
        "high_gradient": interior & (magnitude >= np.quantile(values, 0.80)),
    }


def boundary_jumps(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.abs(gray[:, 1:] - gray[:, :-1])
    vertical = np.abs(gray[1:, :] - gray[:-1, :])
    h_coords = np.arange(1, gray.shape[1])
    v_coords = np.arange(1, gray.shape[0])
    boundary = np.concatenate([horizontal[:, h_coords % 8 == 0].ravel(), vertical[v_coords % 8 == 0, :].ravel()])
    interior = np.concatenate([horizontal[:, h_coords % 8 != 0].ravel(), vertical[v_coords % 8 != 0, :].ravel()])
    return boundary, interior


def goc(pre_gx: np.ndarray, pre_gy: np.ndarray, post_gx: np.ndarray, post_gy: np.ndarray, mask: np.ndarray) -> float:
    pre_norm = np.hypot(pre_gx, pre_gy)
    post_norm = np.hypot(post_gx, post_gy)
    used = mask & (pre_norm > 1e-8) & (post_norm > 1e-8)
    if int(np.sum(used)) < MIN_REGION_PIXELS:
        raise ValueError("insufficient edge pixels for GOC")
    cosine = (pre_gx[used] * post_gx[used] + pre_gy[used] * post_gy[used]) / (pre_norm[used] * post_norm[used])
    value = float(np.mean(cosine))
    if not np.isfinite(value):
        raise ValueError("non-finite GOC")
    return value


def s_grid(pre_gray: np.ndarray, post_gray: np.ndarray) -> float:
    pre_boundary, pre_interior = boundary_jumps(pre_gray)
    post_boundary, post_interior = boundary_jumps(post_gray)
    if min(pre_boundary.size, pre_interior.size, post_boundary.size, post_interior.size) < MIN_REGION_PIXELS:
        raise ValueError("insufficient grid samples")
    epsilon_grid = 1e-6
    jga = np.log((np.median(post_boundary) + epsilon_grid) / (np.median(pre_boundary) + epsilon_grid)) - np.log((np.median(post_interior) + epsilon_grid) / (np.median(pre_interior) + epsilon_grid))
    value = float(-jga)
    if not np.isfinite(value):
        raise ValueError("non-finite S_grid")
    return value


def curve_stats(values: list[float] | np.ndarray) -> list[float]:
    curve = np.asarray(values, dtype=float)
    if curve.size != 5 or not np.all(np.isfinite(curve)):
        raise ValueError("trajectory curve must contain five finite values")
    return [float(np.mean(np.diff(curve[:3]))), float(np.mean(np.diff(curve[2:]))), float(np.trapezoid(curve, np.arange(5, dtype=float)))]


def fixed_features(row: dict[str, object]) -> list[float]:
    goc_values = list(row["goc_trajectory"])
    grid_values = list(row["s_grid_trajectory"])
    if len(goc_values) != 10 or len(grid_values) != 10:
        raise ValueError("trajectory must contain five JPEG and five blur probe values")
    return curve_stats(goc_values[:5]) + curve_stats(goc_values[5:]) + curve_stats(grid_values[:5]) + curve_stats(grid_values[5:])


def condition_name(condition: str, index: int) -> str:
    if condition == "bilateral":
        return f"bilateral_L{index + 1}"
    if condition == "jpeg":
        return f"jpeg_Q{JPEG_QUALITIES[index]}"
    if condition == "gaussian_blur":
        return f"blur_sigma{BLUR_SIGMAS[index]}"
    raise ValueError(condition)


def process_image(image_id: str, image_root: Path) -> list[dict[str, object]]:
    image_path = image_root / image_id
    original = load_rgb(image_path)
    source_hash = sha256_file(image_path)
    rows: list[dict[str, object]] = []
    for base_condition, base_index in BASE_GRID:
        base = apply_condition(original, base_condition, base_index)
        base_gray, base_gx, base_gy, base_magnitude = gray_gradient(base)
        masks = fixed_masks(base_magnitude)
        goc_values: list[float] = []
        grid_values: list[float] = []
        invalid_reason: str | None = None
        for probe_condition, probe_index in PROBE_GRID:
            try:
                probe = apply_condition(base, probe_condition, probe_index)
                probe_gray, probe_gx, probe_gy, _ = gray_gradient(probe)
                goc_values.append(goc(base_gx, base_gy, probe_gx, probe_gy, masks["edge"]))
                grid_values.append(s_grid(base_gray, probe_gray))
            except (ValueError, FloatingPointError) as exc:
                invalid_reason = str(exc)
                break
        row: dict[str, object] = {
            "schema_version": "iqa_ssc_trajectory_detector_v1",
            "image_id": image_id,
            "source_sha256": source_hash,
            "base_family": base_condition,
            "base_index": base_index + 1,
            "base_name": condition_name(base_condition, base_index),
            "probe_grid": [condition_name(condition, index) for condition, index in PROBE_GRID],
            "invalid_reason": invalid_reason,
            "edge_pixels": int(np.sum(masks["edge"])),
            "goc_trajectory": goc_values,
            "s_grid_trajectory": grid_values,
        }
        row["fixed_features_v1"] = fixed_features(row) if invalid_reason is None else None
        rows.append(row)
    return rows


def process_record(record: dict[str, object], image_root: str) -> list[dict[str, object]]:
    root = Path(image_root)
    image_path = root / str(record["image_id"])
    if sha256_file(image_path) != record["sha256"]:
        raise ValueError(f"source hash mismatch: {image_path}")
    return process_image(str(record["image_id"]), root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"][args.start:] if args.count is None else manifest["records"][args.start : args.start + args.count]
    if args.workers < 1:
        raise ValueError("workers must be at least one")
    if args.workers == 1:
        batches = [process_record(record, str(args.image_root)) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            batches = list(executor.map(process_record, records, [str(args.image_root)] * len(records)))
    rows = [row for batch in batches for row in batch]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
