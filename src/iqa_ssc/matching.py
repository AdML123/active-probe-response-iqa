"""Energy-only matching of selective and uniform transformation records."""

from __future__ import annotations

from collections import defaultdict
import json
from typing import Iterable


def _valid(row: dict[str, object]) -> bool:
    return row.get("invalid_reason") is None and isinstance(row.get("delta_global"), (int, float))


def match_records(
    records: Iterable[dict[str, object]],
    *,
    selective: str = "bilateral",
    control: str = "jpeg",
    tolerance: float = 0.05,
) -> list[dict[str, object]]:
    """Greedily match each selective level to one nearest unused control level."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[str(row["image_id"])][str(row["condition"])].append(row)
    matches: list[dict[str, object]] = []
    for image_id in sorted(grouped):
        selective_rows = sorted(grouped[image_id].get(selective, []), key=lambda row: int(row["index"]))
        controls = sorted(grouped[image_id].get(control, []), key=lambda row: int(row["index"]))
        available = list(controls)
        for selected in selective_rows:
            if not _valid(selected):
                continue
            candidates = [row for row in available if _valid(row)]
            if not candidates:
                continue
            selected_energy = float(selected["delta_global"])
            chosen = min(candidates, key=lambda row: (round(abs(selected_energy - float(row["delta_global"])), 12), int(row["index"]), str(row["image_id"])))
            residual = abs(selected_energy - float(chosen["delta_global"]))
            if residual > tolerance + 1e-12:
                continue
            available.remove(chosen)
            matches.append({
                "image_id": image_id,
                "selective_condition": selective,
                "selective_index": int(selected["index"]),
                "control_condition": control,
                "control_index": int(chosen["index"]),
                "energy_residual": residual,
                "selective_delta_global": float(selected["delta_global"]),
                "control_delta_global": float(chosen["delta_global"]),
                "selective_delta_skin": float(selected["delta_skin"]),
                "control_delta_skin": float(chosen["delta_skin"]),
                "selective_sc": float(selected["sc"]),
                "control_sc": float(chosen["sc"]),
            })
    return matches


def read_jsonl(path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
