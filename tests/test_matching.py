from __future__ import annotations

from iqa_ssc.matching import match_records


def _row(image_id: str, condition: str, index: int, energy: float) -> dict[str, object]:
    return {
        "image_id": image_id,
        "condition": condition,
        "index": index,
        "delta_global": energy,
        "delta_skin": energy / 2,
        "sc": 0.0,
        "invalid_reason": None,
    }


def test_matches_nearest_control_within_tolerance_and_one_use() -> None:
    rows = [
        _row("1.jpg", "bilateral", 1, 0.10),
        _row("1.jpg", "bilateral", 2, 0.20),
        _row("1.jpg", "jpeg", 1, 0.11),
        _row("1.jpg", "jpeg", 2, 0.21),
    ]
    pairs = match_records(rows, tolerance=0.02)
    assert [(pair["selective_index"], pair["control_index"]) for pair in pairs] == [(1, 1), (2, 2)]


def test_tie_breaks_by_control_index_and_rejects_out_of_tolerance() -> None:
    rows = [
        _row("1.jpg", "bilateral", 1, 0.15),
        _row("1.jpg", "jpeg", 2, 0.10),
        _row("1.jpg", "jpeg", 1, 0.20),
        _row("2.jpg", "bilateral", 1, 0.50),
        _row("2.jpg", "jpeg", 1, 0.60),
    ]
    pairs = match_records(rows, tolerance=0.05)
    assert len(pairs) == 1
    assert pairs[0]["control_index"] == 1


def test_invalid_records_are_never_matched() -> None:
    row = _row("1.jpg", "bilateral", 1, 0.1)
    row["invalid_reason"] = "K < 20"
    assert match_records([row, _row("1.jpg", "jpeg", 1, 0.1)]) == []

