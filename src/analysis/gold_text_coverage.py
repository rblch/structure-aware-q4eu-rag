"""Post-hoc character-level gold-text coverage.

The unit-level coverage diagnostic counts a gold unit as covered when the
assembled context overlaps any part of its text. This module measures how
much of each gold unit's text the context actually contains, by taking the
union of the canonical character ranges of the source units present in the
context and clipping that union to the gold unit's own range.
"""

from __future__ import annotations

from typing import Any

from scipy import stats

COVERAGE_BANDS = [
    ("zero", lambda value: value == 0.0),
    ("partial", lambda value: 0.0 < value <= 0.5),
    ("high", lambda value: 0.5 < value < 1.0),
    ("complete", lambda value: value == 1.0),
]


def _spans(legal_units: list[dict[str, Any]]) -> dict[str, tuple[str, int, int]]:
    return {
        unit["unit_id"]: (
            unit["document_code"],
            int(unit["canonical_char_start"]),
            int(unit["canonical_char_end"]),
        )
        for unit in legal_units
        if unit.get("canonical_char_start") is not None
    }


def _merged_length(segments: list[tuple[int, int]]) -> int:
    total = 0
    current: tuple[int, int] | None = None
    for start, end in sorted(segments):
        if current and start <= current[1]:
            current = (current[0], max(current[1], end))
            continue
        if current:
            total += current[1] - current[0]
        current = (start, end)
    if current:
        total += current[1] - current[0]
    return total


def gold_unit_coverage(
    gold_unit_id: str,
    included_chunks: list[dict[str, Any]],
    spans: dict[str, tuple[str, int, int]],
) -> float | None:
    gold = spans.get(gold_unit_id)
    if gold is None or gold[2] <= gold[1]:
        return None
    document, gold_start, gold_end = gold
    segments = []
    for chunk in included_chunks:
        for source_unit_id in chunk.get("source_unit_ids", []):
            span = spans.get(source_unit_id)
            if span is None or span[0] != document:
                continue
            start, end = max(gold_start, span[1]), min(gold_end, span[2])
            if end > start:
                segments.append((start, end))
    return _merged_length(segments) / (gold_end - gold_start)


def query_coverage(
    context_records: list[dict[str, Any]], legal_units: list[dict[str, Any]]
) -> dict[tuple[str, str], float]:
    spans = _spans(legal_units)
    coverage: dict[tuple[str, str], float] = {}
    for record in context_records:
        values = [
            value
            for value in (
                gold_unit_coverage(unit_id, record["included_chunks"], spans)
                for unit_id in record["gold_unit_ids"]
            )
            if value is not None
        ]
        if values:
            coverage[(record["query_id"], record["condition_id"])] = sum(values) / len(
                values
            )
    return coverage


def gold_text_coverage_table(
    context_records: list[dict[str, Any]],
    legal_units: list[dict[str, Any]],
    answer_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    coverage = query_coverage(context_records, legal_units)
    correctness = {
        (row["query_id"], row["condition_id"]): float(row["correctness_score"])
        for row in answer_evaluations
    }
    rows = []
    for condition_id in dict.fromkeys(r["condition_id"] for r in context_records):
        keys = [key for key in coverage if key[1] == condition_id]
        if not keys:
            continue
        rows.append(
            {
                "condition_id": condition_id,
                "query_count": len(keys),
                "mean_gold_text_coverage": sum(coverage[k] for k in keys) / len(keys),
                "mean_correctness": sum(correctness[k] for k in keys) / len(keys),
            }
        )
    return rows


def gold_text_coverage_band_table(
    context_records: list[dict[str, Any]],
    legal_units: list[dict[str, Any]],
    answer_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    coverage = query_coverage(context_records, legal_units)
    evaluations = {
        (row["query_id"], row["condition_id"]): row for row in answer_evaluations
    }
    observations = [
        (value, evaluations[key])
        for key, value in coverage.items()
        if key in evaluations
    ]
    rows = []
    for band, predicate in COVERAGE_BANDS:
        members = [row for value, row in observations if predicate(value)]
        if not members:
            continue
        rows.append(
            {
                "coverage_band": band,
                "n": len(members),
                "mean_correctness": sum(float(r["correctness_score"]) for r in members)
                / len(members),
                "abstention_rate": sum(bool(r["answer_abstains"]) for r in members)
                / len(members),
                "mean_faithfulness": sum(
                    float(r["faithfulness_score"]) for r in members
                )
                / len(members),
            }
        )
    values = [value for value, _ in observations]
    scores = [float(row["correctness_score"]) for _, row in observations]
    non_zero = [(v, s) for v, s in zip(values, scores) if v > 0]
    rows.append(
        {
            "coverage_band": "spearman_all",
            "n": len(values),
            "mean_correctness": stats.spearmanr(values, scores).statistic,
            "abstention_rate": "",
            "mean_faithfulness": "",
        }
    )
    rows.append(
        {
            "coverage_band": "spearman_nonzero",
            "n": len(non_zero),
            "mean_correctness": stats.spearmanr(
                [v for v, _ in non_zero], [s for _, s in non_zero]
            ).statistic,
            "abstention_rate": "",
            "mean_faithfulness": "",
        }
    )
    return rows
