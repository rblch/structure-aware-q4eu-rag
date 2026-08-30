from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from data_import.id_normalization import dedupe_preserving_order, normalize_to_article_recital


ANSWER_MAPPABLE_UNIT_TYPES = {"article", "recital", "paragraph", "subparagraph"}


@dataclass(frozen=True)
class UnitSpan:
    unit_id: str
    document_code: str
    unit_type: str
    intervals: tuple[tuple[int, int], ...]
    legal_order: int


def build_unit_spans(legal_units: list[dict[str, Any]]) -> dict[str, list[UnitSpan]]:
    spans_by_document: dict[str, list[UnitSpan]] = {}
    for unit in legal_units:
        if unit["unit_type"] not in ANSWER_MAPPABLE_UNIT_TYPES:
            continue
        spans_by_document.setdefault(unit["document_code"], []).append(
            UnitSpan(
                unit_id=unit["unit_id"],
                document_code=unit["document_code"],
                unit_type=unit["unit_type"],
                intervals=tuple(tuple(interval) for interval in unit["canonical_intervals"]),
                legal_order=unit["metadata"]["legal_order"],
            )
        )

    for spans in spans_by_document.values():
        spans.sort(key=lambda span: (span.legal_order, span.unit_id))
    return spans_by_document


def chunk_to_answer_units(
    *,
    document_code: str,
    canonical_intervals: list[list[int]],
    spans_by_document: dict[str, list[UnitSpan]],
) -> list[str]:
    return dedupe_preserving_order(
        normalize_to_article_recital(unit_id)
        for unit_id in overlapping_source_unit_ids(
            document_code=document_code,
            canonical_intervals=canonical_intervals,
            spans_by_document=spans_by_document,
        )
    )


def overlapping_source_unit_ids(
    *,
    document_code: str,
    canonical_intervals: list[list[int]],
    spans_by_document: dict[str, list[UnitSpan]],
) -> list[str]:
    overlapping: list[str] = []
    for span in spans_by_document.get(document_code, []):
        if intervals_overlap(canonical_intervals, span.intervals):
            overlapping.append(span.unit_id)
    return overlapping


def intervals_overlap(
    left_intervals: Iterable[Iterable[int]],
    right_intervals: Iterable[Iterable[int]],
) -> bool:
    for left_start, left_end in left_intervals:
        for right_start, right_end in right_intervals:
            if left_start < right_end and right_start < left_end:
                return True
    return False
