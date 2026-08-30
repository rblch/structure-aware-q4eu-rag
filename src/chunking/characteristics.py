from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any


def build_chunk_characteristics(
    chunks_by_config: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "configs": {
            config_id: summarize_chunks(chunks)
            for config_id, chunks in sorted(chunks_by_config.items())
        }
    }


def summarize_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = [chunk["token_count"] for chunk in chunks]
    per_document = Counter(chunk["metadata"]["document_code"] for chunk in chunks)
    chunks_without_answer_units = [
        chunk["chunk_id"]
        for chunk in chunks
        if not chunk["metadata"].get("answer_unit_ids")
    ]

    return {
        "chunk_count": len(chunks),
        "chunk_count_by_document": dict(sorted(per_document.items())),
        "token_count": {
            "min": min(token_counts) if token_counts else 0,
            "median": median(token_counts) if token_counts else 0,
            "p95": percentile(token_counts, 0.95) if token_counts else 0,
            "max": max(token_counts) if token_counts else 0,
        },
        "chunks_without_answer_units": chunks_without_answer_units,
    }


def percentile(values: list[int], quantile: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
