from __future__ import annotations

from typing import Any

from chunking.answer_units import (
    build_unit_spans,
    chunk_to_answer_units,
    overlapping_source_unit_ids,
)
from chunking.chunk import Chunk
from chunking.tokenization import DEFAULT_ENCODING, token_offsets


def build_fixed_size_chunks(
    *,
    config_id: str,
    chunk_size: int,
    chunk_overlap: int,
    canonical_texts: dict[str, dict[str, Any]],
    legal_units: list[dict[str, Any]],
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    spans_by_document = build_unit_spans(legal_units)
    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap

    for document_code in sorted(canonical_texts):
        canonical = canonical_texts[document_code]
        offsets, text, token_count = token_offsets(canonical["text"], encoding_name)
        chunk_number = 1
        for token_start in range(0, token_count, step):
            token_end = min(token_start + chunk_size, token_count)
            char_start = offsets[token_start]
            char_end = offsets[token_end] if token_end < token_count else len(text)
            if char_start == char_end:
                continue

            intervals = [[char_start, char_end]]
            source_unit_ids = overlapping_source_unit_ids(
                document_code=document_code,
                canonical_intervals=intervals,
                spans_by_document=spans_by_document,
            )
            answer_unit_ids = chunk_to_answer_units(
                document_code=document_code,
                canonical_intervals=intervals,
                spans_by_document=spans_by_document,
            )
            if not answer_unit_ids:
                # Every window must map to provision text.
                snippet = text[char_start : min(char_end, char_start + 120)]
                raise ValueError(
                    f"{config_id} chunk {document_code}_{chunk_number:06d} "
                    f"has no answer units (chars {char_start}-{char_end}: "
                    f"{snippet!r})"
                )

            chunks.append(
                Chunk(
                    chunk_id=f"{config_id}_{document_code}_{chunk_number:06d}",
                    strategy="fixed_size",
                    config_id=config_id,
                    text=text[char_start:char_end],
                    token_count=token_end - token_start,
                    canonical_intervals=intervals,
                    source_text_sha256=canonical["source_text_sha256"],
                    metadata={
                        "document_code": document_code,
                        "source_unit_ids": source_unit_ids,
                        "answer_unit_ids": answer_unit_ids,
                        "token_start": token_start,
                        "token_end": token_end,
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "boundary_mode": "raw_token_windows",
                    },
                )
            )
            chunk_number += 1
            if token_end == token_count:
                break

    return [chunk.to_dict() for chunk in chunks]
