from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np

from chunking.answer_units import (
    build_unit_spans,
    chunk_to_answer_units,
    overlapping_source_unit_ids,
)
from chunking.chunk import Chunk
from chunking.sentences import SentenceSegment, segment_document_sentences
from chunking.tokenization import DEFAULT_ENCODING, count_tokens, get_tokenizer
from data_import.id_normalization import dedupe_preserving_order


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def build_semantic_chunks(
    *,
    config_id: str,
    breakpoint_percentile: float,
    max_chunk_size: int,
    min_chunk_size: int,
    window_size: int,
    canonical_texts: dict[str, dict[str, Any]],
    legal_units: list[dict[str, Any]],
    embedder: TextEmbedder,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    spans_by_document = build_unit_spans(legal_units)
    chunks: list[Chunk] = []

    for document_code in sorted(canonical_texts):
        canonical = canonical_texts[document_code]
        segments = segment_document_sentences(
            document_code=document_code,
            canonical_text=canonical["text"],
            legal_units=legal_units,
            encoding_name=encoding_name,
        )
        segments = split_oversize_segments(
            segments,
            max_chunk_size=max_chunk_size,
            canonical_text=canonical["text"],
            encoding_name=encoding_name,
        )
        if not segments:
            continue

        embeddings = np.asarray(
            embedder.embed_texts([segment.text for segment in segments]),
            dtype=float,
        )
        scores = compute_boundary_scores(embeddings, window_size)
        groups = initial_segment_groups(len(segments), scores, breakpoint_percentile)
        groups = merge_small_groups(groups, segments, scores, min_chunk_size)
        groups = split_large_groups(groups, segments, scores, max_chunk_size)
        groups = merge_small_groups_bounded(
            groups,
            segments,
            scores,
            min_chunk_size,
            max_chunk_size,
        )

        for chunk_number, (start, end) in enumerate(groups, start=1):
            chunk_segments = segments[start:end]
            char_start = chunk_segments[0].start
            char_end = chunk_segments[-1].end
            intervals = [[char_start, char_end]]
            source_unit_ids = dedupe_preserving_order(
                unit_id
                for segment in chunk_segments
                for unit_id in segment.source_unit_ids
            )
            answer_unit_ids = chunk_to_answer_units(
                document_code=document_code,
                canonical_intervals=intervals,
                spans_by_document=spans_by_document,
            )
            if not answer_unit_ids:
                answer_unit_ids = dedupe_preserving_order(
                    unit_id
                    for segment in chunk_segments
                    for unit_id in segment.answer_unit_ids
                )
            if not answer_unit_ids:
                raise ValueError(f"{config_id} semantic chunk has no answer units")

            chunks.append(
                Chunk(
                    chunk_id=f"{config_id}_{document_code}_{chunk_number:06d}",
                    strategy="semantic",
                    config_id=config_id,
                    text=canonical["text"][char_start:char_end],
                    token_count=count_tokens(
                        canonical["text"][char_start:char_end],
                        encoding_name,
                    ),
                    canonical_intervals=intervals,
                    source_text_sha256=canonical["source_text_sha256"],
                    metadata={
                        "document_code": document_code,
                        "source_unit_ids": source_unit_ids
                        or overlapping_source_unit_ids(
                            document_code=document_code,
                            canonical_intervals=intervals,
                            spans_by_document=spans_by_document,
                        ),
                        "answer_unit_ids": answer_unit_ids,
                        "sentence_count": len(chunk_segments),
                        "breakpoint_percentile": breakpoint_percentile,
                        "min_chunk_size": min_chunk_size,
                        "max_chunk_size": max_chunk_size,
                        "window_size": window_size,
                    },
                )
            )

    return [chunk.to_dict() for chunk in chunks]


def compute_boundary_scores(
    embeddings: np.ndarray,
    window_size: int,
) -> dict[int, float]:
    scores: dict[int, float] = {}
    for boundary_index in range(1, len(embeddings)):
        before = embeddings[max(0, boundary_index - window_size) : boundary_index]
        after = embeddings[boundary_index : min(len(embeddings), boundary_index + window_size)]
        scores[boundary_index] = cosine_similarity(before.mean(axis=0), after.mean(axis=0))
    return scores


def initial_segment_groups(
    segment_count: int,
    scores: dict[int, float],
    breakpoint_percentile: float,
) -> list[tuple[int, int]]:
    if segment_count == 0:
        return []
    if not scores:
        return [(0, segment_count)]

    threshold = float(np.percentile(list(scores.values()), breakpoint_percentile))
    breakpoints = sorted(index for index, score in scores.items() if score < threshold)
    groups: list[tuple[int, int]] = []
    start = 0
    for breakpoint in breakpoints:
        groups.append((start, breakpoint))
        start = breakpoint
    groups.append((start, segment_count))
    return groups


def merge_small_groups(
    groups: list[tuple[int, int]],
    segments: list[SentenceSegment],
    scores: dict[int, float],
    min_chunk_size: int,
) -> list[tuple[int, int]]:
    groups = list(groups)
    while len(groups) > 1:
        merge_index = next(
            (
                index
                for index, group in enumerate(groups)
                if group_token_count(group, segments) < min_chunk_size
            ),
            None,
        )
        if merge_index is None:
            return groups
        if merge_index == 0:
            groups[0:2] = [(groups[0][0], groups[1][1])]
        elif merge_index == len(groups) - 1:
            groups[-2:] = [(groups[-2][0], groups[-1][1])]
        else:
            left_similarity = scores.get(groups[merge_index][0], -1.0)
            right_similarity = scores.get(groups[merge_index][1], -1.0)
            if left_similarity >= right_similarity:
                groups[merge_index - 1 : merge_index + 1] = [
                    (groups[merge_index - 1][0], groups[merge_index][1])
                ]
            else:
                groups[merge_index : merge_index + 2] = [
                    (groups[merge_index][0], groups[merge_index + 1][1])
                ]
    return groups


def split_large_groups(
    groups: list[tuple[int, int]],
    segments: list[SentenceSegment],
    scores: dict[int, float],
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    groups = list(groups)
    changed = True
    while changed:
        changed = False
        for index, group in enumerate(groups):
            if group_token_count(group, segments) <= max_chunk_size:
                continue
            start, end = group
            if end - start <= 1:
                continue
            internal_boundaries = range(start + 1, end)
            split_at = min(internal_boundaries, key=lambda boundary: scores.get(boundary, 1.0))
            groups[index : index + 1] = [(start, split_at), (split_at, end)]
            changed = True
            break
    return groups


def merge_small_groups_bounded(
    groups: list[tuple[int, int]],
    segments: list[SentenceSegment],
    scores: dict[int, float],
    min_chunk_size: int,
    max_chunk_size: int,
) -> list[tuple[int, int]]:
    groups = list(groups)
    while len(groups) > 1:
        merge_index = next_bounded_merge_index(
            groups,
            segments,
            min_chunk_size,
            max_chunk_size,
        )
        if merge_index is None:
            return groups
        merge_with = best_allowed_neighbour(
            groups,
            segments,
            scores,
            merge_index,
            max_chunk_size,
        )
        if merge_with is None:
            return groups
        left = min(merge_index, merge_with)
        right = max(merge_index, merge_with)
        groups[left : right + 1] = [(groups[left][0], groups[right][1])]
    return groups


def next_bounded_merge_index(
    groups: list[tuple[int, int]],
    segments: list[SentenceSegment],
    min_chunk_size: int,
    max_chunk_size: int,
) -> int | None:
    for index, group in enumerate(groups):
        if group_token_count(group, segments) >= min_chunk_size:
            continue
        if (
            best_allowed_neighbour(groups, segments, {}, index, max_chunk_size)
            is not None
        ):
            return index
    return None


def best_allowed_neighbour(
    groups: list[tuple[int, int]],
    segments: list[SentenceSegment],
    scores: dict[int, float],
    index: int,
    max_chunk_size: int,
) -> int | None:
    options: list[tuple[float, int]] = []
    if (
        index > 0
        and combined_group_token_count(groups[index - 1], groups[index], segments)
        <= max_chunk_size
    ):
        options.append((scores.get(groups[index][0], -1.0), index - 1))
    if (
        index < len(groups) - 1
        and combined_group_token_count(groups[index], groups[index + 1], segments)
        <= max_chunk_size
    ):
        options.append((scores.get(groups[index][1], -1.0), index + 1))
    if not options:
        return None
    return max(options, key=lambda item: (item[0], -abs(item[1] - index)))[1]


def combined_group_token_count(
    left: tuple[int, int],
    right: tuple[int, int],
    segments: list[SentenceSegment],
) -> int:
    return group_token_count((left[0], right[1]), segments)


def split_oversize_segments(
    segments: list[SentenceSegment],
    *,
    max_chunk_size: int,
    canonical_text: str,
    encoding_name: str,
) -> list[SentenceSegment]:
    split_segments: list[SentenceSegment] = []
    for segment in segments:
        if segment.token_count <= max_chunk_size:
            split_segments.append(segment)
            continue
        split_segments.extend(
            split_segment_by_tokens(
                segment,
                max_chunk_size=max_chunk_size,
                canonical_text=canonical_text,
                encoding_name=encoding_name,
            )
        )
    return split_segments


def split_segment_by_tokens(
    segment: SentenceSegment,
    *,
    max_chunk_size: int,
    canonical_text: str,
    encoding_name: str,
) -> list[SentenceSegment]:
    tokenizer = get_tokenizer(encoding_name)
    token_ids = tokenizer.encode(segment.text)
    _, offsets = tokenizer.decode_with_offsets(token_ids)
    pieces: list[SentenceSegment] = []
    token_start = 0
    while token_start < len(token_ids):
        remaining_tokens = len(token_ids) - token_start
        remaining_pieces = math.ceil(remaining_tokens / max_chunk_size)
        piece_size = math.ceil(remaining_tokens / remaining_pieces)
        token_end = token_start + piece_size
        local_start = offsets[token_start]
        local_end = offsets[token_end] if token_end < len(offsets) else len(segment.text)
        start = segment.start + local_start
        end = segment.start + local_end
        start, end = trim_interval(canonical_text, start, end)
        text = canonical_text[start:end]
        token_count = count_tokens(text, encoding_name)
        while token_count > max_chunk_size and token_end > token_start + 1:
            token_end -= 1
            local_end = offsets[token_end] if token_end < len(offsets) else len(segment.text)
            end = segment.start + local_end
            start, end = trim_interval(canonical_text, start, end)
            text = canonical_text[start:end]
            token_count = count_tokens(text, encoding_name)
        if not text:
            token_start = token_end
            continue
        pieces.append(
            SentenceSegment(
                text=text,
                canonical_intervals=[[start, end]],
                source_unit_ids=segment.source_unit_ids,
                answer_unit_ids=segment.answer_unit_ids,
                token_count=token_count,
            )
        )
        token_start = token_end
    return pieces


def trim_interval(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def group_token_count(group: tuple[int, int], segments: list[SentenceSegment]) -> int:
    start, end = group
    separator_allowance = max(0, end - start - 1)
    return sum(segment.token_count for segment in segments[start:end]) + separator_allowance


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)
