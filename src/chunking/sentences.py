from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import spacy

from chunking.answer_units import (
    build_unit_spans,
    chunk_to_answer_units,
    overlapping_source_unit_ids,
)
from chunking.tokenization import DEFAULT_ENCODING, count_tokens
from data_import.id_normalization import dedupe_preserving_order


BLOCK_RE = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)
LEGAL_MARKER_RE = re.compile(
    r"^(?:\(?[0-9]+[a-z]?\)|\(?[a-z]\)|[0-9]+[.)]|[a-z][.)])$",
    re.IGNORECASE,
)
ARTICLE_LABEL_RE = re.compile(r"^(?:Article|Recital)\s+\d+[A-Za-z]?\.$")
PUNCTUATION_FRAGMENT_RE = re.compile(r"^[^\w]+$", re.UNICODE)
FRAGMENT_MAX_TOKENS = 8


@dataclass(frozen=True)
class SentenceSegment:
    text: str
    canonical_intervals: list[list[int]]
    source_unit_ids: list[str]
    answer_unit_ids: list[str]
    token_count: int

    @property
    def start(self) -> int:
        return self.canonical_intervals[0][0]

    @property
    def end(self) -> int:
        return self.canonical_intervals[-1][1]


@lru_cache(maxsize=1)
def sentence_nlp() -> spacy.language.Language:
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000
    return nlp


def segment_document_sentences(
    *,
    document_code: str,
    canonical_text: str,
    legal_units: list[dict[str, Any]],
    encoding_name: str = DEFAULT_ENCODING,
) -> list[SentenceSegment]:
    spans_by_document = build_unit_spans(legal_units)
    segments: list[SentenceSegment] = []
    for block_match in BLOCK_RE.finditer(canonical_text):
        block = block_match.group(0)
        block_start = block_match.start()
        doc = sentence_nlp()(block)
        for sentence in doc.sents:
            raw_sentence = sentence.text
            text = raw_sentence.strip()
            if not text:
                continue
            leading_whitespace = len(raw_sentence) - len(raw_sentence.lstrip())
            trailing_whitespace = len(raw_sentence) - len(raw_sentence.rstrip())
            start = block_start + sentence.start_char + leading_whitespace
            end = block_start + sentence.end_char - trailing_whitespace
            intervals = [[start, end]]
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
                continue
            segments.append(
                SentenceSegment(
                    text=text,
                    canonical_intervals=intervals,
                    source_unit_ids=source_unit_ids,
                    answer_unit_ids=answer_unit_ids,
                    token_count=count_tokens(text, encoding_name),
                )
            )
    return merge_fragment_segments(segments, canonical_text, encoding_name)


def merge_fragment_segments(
    segments: list[SentenceSegment],
    canonical_text: str,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[SentenceSegment]:
    merged = list(segments)
    changed = True
    while changed:
        changed = False
        next_pass: list[SentenceSegment] = []
        index = 0
        while index < len(merged):
            current = merged[index]
            if (
                index + 1 < len(merged)
                and is_fragment_segment(current)
                and segments_are_related(current, merged[index + 1])
            ):
                next_pass.append(
                    merge_two_segments(
                        current,
                        merged[index + 1],
                        canonical_text,
                        encoding_name,
                    )
                )
                changed = True
                index += 2
            elif (
                is_fragment_segment(current)
                and next_pass
                and segments_are_related(next_pass[-1], current)
            ):
                next_pass[-1] = merge_two_segments(
                    next_pass[-1],
                    current,
                    canonical_text,
                    encoding_name,
                )
                changed = True
                index += 1
            else:
                next_pass.append(current)
                index += 1
        merged = next_pass
    return merged


def is_fragment_segment(segment: SentenceSegment) -> bool:
    text = segment.text.strip()
    if is_fragment_text_line(text):
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1 and all(is_fragment_text_line(line) for line in lines):
        return True
    if segment.token_count <= FRAGMENT_MAX_TOKENS and text[-1:] not in ".;:!?":
        return True
    return False


def is_fragment_text_line(text: str) -> bool:
    if PUNCTUATION_FRAGMENT_RE.fullmatch(text):
        return True
    if LEGAL_MARKER_RE.fullmatch(text) or ARTICLE_LABEL_RE.fullmatch(text):
        return True
    return len(text.split()) <= FRAGMENT_MAX_TOKENS and text[-1:] not in ".;:!?"


def segments_are_related(left: SentenceSegment, right: SentenceSegment) -> bool:
    return bool(
        set(left.answer_unit_ids).intersection(right.answer_unit_ids)
        or set(left.source_unit_ids).intersection(right.source_unit_ids)
    )


def merge_two_segments(
    left: SentenceSegment,
    right: SentenceSegment,
    canonical_text: str,
    encoding_name: str,
) -> SentenceSegment:
    start = left.start
    end = right.end
    text = canonical_text[start:end]
    source_unit_ids = dedupe_preserving_order(left.source_unit_ids + right.source_unit_ids)
    answer_unit_ids = dedupe_preserving_order(left.answer_unit_ids + right.answer_unit_ids)
    return SentenceSegment(
        text=text,
        canonical_intervals=[[start, end]],
        source_unit_ids=source_unit_ids,
        answer_unit_ids=answer_unit_ids,
        token_count=count_tokens(text, encoding_name),
    )
