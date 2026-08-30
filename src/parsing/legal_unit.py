from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class LegalUnit:
    unit_id: str
    document_code: str
    unit_type: str
    number: str
    title: str | None
    text: str
    parent_unit_id: str | None
    child_unit_ids: list[str] = field(default_factory=list)
    canonical_intervals: list[list[int]] = field(default_factory=list)
    source_text_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        start = min(interval[0] for interval in self.canonical_intervals)
        end = max(interval[1] for interval in self.canonical_intervals)
        return {
            "unit_id": self.unit_id,
            "document_code": self.document_code,
            "unit_type": self.unit_type,
            "number": self.number,
            "title": self.title,
            "text": self.text,
            "parent_unit_id": self.parent_unit_id,
            "child_unit_ids": self.child_unit_ids,
            "canonical_char_start": start,
            "canonical_char_end": end,
            "canonical_intervals": self.canonical_intervals,
            "source_text_sha256": self.source_text_sha256,
            "metadata": self.metadata,
        }


class CanonicalBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self._length = 0

    @property
    def text(self) -> str:
        return "".join(self._parts).rstrip()

    @property
    def length(self) -> int:
        return len(self.text)

    def append_block(self, text: str) -> list[int]:
        cleaned = clean_text(text)
        if not cleaned:
            return [self._length, self._length]
        if self._parts and not self._parts[-1].endswith("\n\n"):
            self._parts.append("\n\n")
            self._length += 2
        start = self._length
        self._parts.append(cleaned)
        self._length += len(cleaned)
        return [start, self._length]


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return WHITESPACE_RE.sub(" ", text.replace("\xa0", " ")).strip()


def source_sha256(content: str | bytes) -> str:
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def canonical_text_record(
    document_code: str,
    text: str,
    source_text_sha256: str,
    units: list[LegalUnit],
) -> dict[str, Any]:
    return {
        "document_code": document_code,
        "text": text,
        "char_length": len(text),
        "source_text_sha256": source_text_sha256,
        "unit_offsets": {
            unit.unit_id: [
                min(interval[0] for interval in unit.canonical_intervals),
                max(interval[1] for interval in unit.canonical_intervals),
            ]
            for unit in units
        },
    }


def add_child(units_by_id: dict[str, LegalUnit], parent_id: str | None, child_id: str) -> None:
    if parent_id and parent_id in units_by_id:
        units_by_id[parent_id].child_unit_ids.append(child_id)


def make_unit(
    *,
    unit_id: str,
    document_code: str,
    unit_type: str,
    number: str,
    title: str | None,
    text: str,
    parent_unit_id: str | None,
    intervals: list[list[int]],
    source_hash: str,
    legal_order: int,
    source_path: str,
) -> LegalUnit:
    return LegalUnit(
        unit_id=unit_id,
        document_code=document_code,
        unit_type=unit_type,
        number=number,
        title=clean_text(title),
        text=clean_text(text),
        parent_unit_id=parent_unit_id,
        canonical_intervals=intervals,
        source_text_sha256=source_hash,
        metadata={"source_path": source_path, "legal_order": legal_order},
    )

