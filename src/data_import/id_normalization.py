from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


DOCUMENT_CODES = ("B", "RI", "RII", "G", "E", "W")

REFERENCE_RE = re.compile(
    r"^\s*(?P<document>B|RI|RII|G|E|W)\s+"
    r"(?P<kind>Art\.?|Article|Rec\.?|Recital)\s+"
    r"(?P<number>\d+[a-zA-Z]?(?:\.(?:\d+[a-zA-Z]?|[a-zA-Z]+))*)(?:\s*)$"
)


@dataclass(frozen=True)
class ReferenceLabel:
    document_code: str
    unit_type: str
    number: str

    @property
    def unit_id(self) -> str:
        return f"{self.document_code} {self.unit_type}. {self.number}"


def parse_reference_label(label: str) -> ReferenceLabel:
    match = REFERENCE_RE.match(label)
    if not match:
        raise ValueError(f"Unsupported legal reference label: {label!r}")

    raw_kind = match.group("kind").lower().rstrip(".")
    unit_type = "Art" if raw_kind in {"art", "article"} else "Rec"
    return ReferenceLabel(
        document_code=match.group("document"),
        unit_type=unit_type,
        number=match.group("number"),
    )


def normalize_to_article_recital(label: str) -> str:
    reference = parse_reference_label(label)
    number = reference.number.split(".", 1)[0] if reference.unit_type == "Art" else reference.number
    return ReferenceLabel(reference.document_code, reference.unit_type, number).unit_id


def normalize_to_paragraph(label: str) -> str:
    return parse_reference_label(label).unit_id


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
