from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from parsing.legal_unit import (
    CanonicalBuilder,
    LegalUnit,
    add_child,
    canonical_text_record,
    clean_text,
    make_unit,
    source_sha256,
)


ARTICLE_RE = re.compile(r"^Article\s+(?P<number>\d+[a-z]?)$", re.IGNORECASE)
RECITAL_RE = re.compile(r"^\((?P<number>\d+)\)$")
PARAGRAPH_RE = re.compile(r"^(?P<number>\d+)\.\s*(?P<body>.+)$")
ITEM_RE = re.compile(r"^\((?P<number>[a-zivx]+)\)\s*(?P<body>.+)$", re.IGNORECASE)


class ParagraphCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag == "p":
            attr_dict = {key: value or "" for key, value in attrs}
            self._active = {
                "classes": attr_dict.get("class", "").split(),
                "parts": [],
            }

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "p" and self._active is not None:
            text = clean_text(" ".join(self._active["parts"]))
            if text:
                self.paragraphs.append(
                    {"classes": self._active["classes"], "text": text}
                )
            self._active = None

    def handle_data(self, data: str) -> None:
        if self._skip_depth or self._active is None:
            return
        self._active["parts"].append(data)


def collect_paragraphs(path: Path) -> list[dict[str, Any]]:
    parser = ParagraphCollector()
    parser.feed(path.read_text(encoding="utf-8", errors="ignore"))
    return parser.paragraphs


def parse_warrant_document(path: Path) -> tuple[list[LegalUnit], dict[str, Any]]:
    source_content = path.read_bytes()
    source_hash = source_sha256(source_content)
    paragraphs = collect_paragraphs(path)
    builder = CanonicalBuilder()
    units: list[LegalUnit] = []
    units_by_id: dict[str, LegalUnit] = {}
    legal_order = 0
    document_code = "W"

    def register(unit: LegalUnit) -> None:
        units.append(unit)
        units_by_id[unit.unit_id] = unit
        add_child(units_by_id, unit.parent_unit_id, unit.unit_id)

    root_id = "W Root"
    root_unit = make_unit(
        unit_id=root_id,
        document_code=document_code,
        unit_type="document",
        number=document_code,
        title="European Arrest Warrant Framework Decision",
        text="",
        parent_unit_id=None,
        intervals=[[0, 0]],
        source_hash=source_hash,
        legal_order=legal_order,
        source_path=path.name,
    )
    register(root_unit)
    legal_order += 1

    first_article_index = next(
        index
        for index, paragraph in enumerate(paragraphs)
        if "title-article-norm" in paragraph["classes"]
        and ARTICLE_RE.match(paragraph["text"])
    )

    legal_order = parse_recitals(
        paragraphs[:first_article_index],
        builder,
        units,
        units_by_id,
        source_hash,
        path.name,
        legal_order,
        root_id,
    )
    legal_order = parse_articles(
        paragraphs[first_article_index:],
        builder,
        units,
        units_by_id,
        source_hash,
        path.name,
        legal_order,
        root_id,
    )

    root_unit.canonical_intervals = [[0, builder.length]]
    root_unit.text = builder.text
    return units, canonical_text_record(document_code, builder.text, source_hash, units)


def parse_recitals(
    paragraphs: list[dict[str, Any]],
    builder: CanonicalBuilder,
    units: list[LegalUnit],
    units_by_id: dict[str, LegalUnit],
    source_hash: str,
    source_path: str,
    legal_order: int,
    root_id: str,
) -> int:
    index = 0
    while index < len(paragraphs) - 1:
        current = paragraphs[index]
        next_paragraph = paragraphs[index + 1]
        match = RECITAL_RE.match(current["text"])
        if (
            match
            and "norm" in current["classes"]
            and "norm" in next_paragraph["classes"]
        ):
            number = match.group("number")
            text = clean_text(f"Recital {number}. {next_paragraph['text']}")
            interval = builder.append_block(text)
            unit_id = f"W Rec. {number}"
            unit = make_unit(
                unit_id=unit_id,
                document_code="W",
                unit_type="recital",
                number=number,
                title=None,
                text=text,
                parent_unit_id=root_id,
                intervals=[interval],
                source_hash=source_hash,
                legal_order=legal_order,
                source_path=source_path,
            )
            units.append(unit)
            units_by_id[unit_id] = unit
            add_child(units_by_id, root_id, unit_id)
            legal_order += 1
            index += 2
        else:
            index += 1
    return legal_order


def parse_articles(
    paragraphs: list[dict[str, Any]],
    builder: CanonicalBuilder,
    units: list[LegalUnit],
    units_by_id: dict[str, LegalUnit],
    source_hash: str,
    source_path: str,
    legal_order: int,
    root_id: str,
) -> int:
    index = 0
    while index < len(paragraphs):
        paragraph = paragraphs[index]
        article_match = ARTICLE_RE.match(paragraph["text"])
        if "title-article-norm" not in paragraph["classes"] or not article_match:
            index += 1
            continue

        article_number = article_match.group("number")
        article_id = f"W Art. {article_number}"
        title_parts: list[str] = []
        body_parts: list[str] = []
        index += 1
        while index < len(paragraphs):
            candidate = paragraphs[index]
            if "title-article-norm" in candidate["classes"] and ARTICLE_RE.match(
                candidate["text"]
            ):
                break
            if "title-article-norm" in candidate["classes"]:
                title_parts.append(candidate["text"])
            elif "norm" in candidate["classes"]:
                body_parts.append(candidate["text"])
            index += 1

        article_title = clean_text(" ".join(title_parts))
        header_interval = builder.append_block(clean_text(f"Article {article_number}. {article_title}"))
        article_start = header_interval[0]
        body_interval_start = builder.length
        if body_parts:
            builder.append_block("\n".join(body_parts))
        article_end = builder.length
        article_text = builder.text[article_start:article_end]

        article_unit = make_unit(
            unit_id=article_id,
            document_code="W",
            unit_type="article",
            number=article_number,
            title=article_title,
            text=article_text,
            parent_unit_id=root_id,
            intervals=[[article_start, article_end]],
            source_hash=source_hash,
            legal_order=legal_order,
            source_path=source_path,
        )
        units.append(article_unit)
        units_by_id[article_id] = article_unit
        add_child(units_by_id, root_id, article_id)
        legal_order += 1

        search_start = 0
        body_text = builder.text[body_interval_start:article_end]
        fallback_paragraph = 1
        current_parent = article_id
        for body_part in body_parts:
            relative_start = body_text.find(body_part, search_start)
            if relative_start == -1:
                continue
            search_start = relative_start + len(body_part)
            interval = [
                body_interval_start + relative_start,
                body_interval_start + relative_start + len(body_part),
            ]
            paragraph_match = PARAGRAPH_RE.match(body_part)
            item_match = ITEM_RE.match(body_part)
            if paragraph_match:
                number = paragraph_match.group("number")
                unit_id = f"{article_id}.{number}"
                current_parent = unit_id
                unit_type = "paragraph"
                parent_id = article_id
                unit_number = f"{article_number}.{number}"
            elif item_match:
                number = item_match.group("number")
                unit_id = f"{current_parent}.{number}"
                unit_type = "subparagraph"
                parent_id = current_parent
                unit_number = f"{article_number}.{number}"
            else:
                number = str(fallback_paragraph)
                fallback_paragraph += 1
                unit_id = f"{article_id}.{number}"
                current_parent = unit_id
                unit_type = "paragraph"
                parent_id = article_id
                unit_number = f"{article_number}.{number}"

            if unit_id in units_by_id:
                continue
            unit = make_unit(
                unit_id=unit_id,
                document_code="W",
                unit_type=unit_type,
                number=unit_number,
                title=None,
                text=body_part,
                parent_unit_id=parent_id,
                intervals=[interval],
                source_hash=source_hash,
                legal_order=legal_order,
                source_path=source_path,
            )
            units.append(unit)
            units_by_id[unit_id] = unit
            add_child(units_by_id, parent_id, unit_id)
            legal_order += 1
    return legal_order
