from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsing.legal_unit import (
    CanonicalBuilder,
    LegalUnit,
    add_child,
    canonical_text_record,
    clean_text,
    make_unit,
    source_sha256,
)


DOCUMENT_FILES = {
    "B": "bruss.akn",
    "RI": "rome_i.akn",
    "RII": "rome_ii.akn",
    "G": "gdpr.akn",
    "E": "eidas.akn",
}


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def direct_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == name:
            return child
    return None


def children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in element.iter() if local_name(child.tag) == name]


def element_text(element: ET.Element | None) -> str:
    return clean_text(" ".join(element.itertext()) if element is not None else "")


def element_text_without_direct(element: ET.Element, excluded_names: set[str]) -> str:
    parts: list[str] = []
    if element.text:
        parts.append(element.text)
    for child in element:
        if local_name(child.tag) not in excluded_names:
            parts.extend(child.itertext())
        if child.tail:
            parts.append(child.tail)
    return clean_text(" ".join(parts))


def content_text(element: ET.Element) -> str:
    content = direct_child(element, "content")
    if content is not None:
        return element_text(content)
    return element_text_without_direct(element, {"num", "heading"})


def number_text(element: ET.Element) -> str:
    raw = element_text(direct_child(element, "num"))
    raw = raw.replace("Article", "").replace("CHAPTER", "").strip()
    return raw.strip("() .")


def unit_text(prefix: str, *parts: str | None) -> str:
    return clean_text(" ".join(part for part in (prefix, *parts) if part))


def find_relative_interval(haystack: str, needle: str, start: int = 0) -> list[int] | None:
    cleaned_needle = clean_text(needle)
    if not cleaned_needle:
        return None
    index = haystack.find(cleaned_needle, start)
    if index == -1:
        return None
    return [index, index + len(cleaned_needle)]


def parse_akn_document(document_code: str, path: Path) -> tuple[list[LegalUnit], dict[str, Any]]:
    source_content = path.read_bytes()
    source_hash = source_sha256(source_content)
    tree = ET.parse(path)
    root = tree.getroot()
    builder = CanonicalBuilder()
    units: list[LegalUnit] = []
    units_by_id: dict[str, LegalUnit] = {}
    legal_order = 0

    def register(unit: LegalUnit) -> None:
        units.append(unit)
        units_by_id[unit.unit_id] = unit
        add_child(units_by_id, unit.parent_unit_id, unit.unit_id)

    root_id = f"{document_code} Root"
    root_unit = make_unit(
        unit_id=root_id,
        document_code=document_code,
        unit_type="document",
        number=document_code,
        title=None,
        text="",
        parent_unit_id=None,
        intervals=[[0, 0]],
        source_hash=source_hash,
        legal_order=legal_order,
        source_path=path.name,
    )
    register(root_unit)
    legal_order += 1

    for recital in children(root, "recital"):
        number = number_text(recital)
        unit_id = f"{document_code} Rec. {number}"
        text = unit_text(f"Recital {number}.", content_text(recital))
        interval = builder.append_block(text)
        register(
            make_unit(
                unit_id=unit_id,
                document_code=document_code,
                unit_type="recital",
                number=number,
                title=None,
                text=text,
                parent_unit_id=root_id,
                intervals=[interval],
                source_hash=source_hash,
                legal_order=legal_order,
                source_path=path.name,
            )
        )
        legal_order += 1

    for chapter in children(root, "chapter"):
        chapter_number = number_text(chapter)
        chapter_title = element_text(direct_child(chapter, "heading"))
        if not chapter_number:
            # Some acts put the chapter number in <heading>, not <num>.
            match = re.match(r"CHAPTER\s+(\S+)", chapter_title, re.IGNORECASE)
            if not match:
                raise ValueError(
                    f"{document_code} chapter has no parseable number "
                    f"(heading={chapter_title!r})"
                )
            chapter_number = match.group(1).strip(" .")
            chapter_title = None
        chapter_id = f"{document_code} Chap. {chapter_number}"
        chapter_start = builder.length
        chapter_unit = make_unit(
            unit_id=chapter_id,
            document_code=document_code,
            unit_type="chapter",
            number=chapter_number,
            title=chapter_title,
            text="",
            parent_unit_id=root_id,
            intervals=[[chapter_start, chapter_start]],
            source_hash=source_hash,
            legal_order=legal_order,
            source_path=path.name,
        )
        register(chapter_unit)
        legal_order += 1

        for article in [node for node in chapter.iter() if local_name(node.tag) == "article"]:
            legal_order = parse_article(
                document_code=document_code,
                article=article,
                parent_id=chapter_id,
                builder=builder,
                units=units,
                units_by_id=units_by_id,
                source_hash=source_hash,
                source_path=path.name,
                legal_order=legal_order,
            )

        chapter_end = builder.length
        chapter_unit.canonical_intervals = [[chapter_start, chapter_end]]
        chapter_unit.text = builder.text[chapter_start:chapter_end]

    root_unit.canonical_intervals = [[0, builder.length]]
    root_unit.text = builder.text
    return units, canonical_text_record(document_code, builder.text, source_hash, units)


def parse_article(
    *,
    document_code: str,
    article: ET.Element,
    parent_id: str,
    builder: CanonicalBuilder,
    units: list[LegalUnit],
    units_by_id: dict[str, LegalUnit],
    source_hash: str,
    source_path: str,
    legal_order: int,
) -> int:
    article_number = number_text(article)
    article_id = f"{document_code} Art. {article_number}"
    article_title = element_text(direct_child(article, "heading"))
    header_interval = builder.append_block(unit_text(f"Article {article_number}.", article_title))
    article_start = header_interval[0]

    article_unit = make_unit(
        unit_id=article_id,
        document_code=document_code,
        unit_type="article",
        number=article_number,
        title=article_title,
        text="",
        parent_unit_id=parent_id,
        intervals=[header_interval],
        source_hash=source_hash,
        legal_order=legal_order,
        source_path=source_path,
    )
    units.append(article_unit)
    units_by_id[article_id] = article_unit
    add_child(units_by_id, parent_id, article_id)
    legal_order += 1

    paragraph_elements = [child for child in article if local_name(child.tag) == "paragraph"]
    if not paragraph_elements:
        paragraph_elements = [article]

    for index, paragraph in enumerate(paragraph_elements, start=1):
        paragraph_number = number_text(paragraph) or str(index)
        paragraph_id = f"{article_id}.{paragraph_number}"
        paragraph_text = unit_text(paragraph_number + ".", content_text(paragraph))
        paragraph_interval = builder.append_block(paragraph_text)
        paragraph_unit = make_unit(
            unit_id=paragraph_id,
            document_code=document_code,
            unit_type="paragraph",
            number=f"{article_number}.{paragraph_number}",
            title=None,
            text=paragraph_text,
            parent_unit_id=article_id,
            intervals=[paragraph_interval],
            source_hash=source_hash,
            legal_order=legal_order,
            source_path=source_path,
        )
        units.append(paragraph_unit)
        units_by_id[paragraph_id] = paragraph_unit
        add_child(units_by_id, article_id, paragraph_id)
        legal_order += 1

        search_start = 0
        for item in [node for node in paragraph.iter() if local_name(node.tag) == "item"]:
            item_num = number_text(item)
            if not item_num:
                continue
            item_body = content_text(item)
            item_text = unit_text(item_num, item_body)
            relative = find_relative_interval(paragraph_text, item_body, search_start)
            if relative is None:
                continue
            search_start = relative[1]
            item_interval = [
                paragraph_interval[0] + relative[0],
                paragraph_interval[0] + relative[1],
            ]
            item_id = f"{paragraph_id}.{item_num.strip('()')}"
            item_unit = make_unit(
                unit_id=item_id,
                document_code=document_code,
                unit_type="subparagraph",
                number=f"{article_number}.{paragraph_number}.{item_num.strip('()')}",
                title=None,
                text=item_text,
                parent_unit_id=paragraph_id,
                intervals=[item_interval],
                source_hash=source_hash,
                legal_order=legal_order,
                source_path=source_path,
            )
            units.append(item_unit)
            units_by_id[item_id] = item_unit
            add_child(units_by_id, paragraph_id, item_id)
            legal_order += 1

    article_end = builder.length
    article_unit.canonical_intervals = [[article_start, article_end]]
    article_unit.text = builder.text[article_start:article_end]
    return legal_order


def parse_all_akn_documents(raw_documents_dir: Path) -> tuple[list[LegalUnit], dict[str, Any]]:
    all_units: list[LegalUnit] = []
    canonical_texts: dict[str, Any] = {}
    for document_code, filename in DOCUMENT_FILES.items():
        units, canonical = parse_akn_document(document_code, raw_documents_dir / filename)
        all_units.extend(units)
        canonical_texts[document_code] = canonical
    return all_units, canonical_texts
