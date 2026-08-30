from __future__ import annotations

from typing import Any

from chunking.answer_units import build_unit_spans, chunk_to_answer_units
from chunking.chunk import Chunk, chunk_id_for_unit
from chunking.tokenization import DEFAULT_ENCODING, count_tokens


LEAF_LEVELS = {
    "article_or_recital",
    "paragraph_or_recital",
    "subparagraph_or_recital",
}


def build_hierarchical_chunks(
    *,
    config_id: str,
    leaf_level: str,
    legal_units: list[dict[str, Any]],
    embedding_context: str | None = None,
    encoding_name: str = DEFAULT_ENCODING,
) -> list[dict[str, Any]]:
    if leaf_level not in LEAF_LEVELS:
        raise ValueError(f"Unsupported hierarchical leaf level: {leaf_level}")
    if embedding_context not in {None, "structural_breadcrumb"}:
        raise ValueError(f"Unsupported hierarchical embedding context: {embedding_context}")

    units_by_id = {unit["unit_id"]: unit for unit in legal_units}
    spans_by_document = build_unit_spans(legal_units)
    leaf_units = select_leaf_units(legal_units, units_by_id, leaf_level)
    chunks: list[Chunk] = []

    for unit in leaf_units:
        document_code = unit["document_code"]
        answer_unit_ids = chunk_to_answer_units(
            document_code=document_code,
            canonical_intervals=unit["canonical_intervals"],
            spans_by_document=spans_by_document,
        )
        if not answer_unit_ids:
            raise ValueError(f"{config_id} chunk {unit['unit_id']} has no answer units")

        metadata = {
            "document_code": document_code,
            "source_unit_ids": [unit["unit_id"]],
            "answer_unit_ids": answer_unit_ids,
            "parent_chunk_id": parent_chunk_id(unit, units_by_id, leaf_level),
            "child_chunk_ids": child_chunk_ids(unit, units_by_id, leaf_level),
            "hierarchy_level": unit["unit_type"],
            "legal_order": unit["metadata"]["legal_order"],
        }
        if embedding_context is not None:
            metadata["embedding_context"] = embedding_context

        chunks.append(
            Chunk(
                chunk_id=chunk_id_for_unit(config_id, unit["unit_id"]),
                strategy="hierarchical",
                config_id=config_id,
                text=unit["text"],
                token_count=count_tokens(unit["text"], encoding_name),
                canonical_intervals=unit["canonical_intervals"],
                source_text_sha256=unit["source_text_sha256"],
                metadata=metadata,
                embedding_text=structural_embedding_text(unit, units_by_id)
                if embedding_context == "structural_breadcrumb"
                else None,
            )
        )

    chunks.sort(
        key=lambda chunk: (
            chunk.metadata["document_code"],
            chunk.metadata["legal_order"],
            chunk.chunk_id,
        )
    )
    return [chunk.to_dict() for chunk in chunks]


def structural_embedding_text(
    unit: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
) -> str:
    breadcrumb = " > ".join(
        unit_label(ancestor)
        for ancestor in ancestor_chain(unit, units_by_id)
        if ancestor["unit_type"] != "document"
    )
    if not breadcrumb:
        breadcrumb = unit_label(unit)
    return f"{breadcrumb}\n\n{unit['text']}"


def ancestor_chain(
    unit: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    chain = [unit]
    parent_id = unit["parent_unit_id"]
    while parent_id is not None and parent_id in units_by_id:
        parent = units_by_id[parent_id]
        chain.append(parent)
        parent_id = parent["parent_unit_id"]
    return list(reversed(chain))


def unit_label(unit: dict[str, Any]) -> str:
    title = unit.get("title", "").strip()
    return f"{unit['unit_id']}: {title}" if title else unit["unit_id"]


def select_leaf_units(
    legal_units: list[dict[str, Any]],
    units_by_id: dict[str, dict[str, Any]],
    leaf_level: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for unit in legal_units:
        unit_type = unit["unit_type"]
        if unit_type == "recital":
            selected.append(unit)
        elif leaf_level == "article_or_recital" and unit_type == "article":
            selected.append(unit)
        elif leaf_level == "paragraph_or_recital" and is_paragraph_leaf(unit, units_by_id):
            selected.append(unit)
        elif leaf_level == "subparagraph_or_recital" and is_subparagraph_leaf(
            unit,
            units_by_id,
        ):
            selected.append(unit)

    selected.sort(
        key=lambda unit: (unit["document_code"], unit["metadata"]["legal_order"])
    )
    return selected


def is_paragraph_leaf(unit: dict[str, Any], units_by_id: dict[str, dict[str, Any]]) -> bool:
    if unit["unit_type"] == "paragraph":
        return True
    if unit["unit_type"] != "article":
        return False
    return not has_child_type(unit, units_by_id, {"paragraph"})


def is_subparagraph_leaf(unit: dict[str, Any], units_by_id: dict[str, dict[str, Any]]) -> bool:
    if unit["unit_type"] == "subparagraph":
        return True
    if unit["unit_type"] == "paragraph":
        return not has_child_type(unit, units_by_id, {"subparagraph"})
    if unit["unit_type"] == "article":
        return not has_child_type(unit, units_by_id, {"paragraph", "subparagraph"})
    return False


def has_child_type(
    unit: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    child_types: set[str],
) -> bool:
    return any(
        units_by_id[child_id]["unit_type"] in child_types
        for child_id in unit["child_unit_ids"]
        if child_id in units_by_id
    )


def parent_chunk_id(
    unit: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    leaf_level: str,
) -> str | None:
    parent_id = unit["parent_unit_id"]
    if parent_id is None or parent_id not in units_by_id:
        return None
    parent = units_by_id[parent_id]
    if unit["unit_type"] == "paragraph" and parent["unit_type"] == "article":
        return chunk_id_for_unit("hier_article", parent_id)
    if unit["unit_type"] == "subparagraph":
        if parent["unit_type"] == "paragraph":
            return chunk_id_for_unit("hier_paragraph", parent_id)
        if parent["unit_type"] == "article":
            return chunk_id_for_unit("hier_article", parent_id)
    if leaf_level == "subparagraph_or_recital" and unit["unit_type"] == "paragraph":
        if parent["unit_type"] == "article":
            return chunk_id_for_unit("hier_article", parent_id)
    return None


def child_chunk_ids(
    unit: dict[str, Any],
    units_by_id: dict[str, dict[str, Any]],
    leaf_level: str,
) -> list[str]:
    if leaf_level == "article_or_recital" and unit["unit_type"] == "article":
        return [
            chunk_id_for_unit("hier_paragraph", child_id)
            for child_id in unit["child_unit_ids"]
            if child_id in units_by_id
            and units_by_id[child_id]["unit_type"] == "paragraph"
        ]
    if leaf_level == "paragraph_or_recital" and unit["unit_type"] == "paragraph":
        return [
            chunk_id_for_unit("hier_subparagraph", child_id)
            for child_id in unit["child_unit_ids"]
            if child_id in units_by_id
            and units_by_id[child_id]["unit_type"] == "subparagraph"
        ]
    return []
