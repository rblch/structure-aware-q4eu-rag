from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import.id_normalization import normalize_to_article_recital


ARTICLE_SINGLE_RE = re.compile(
    r"\bArticle\s+(?P<number>\d+[a-z]?)"
    r"(?:\((?P<paragraph>\d+)\))?"
    r"(?:\((?P<point>[a-z])\))?",
    re.IGNORECASE,
)
ARTICLE_RANGE_RE = re.compile(
    r"\bArticles\s+(?P<start>\d+)\s+(?:to|-)\s+(?P<end>\d+)",
    re.IGNORECASE,
)
# Exclude "to" so ARTICLE_RANGE_RE retains ranges.
ARTICLE_LIST_RE = re.compile(
    r"\bArticles\s+(?P<body>\d+[a-z]?(?:\(\d+\))?"
    r"(?:\s*(?:,|\band\b|\bor\b)\s*\d+[a-z]?(?:\(\d+\))?)+)",
    re.IGNORECASE,
)
ARTICLE_LIST_ITEM_RE = re.compile(r"(?P<number>\d+[a-z]?)(?:\((?P<paragraph>\d+)\))?")
RECITAL_RE = re.compile(r"\brecital\s+(?P<number>\d+)", re.IGNORECASE)
RECITAL_LIST_RE = re.compile(
    r"\brecitals\s+(?P<body>\d+(?:\s*(?:,|\band\b|\bor\b)\s*\d+)+)",
    re.IGNORECASE,
)
PARAGRAPH_RE = re.compile(r"\bparagraphs?\s+(?P<number>\d+)", re.IGNORECASE)
EXTERNAL_TERMS_RE = re.compile(
    r"\b(Directive|Treaty|TFEU|TEU|Convention|Charter|Decision\s+\d|"
    r"Regulation\s+\([A-Z]{2,}\))\b",
    re.IGNORECASE,
)
LOCAL_ACT_TERMS_RE = re.compile(
    r"\b(of\s+)?this\s+(Regulation|Framework Decision)\b",
    re.IGNORECASE,
)


def extract_xrefs(
    legal_units: list[dict[str, Any]],
) -> dict[str, Any]:
    units_by_id = {unit["unit_id"]: unit for unit in legal_units}
    article_ids_by_doc: dict[str, set[str]] = {}
    for unit in legal_units:
        if unit["unit_type"] == "article":
            article_ids_by_doc.setdefault(unit["document_code"], set()).add(unit["unit_id"])
    source_units = [
        unit
        for unit in legal_units
        if unit["unit_type"] in {"article", "recital", "paragraph", "subparagraph"}
    ]

    edges: list[dict[str, Any]] = []
    external_references: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for unit in source_units:
        text = unit["text"]
        document_code = unit["document_code"]
        for candidate in iter_reference_candidates(unit, text):
            raw_match = candidate["raw_match"]
            if is_external_reference(text, candidate["start"], candidate["end"]):
                external_references.append(
                    {"source_unit_id": unit["unit_id"], "raw_match": raw_match}
                )
                continue

            for target_unit_id in resolve_candidate(candidate, unit, article_ids_by_doc):
                if target_unit_id not in units_by_id:
                    normalized = normalize_to_article_recital(target_unit_id)
                    if normalized in units_by_id:
                        target_unit_id = normalized
                    else:
                        continue
                normalized_target = normalize_to_article_recital(target_unit_id)
                if normalized_target == normalize_to_article_recital(unit["unit_id"]):
                    continue
                key = (unit["unit_id"], target_unit_id, raw_match)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    {
                        "source_unit_id": unit["unit_id"],
                        "target_unit_id": target_unit_id,
                        "target_unit_id_normalized": normalized_target,
                        "document_code": document_code,
                        "reference_kind": candidate["reference_kind"],
                        "raw_match": raw_match,
                    }
                )

    return {
        "edges": edges,
        "external_references_excluded": external_references,
        "summary": {
            "extractor_backend": "canonical_text_patterns",
            "scope": "intra_document_only",
            "external_reference_policy": (
                "References with nearby external-instrument markers are recorded "
                "in external_references_excluded and omitted from enrichment edges."
            ),
            "edge_count": len(edges),
            "external_reference_count": len(external_references),
            "edge_counts_by_act": dict(
                sorted(Counter(edge["document_code"] for edge in edges).items())
            ),
            "external_reference_counts_by_act": dict(
                sorted(
                    Counter(
                        source_doc(reference["source_unit_id"])
                        for reference in external_references
                    ).items()
                )
            ),
        },
    }


def iter_reference_candidates(unit: dict[str, Any], text: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for match in ARTICLE_RANGE_RE.finditer(text):
        candidates.append(
            {
                "reference_kind": "range_expanded",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "article_range": (int(match.group("start")), int(match.group("end"))),
            }
        )
    for match in ARTICLE_LIST_RE.finditer(text):
        candidates.append(
            {
                "reference_kind": "list_expanded",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "article_list": [
                    (item.group("number"), item.group("paragraph"))
                    for item in ARTICLE_LIST_ITEM_RE.finditer(match.group("body"))
                ],
            }
        )
    for match in ARTICLE_SINGLE_RE.finditer(text):
        candidates.append(
            {
                "reference_kind": "single",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "article": match.group("number"),
                "paragraph": match.group("paragraph"),
                "point": match.group("point"),
            }
        )
    for match in RECITAL_LIST_RE.finditer(text):
        candidates.append(
            {
                "reference_kind": "list_expanded",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "recital_list": re.findall(r"\d+", match.group("body")),
            }
        )
    for match in RECITAL_RE.finditer(text):
        candidates.append(
            {
                "reference_kind": "recital",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "recital": match.group("number"),
            }
        )
    for match in PARAGRAPH_RE.finditer(text):
        article_id = normalize_to_article_recital(unit["unit_id"])
        if " Art. " not in article_id:
            continue
        candidates.append(
            {
                "reference_kind": "relative_paragraph",
                "raw_match": match.group(0),
                "start": match.start(),
                "end": match.end(),
                "relative_paragraph": match.group("number"),
            }
        )
    candidates.sort(key=lambda candidate: (candidate["start"], candidate["end"]))
    return candidates


def resolve_candidate(
    candidate: dict[str, Any],
    source_unit: dict[str, Any],
    article_ids_by_doc: dict[str, set[str]],
) -> list[str]:
    document_code = source_unit["document_code"]
    if "article_range" in candidate:
        start, end = candidate["article_range"]
        if end < start or end - start > 50:
            return []
        return [
            f"{document_code} Art. {number}"
            for number in range(start, end + 1)
            if f"{document_code} Art. {number}" in article_ids_by_doc[document_code]
        ]
    if "article_list" in candidate:
        unit_ids = []
        for number, paragraph in candidate["article_list"]:
            unit_id = f"{document_code} Art. {number}"
            if paragraph:
                unit_id += f".{paragraph}"
            unit_ids.append(unit_id)
        return unit_ids
    if "article" in candidate:
        unit_id = f"{document_code} Art. {candidate['article']}"
        if candidate.get("paragraph"):
            unit_id += f".{candidate['paragraph']}"
        if candidate.get("point"):
            unit_id += f".{candidate['point']}"
        return [unit_id]
    if "recital_list" in candidate:
        return [f"{document_code} Rec. {number}" for number in candidate["recital_list"]]
    if "recital" in candidate:
        return [f"{document_code} Rec. {candidate['recital']}"]
    if "relative_paragraph" in candidate:
        article_id = normalize_to_article_recital(source_unit["unit_id"])
        return [f"{article_id}.{candidate['relative_paragraph']}"]
    return []


def is_external_reference(text: str, start: int, end: int) -> bool:
    following_context = text[end : min(len(text), end + 80)]
    if LOCAL_ACT_TERMS_RE.search(following_context):
        return False
    window = text[max(0, start - 80) : min(len(text), end + 100)]
    return EXTERNAL_TERMS_RE.search(window) is not None


def source_doc(unit_id: str) -> str:
    return unit_id.split(" ", 1)[0]


def update_validation_report(validation_report_path: Path, xref_graph: dict[str, Any]) -> None:
    report = json.loads(validation_report_path.read_text(encoding="utf-8"))
    report["xref_edge_counts_by_act"] = xref_graph["summary"]["edge_counts_by_act"]
    report["external_reference_counts_by_act"] = xref_graph["summary"][
        "external_reference_counts_by_act"
    ]
    report["xref_spot_check_sample"] = xref_graph["edges"][:20]
    validation_report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_xref_outputs(
    legal_units_path: Path,
    xref_graph_path: Path,
    validation_report_path: Path | None = None,
) -> dict[str, Any]:
    legal_units = json.loads(legal_units_path.read_text(encoding="utf-8"))
    xref_graph = extract_xrefs(legal_units)
    # Bind the graph to its parsed legal units.
    xref_graph["legal_units_sha256"] = hashlib.sha256(
        legal_units_path.read_bytes()
    ).hexdigest()
    xref_graph_path.parent.mkdir(parents=True, exist_ok=True)
    xref_graph_path.write_text(json.dumps(xref_graph, indent=2) + "\n", encoding="utf-8")
    if validation_report_path is not None:
        update_validation_report(validation_report_path, xref_graph)
    return xref_graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract intra-document legal cross-references.")
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    parser.add_argument(
        "--xref-graph-path",
        type=Path,
        default=Path("data/parsed/xref_graph.json"),
    )
    parser.add_argument(
        "--validation-report-path",
        type=Path,
        default=Path("data/parsed/validation_report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    graph = write_xref_outputs(
        legal_units_path=args.legal_units_path,
        xref_graph_path=args.xref_graph_path,
        validation_report_path=args.validation_report_path,
    )
    print(
        "Extracted "
        f"{graph['summary']['edge_count']} intra-document xrefs and excluded "
        f"{graph['summary']['external_reference_count']} external references."
    )


if __name__ == "__main__":
    main()
