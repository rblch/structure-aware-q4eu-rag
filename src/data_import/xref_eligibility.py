from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def apply_structural_xref_eligibility(
    queries: list[dict[str, Any]],
    xref_graph: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    targets_by_doc = {
        document_code: {
            edge["target_unit_id_normalized"]
            for edge in xref_graph["edges"]
            if edge["document_code"] == document_code
        }
        for document_code in {edge["document_code"] for edge in xref_graph["edges"]}
    }

    updated_queries: list[dict[str, Any]] = []
    for query in queries:
        gold_units = set(query["gold_unit_ids"])
        query = dict(query)
        query["gold_multi_unit"] = len(gold_units) >= 2
        query["gold_has_incoming_xref"] = any(
            gold_unit in targets_by_doc.get(gold_unit.split(" ", 1)[0], set())
            for gold_unit in gold_units
        )
        query["xref_eligible"] = False
        updated_queries.append(query)

    report = build_report(updated_queries, xref_graph)
    return updated_queries, report


def build_report(queries: list[dict[str, Any]], xref_graph: dict[str, Any]) -> dict[str, Any]:
    structural_candidates = [
        query
        for query in queries
        if query["gold_multi_unit"] and query["gold_has_incoming_xref"]
    ]
    by_specificity = Counter(query["specificity"] for query in structural_candidates)
    by_act = Counter(
        document_code
        for query in structural_candidates
        for document_code in query["target_document_codes"]
    )
    return {
        "stage": "1c-i",
        "structural_upper_bound_count": len(structural_candidates),
        "structural_upper_bound_by_specificity": dict(sorted(by_specificity.items())),
        "structural_upper_bound_by_act": dict(sorted(by_act.items())),
        "retrieval_conditioned_count": None,
        "retrieval_conditioned_by_specificity": None,
        "retrieval_conditioned_by_act": None,
        "xref_power_decision": "pending_stage_1c_ii",
        "gold_multi_unit_count": sum(query["gold_multi_unit"] for query in queries),
        "gold_has_incoming_xref_count": sum(
            query["gold_has_incoming_xref"] for query in queries
        ),
        "xref_edge_count": xref_graph["summary"]["edge_count"],
    }


def write_structural_eligibility_outputs(
    queries_path: Path,
    xref_graph_path: Path,
    report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    xref_graph = json.loads(xref_graph_path.read_text(encoding="utf-8"))
    updated_queries, report = apply_structural_xref_eligibility(queries, xref_graph)
    queries_path.write_text(json.dumps(updated_queries, indent=2) + "\n", encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return updated_queries, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 1c-i xref eligibility audit.")
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/dataset/q4eu_queries.json"),
    )
    parser.add_argument(
        "--xref-graph-path",
        type=Path,
        default=Path("data/parsed/xref_graph.json"),
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/dataset/xref_eligibility_report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = write_structural_eligibility_outputs(
        queries_path=args.queries_path,
        xref_graph_path=args.xref_graph_path,
        report_path=args.report_path,
    )
    print(
        "Structural xref upper bound: "
        f"{report['structural_upper_bound_count']} questions."
    )


if __name__ == "__main__":
    main()
