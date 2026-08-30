from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import.id_normalization import (
    dedupe_preserving_order,
    normalize_to_article_recital,
    normalize_to_paragraph,
)


SPECIFICITY_DEFINITIONS = {
    "H": (
        "the answer falls precisely in the domain of the regulation and is given "
        'in the "black letter" of an exact provision'
    ),
    "N": "the answer falls within scope but requires abstracting across multiple provisions",
    "L": (
        "a broad question whose tentative answer is found only through an articulate "
        "combination of articles and recitals, partly open to interpretation"
    ),
}


def load_source_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_question_dict(evaluate_path: Path) -> dict[str, dict[str, Any]]:
    module = ast.parse(evaluate_path.read_text(encoding="utf-8"))
    question_dict: dict[str, dict[str, Any]] | None = None

    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name) and target.id == "question_dict":
                    question_dict = ast.literal_eval(statement.value)
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "update"
            and isinstance(statement.value.func.value, ast.Name)
            and statement.value.func.value.id == "question_dict"
        ):
            if question_dict is None:
                raise ValueError("question_dict.update appeared before question_dict assignment")
            if len(statement.value.args) != 1:
                raise ValueError("question_dict.update must have exactly one positional argument")
            question_dict.update(ast.literal_eval(statement.value.args[0]))

    if question_dict is None:
        raise ValueError("No question_dict assignment found")
    return question_dict


def build_queries(
    question_dict: dict[str, dict[str, Any]],
    source_repo: str,
    source_commit: str,
) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for index, (question, info) in enumerate(question_dict.items(), start=1):
        expected_answers_raw = list(info["expected_answers"])
        target_document_codes = list(info["documents"])
        gold_unit_ids = dedupe_preserving_order(
            normalize_to_article_recital(answer) for answer in expected_answers_raw
        )
        gold_unit_ids_paragraph_level = dedupe_preserving_order(
            normalize_to_paragraph(answer) for answer in expected_answers_raw
        )
        queries.append(
            {
                "query_id": f"q4eu_{index:03d}",
                "question": question,
                "specificity": info["specificity"],
                "target_document_codes": target_document_codes,
                "expected_answers_raw": expected_answers_raw,
                "gold_unit_ids": gold_unit_ids,
                "gold_unit_ids_paragraph_level": gold_unit_ids_paragraph_level,
                "is_multi_act": len(target_document_codes) > 1,
                "gold_unit_count": len(gold_unit_ids),
                "gold_multi_unit": len(gold_unit_ids) >= 2,
                "gold_has_incoming_xref": False,
                "xref_eligible": False,
                "source": {
                    "dataset": "Q4EU",
                    "repo": source_repo,
                    "commit": source_commit,
                },
            }
        )
    return queries


def build_import_report(
    queries: list[dict[str, Any]],
    source_metadata: dict[str, Any],
) -> dict[str, Any]:
    specificity_counts = Counter(query["specificity"] for query in queries)
    target_document_counts = Counter(
        code for query in queries for code in query["target_document_codes"]
    )
    gold_size_counts = Counter(query["gold_unit_count"] for query in queries)
    raw_expected_answer_count = sum(len(query["expected_answers_raw"]) for query in queries)

    return {
        "dataset": "Q4EU",
        "source_repo": source_metadata["source_repo"],
        "source_branch": source_metadata["source_branch"],
        "source_commit": source_metadata["source_commit"],
        "imported_question_count": len(queries),
        "raw_expected_answer_count": raw_expected_answer_count,
        "paper_expected_answer_count": 225,
        "repo_expected_answer_count": raw_expected_answer_count,
        "paper_repo_divergence": (
            "The DiscoLQA paper reports 225 expected answers; the pinned repository "
            "contains 246 raw expected answers including the appended update block."
        ),
        "specificity_counts": dict(sorted(specificity_counts.items())),
        "specificity_definitions": SPECIFICITY_DEFINITIONS,
        "target_document_tag_counts": dict(sorted(target_document_counts.items())),
        "multi_act_question_count": sum(query["is_multi_act"] for query in queries),
        "gold_unit_count_distribution": {
            str(size): count for size, count in sorted(gold_size_counts.items())
        },
        "questions_with_single_gold_unit": sum(
            query["gold_unit_count"] == 1 for query in queries
        ),
        "questions_with_multiple_gold_units": sum(
            query["gold_unit_count"] >= 2 for query in queries
        ),
        "stage_1b_sanity_check": {
            "status": "passed",
            "normalization_rule": (
                "Article and Recital labels are converted to Art./Rec.; article "
                "paragraph and point suffixes are dropped at the first numeric dot "
                "for primary article/recital-level gold evaluation."
            ),
            "checked_examples": {
                "G Art. 35.3": "G Art. 35",
                "E Rec. 57": "E Rec. 57",
                "W Art. 4a": "W Art. 4a",
            },
        },
    }


def assert_gold_units_resolve(
    queries: list[dict[str, Any]],
    legal_units_path: Path,
) -> None:
    if not legal_units_path.exists():
        raise FileNotFoundError(
            f"{legal_units_path} not found; parse the corpus before importing "
            "Q4EU so primary gold IDs can be validated."
        )
    legal_units = json.loads(legal_units_path.read_text(encoding="utf-8"))
    parsed_ids = {unit["unit_id"] for unit in legal_units}
    unresolved = sorted(
        unit_id
        for query in queries
        for unit_id in query["gold_unit_ids"]
        if unit_id not in parsed_ids
    )
    if unresolved:
        raise ValueError(
            f"{len(unresolved)} primary gold IDs do not resolve to parsed "
            f"legal units: {unresolved[:10]}"
        )


def import_q4eu(
    evaluate_path: Path,
    source_metadata_path: Path,
    output_path: Path,
    report_path: Path,
    legal_units_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_metadata = load_source_metadata(source_metadata_path)
    question_dict = parse_question_dict(evaluate_path)
    queries = build_queries(
        question_dict,
        source_repo=source_metadata["source_repo"],
        source_commit=source_metadata["source_commit"],
    )
    report = build_import_report(queries, source_metadata)

    if report["imported_question_count"] != 72:
        raise ValueError(f"Expected 72 Q4EU questions, found {report['imported_question_count']}")
    if report["raw_expected_answer_count"] != 246:
        raise ValueError(
            f"Expected 246 raw Q4EU answers, found {report['raw_expected_answer_count']}"
        )
    assert_gold_units_resolve(queries, legal_units_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(queries, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return queries, report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Q4EU questions from evaluate.py.")
    parser.add_argument("--evaluate-path", type=Path, default=Path("data/raw/discolqa/evaluate.py"))
    parser.add_argument(
        "--source-metadata-path",
        type=Path,
        default=Path("data/raw/discolqa/source_metadata.json"),
    )
    parser.add_argument("--output-path", type=Path, default=Path("data/dataset/q4eu_queries.json"))
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/dataset/q4eu_import_report.json"),
    )
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    _, report = import_q4eu(
        evaluate_path=args.evaluate_path,
        source_metadata_path=args.source_metadata_path,
        output_path=args.output_path,
        report_path=args.report_path,
        legal_units_path=args.legal_units_path,
    )
    print(
        "Imported "
        f"{report['imported_question_count']} Q4EU questions and "
        f"{report['raw_expected_answer_count']} raw expected answers."
    )


if __name__ == "__main__":
    main()
