from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import.id_normalization import DOCUMENT_CODES
from parsing.akn_parser import parse_all_akn_documents
from parsing.legal_unit import LegalUnit
from parsing.warrant_html_parser import parse_warrant_document


def parse_corpus(raw_documents_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    akn_units, canonical_texts = parse_all_akn_documents(raw_documents_dir)
    warrant_units, warrant_canonical = parse_warrant_document(raw_documents_dir / "warrant.html")
    all_units = akn_units + warrant_units
    canonical_texts["W"] = warrant_canonical
    validation_report = build_validation_report(all_units)
    return [unit.to_dict() for unit in all_units], canonical_texts, validation_report


def build_validation_report(units: list[LegalUnit]) -> dict[str, Any]:
    counts_by_act: dict[str, Counter[str]] = {}
    units_by_id = {unit.unit_id: unit for unit in units}
    id_counts = Counter(unit.unit_id for unit in units)
    duplicate_unit_ids = sorted(
        unit_id for unit_id, count in id_counts.items() if count > 1
    )
    for unit in units:
        counts_by_act.setdefault(unit.document_code, Counter())[unit.unit_type] += 1

    warrant_spot_checks = {}
    for unit_id in ["W Rec. 12", "W Art. 4a", "W Art. 30"]:
        unit = units_by_id.get(unit_id)
        warrant_spot_checks[unit_id] = {
            "exists": unit is not None,
            "text_preview": unit.text[:240] if unit else None,
        }

    return {
        "legal_unit_count": len(units),
        "duplicate_unit_ids": duplicate_unit_ids,
        "counts_by_act": {
            document_code: dict(sorted(counter.items()))
            for document_code, counter in sorted(counts_by_act.items())
        },
        "warrant_spot_checks": warrant_spot_checks,
        "units_missing_document_code": [
            unit.unit_id for unit in units if not unit.document_code
        ],
        "units_missing_legal_order": [
            unit.unit_id for unit in units if "legal_order" not in unit.metadata
        ],
    }


def assert_valid_corpus(validation_report: dict[str, Any]) -> None:
    problems: list[str] = []
    if validation_report["duplicate_unit_ids"]:
        problems.append(
            f"duplicate unit IDs: {validation_report['duplicate_unit_ids'][:10]}"
        )
    missing_acts = sorted(
        set(DOCUMENT_CODES) - set(validation_report["counts_by_act"])
    )
    if missing_acts:
        problems.append(f"acts not parsed: {missing_acts}")
    failed_spot_checks = sorted(
        unit_id
        for unit_id, check in validation_report["warrant_spot_checks"].items()
        if not check["exists"]
    )
    if failed_spot_checks:
        problems.append(f"warrant spot checks failed: {failed_spot_checks}")
    if validation_report["units_missing_document_code"]:
        problems.append(
            "units missing document code: "
            f"{validation_report['units_missing_document_code'][:10]}"
        )
    if validation_report["units_missing_legal_order"]:
        problems.append(
            "units missing legal order: "
            f"{validation_report['units_missing_legal_order'][:10]}"
        )
    if problems:
        raise ValueError("Corpus validation failed: " + "; ".join(problems))


def write_corpus_outputs(
    raw_documents_dir: Path,
    legal_units_path: Path,
    canonical_texts_path: Path,
    validation_report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    legal_units, canonical_texts, validation_report = parse_corpus(raw_documents_dir)
    legal_units_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_texts_path.parent.mkdir(parents=True, exist_ok=True)
    validation_report_path.parent.mkdir(parents=True, exist_ok=True)
    legal_units_path.write_text(json.dumps(legal_units, indent=2) + "\n", encoding="utf-8")
    canonical_texts_path.write_text(
        json.dumps(canonical_texts, indent=2) + "\n",
        encoding="utf-8",
    )
    validation_report_path.write_text(
        json.dumps(validation_report, indent=2) + "\n",
        encoding="utf-8",
    )
    # Validate after writing so the report is available for inspection.
    assert_valid_corpus(validation_report)
    return legal_units, canonical_texts, validation_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse Q4EU legal source corpus.")
    parser.add_argument("--raw-documents-dir", type=Path, default=Path("data/raw/discolqa/documents"))
    parser.add_argument("--legal-units-path", type=Path, default=Path("data/parsed/legal_units.json"))
    parser.add_argument(
        "--canonical-texts-path",
        type=Path,
        default=Path("data/parsed/canonical_texts.json"),
    )
    parser.add_argument(
        "--validation-report-path",
        type=Path,
        default=Path("data/parsed/validation_report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    legal_units, _, validation_report = write_corpus_outputs(
        raw_documents_dir=args.raw_documents_dir,
        legal_units_path=args.legal_units_path,
        canonical_texts_path=args.canonical_texts_path,
        validation_report_path=args.validation_report_path,
    )
    print(f"Parsed {len(legal_units)} legal units.")
    print(json.dumps(validation_report["counts_by_act"], indent=2))


if __name__ == "__main__":
    main()
