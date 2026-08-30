from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from evaluation.manual_audit import MANUAL_COLUMNS

REVIEW_COLUMNS = ["audit_id", *MANUAL_COLUMNS]
PHASE_ONE_COLUMNS = [
    "manual_faithfulness_score",
    "manual_correctness_score",
    "manual_abstention_handled_correctly",
]
COMPLETION_COLUMNS = [
    "manual_judge_score_reasonable",
    *PHASE_ONE_COLUMNS,
    "manual_unsupported_claims_missed",
    "reviewer_id",
    "reviewed_at_utc",
]
CHOICE_VALUES = {
    "manual_judge_score_reasonable": {"Yes", "No", "Unclear"},
    "manual_unsupported_claims_missed": {"Yes", "No", "Unclear"},
    "manual_abstention_handled_correctly": {
        "Yes",
        "No",
        "Not applicable",
        "Unclear",
    },
}
REVIEWER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def load_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Audit packet is empty: {path}")
    required = {
        "audit_id",
        "question",
        "answer_text",
        "context_text",
        "gold_unit_ids",
        "faithfulness_score",
        "correctness_score",
        "faithfulness_unfaithful_claims",
        "faithfulness_judge_rationale",
        "correctness_judge_rationale",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"Audit packet is missing columns: {sorted(missing)}")
    audit_ids = [row["audit_id"] for row in rows]
    if len(audit_ids) != len(set(audit_ids)) or any(not value for value in audit_ids):
        raise ValueError("Audit packet must contain unique, non-empty audit_id values")
    return rows


def load_legal_units(path: Path) -> dict[str, str]:
    records = json.loads(path.read_text(encoding="utf-8"))
    return {record["unit_id"]: record["text"] for record in records}


def gold_reference_text(row: dict[str, str], legal_units: dict[str, str]) -> str:
    unit_ids = json.loads(row["gold_unit_ids"])
    sections = []
    for unit_id in unit_ids:
        text = legal_units.get(unit_id)
        sections.append(f"[{unit_id}]\n{text or '[Gold unit text not found]'}")
    return "\n\n".join(sections)


def reviewer_output_path(reviews_dir: Path, reviewer_id: str) -> Path:
    reviewer_id = reviewer_id.strip()
    if not REVIEWER_ID_PATTERN.fullmatch(reviewer_id):
        raise ValueError(
            "Reviewer ID must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens (maximum 64 characters)."
        )
    return reviews_dir / f"reviewer_{reviewer_id}.csv"


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["audit_id"]: row for row in rows}


def save_review(path: Path, review: dict[str, Any]) -> None:
    normalized = {column: str(review.get(column, "")) for column in REVIEW_COLUMNS}
    validate_review(normalized)
    reviews = load_reviews(path)
    reviews[normalized["audit_id"]] = normalized
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(reviews[audit_id] for audit_id in sorted(reviews))
    temporary_path.replace(path)


def validate_review(review: dict[str, str]) -> None:
    if not review.get("audit_id"):
        raise ValueError("audit_id is required")
    for column in ("manual_faithfulness_score", "manual_correctness_score"):
        value = review.get(column, "")
        if value and float(value) not in {step / 10 for step in range(11)}:
            raise ValueError(f"{column} must be between 0.0 and 1.0 in 0.1 steps")
    for column, allowed in CHOICE_VALUES.items():
        value = review.get(column, "")
        if value and value not in allowed:
            raise ValueError(f"Invalid {column}: {value}")


def phase_one_complete(review: dict[str, str]) -> bool:
    return all(review.get(column, "") != "" for column in PHASE_ONE_COLUMNS)


def review_complete(review: dict[str, str]) -> bool:
    return all(review.get(column, "") != "" for column in COMPLETION_COLUMNS)


def review_order(
    rows: list[dict[str, str]], random_seed: int = 42
) -> list[dict[str, str]]:
    ordered = list(rows)
    random.Random(random_seed).shuffle(ordered)
    return ordered


def merge_reviews(
    audit_rows: list[dict[str, str]], reviews: dict[str, dict[str, str]]
) -> list[dict[str, str]]:
    merged = []
    for audit_row in audit_rows:
        row = dict(audit_row)
        review = reviews.get(row["audit_id"], {})
        row.update({column: review.get(column, "") for column in MANUAL_COLUMNS})
        merged.append(row)
    return merged


def rows_to_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    from io import StringIO

    handle = StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()
