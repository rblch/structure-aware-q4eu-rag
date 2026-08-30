from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


FAITHFULNESS_BANDS = [
    ("low", 0.0, 0.5),
    ("mid", 0.5, 0.8),
    ("high", 0.8, 1.000000001),
]
MANUAL_COLUMNS = [
    "manual_judge_score_reasonable",
    "manual_faithfulness_score",
    "manual_correctness_score",
    "manual_unsupported_claims_missed",
    "manual_abstention_handled_correctly",
    "manual_notes",
    "reviewer_id",
    "reviewed_at_utc",
]
AUDIT_COLUMNS = [
    "audit_id",
    "sample_rank",
    "condition_id",
    "faithfulness_band",
    "query_id",
    "question",
    "target_document_codes",
    "gold_unit_ids",
    "context_answer_unit_ids",
    "context_gold_recall",
    "faithfulness_score",
    "correctness_score",
    "answer_abstains",
    "faithfulness_abstention_justified",
    "correctness_answer_abstains",
    "correctness_abstention_justified",
    "either_judge_unjustified_abstention",
    "citation_precision",
    "citation_recall",
    "citation_f1",
    "cited_unit_ids",
    "extra_citation_unit_ids",
    "missing_gold_citation_unit_ids",
    "faithfulness_unfaithful_claims",
    "faithfulness_supporting_evidence",
    "faithfulness_judge_rationale",
    "correctness_material_omissions",
    "correctness_incorrect_claims",
    "correctness_supporting_gold_evidence",
    "correctness_judge_rationale",
    "answer_text",
    "context_text",
    *MANUAL_COLUMNS,
]


def write_manual_audit_outputs(
    *,
    config_path: Path,
    answer_evaluation_records_path: Path,
    answers_dir: Path,
    faithfulness_scores_path: Path,
    correctness_scores_path: Path,
    output_csv_path: Path,
    metadata_path: Path,
    sample_fraction: float | None = None,
    sample_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records = json.loads(answer_evaluation_records_path.read_text(encoding="utf-8"))
    faithfulness_scores = keyed_records(
        json.loads(faithfulness_scores_path.read_text(encoding="utf-8"))
    )
    correctness_scores = keyed_records(
        json.loads(correctness_scores_path.read_text(encoding="utf-8"))
    )
    answers = keyed_answers(load_answers(answers_dir))
    settings = manual_audit_settings_from_config(config)
    fraction = sample_fraction if sample_fraction is not None else settings["fraction"]
    target_size = sample_size if sample_size is not None else round(len(records) * fraction)

    sampled_records, sample_metadata = stratified_sample(
        records=records,
        sample_size=target_size,
        random_seed=settings["random_seed"],
    )
    rows = [
        build_audit_row(
            index=index,
            record=record,
            answer=answers.get(record_key(record), {}),
            faithfulness=faithfulness_scores.get(record_key(record), {}),
            correctness=correctness_scores.get(record_key(record), {}),
        )
        for index, record in enumerate(sampled_records, start=1)
    ]
    metadata = {
        "status": "pending_manual_review",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_record_count": len(records),
        "input_paths": {
            "answer_evaluation_records": str(answer_evaluation_records_path),
            "answers_dir": str(answers_dir),
            "faithfulness_scores": str(faithfulness_scores_path),
            "correctness_scores": str(correctness_scores_path),
        },
        "sample_fraction": fraction,
        "target_sample_size": target_size,
        "sampled_record_count": len(rows),
        "random_seed": settings["random_seed"],
        "faithfulness_bands": [
            {"id": band_id, "lower": lower, "upper": upper}
            for band_id, lower, upper in FAITHFULNESS_BANDS
        ],
        "stratum_counts": sample_metadata["stratum_counts"],
        "stratum_sample_counts": sample_metadata["stratum_sample_counts"],
        "condition_sample_targets": sample_metadata["condition_sample_targets"],
        "manual_columns": MANUAL_COLUMNS,
        "completion_rule": (
            "Complete all manual_* columns, reviewer_id, and reviewed_at_utc "
            "before using this audit as evidence for LLM-judge validity."
        ),
        "selected_records": [
            {
                "audit_id": row["audit_id"],
                "condition_id": row["condition_id"],
                "faithfulness_band": row["faithfulness_band"],
                "query_id": row["query_id"],
            }
            for row in rows
        ],
    }
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv_path, rows)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return rows, metadata


def manual_audit_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    analysis_config = config.get("analysis", {})
    return {
        "fraction": float(analysis_config.get("manual_audit_fraction", 0.10)),
        "random_seed": int(config.get("random_seed", 42)),
    }


def stratified_sample(
    *,
    records: list[dict[str, Any]],
    sample_size: int,
    random_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated = [
        {**record, "faithfulness_band": faithfulness_band(record["faithfulness_score"])}
        for record in records
    ]
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in annotated:
        by_condition[record["condition_id"]].append(record)

    condition_targets = allocate_condition_targets(annotated, sample_size)
    rng = random.Random(random_seed)
    sampled: list[dict[str, Any]] = []
    stratum_sample_counts: Counter[tuple[str, str]] = Counter()
    for condition_id in sorted(by_condition):
        condition_records = by_condition[condition_id]
        target = condition_targets[condition_id]
        by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in condition_records:
            by_band[record["faithfulness_band"]].append(record)
        band_targets = allocate_band_targets(by_band, target)
        for band_id in sorted(band_targets):
            candidates = sorted(
                by_band[band_id],
                key=lambda record: (record["query_id"], record["condition_id"]),
            )
            selected = rng.sample(candidates, band_targets[band_id])
            sampled.extend(selected)
            stratum_sample_counts[(condition_id, band_id)] += len(selected)

    sampled.sort(key=lambda record: (record["condition_id"], record["query_id"]))
    stratum_counts = Counter(
        (record["condition_id"], record["faithfulness_band"]) for record in annotated
    )
    metadata = {
        "condition_sample_targets": dict(sorted(condition_targets.items())),
        "stratum_counts": stringify_counter(stratum_counts),
        "stratum_sample_counts": stringify_counter(stratum_sample_counts),
    }
    return sampled, metadata


def allocate_condition_targets(
    records: list[dict[str, Any]],
    sample_size: int,
) -> dict[str, int]:
    condition_counts = Counter(record["condition_id"] for record in records)
    rare_counts = Counter(
        record["condition_id"]
        for record in records
        if record["faithfulness_band"] != "high"
    )
    raw = {
        condition_id: sample_size * count / len(records)
        for condition_id, count in condition_counts.items()
    }
    targets = {condition_id: int(value) for condition_id, value in raw.items()}
    remaining = sample_size - sum(targets.values())
    ranked = sorted(
        condition_counts,
        key=lambda condition_id: (
            raw[condition_id] - targets[condition_id],
            rare_counts[condition_id],
            condition_counts[condition_id],
            condition_id,
        ),
        reverse=True,
    )
    for condition_id in ranked[:remaining]:
        targets[condition_id] += 1
    return targets


def allocate_band_targets(
    by_band: dict[str, list[dict[str, Any]]],
    target: int,
) -> dict[str, int]:
    band_counts = {band_id: len(records) for band_id, records in by_band.items()}
    minimum = {band_id: 1 for band_id, count in band_counts.items() if count > 0}
    if sum(minimum.values()) > target:
        ranked = sorted(band_counts, key=lambda band_id: band_counts[band_id])
        return {band_id: 1 for band_id in ranked[:target]}

    targets = dict(minimum)
    remaining = target - sum(targets.values())
    available = {
        band_id: band_counts[band_id] - targets.get(band_id, 0)
        for band_id in band_counts
    }
    total_available = sum(available.values())
    if remaining <= 0 or total_available == 0:
        return targets

    raw = {
        band_id: remaining * count / total_available
        for band_id, count in available.items()
    }
    additions = {band_id: min(available[band_id], int(value)) for band_id, value in raw.items()}
    for band_id, addition in additions.items():
        targets[band_id] = targets.get(band_id, 0) + addition
    remainder = remaining - sum(additions.values())
    ranked = sorted(
        available,
        key=lambda band_id: (
            raw[band_id] - additions[band_id],
            available[band_id],
            band_id,
        ),
        reverse=True,
    )
    for band_id in ranked:
        if remainder == 0:
            break
        if targets.get(band_id, 0) < band_counts[band_id]:
            targets[band_id] = targets.get(band_id, 0) + 1
            remainder -= 1
    return dict(sorted(targets.items()))


def faithfulness_band(score: float) -> str:
    for band_id, lower, upper in FAITHFULNESS_BANDS:
        if lower <= score < upper:
            return band_id
    if score == 1.0:
        return "high"
    raise ValueError(f"faithfulness_score is outside [0, 1]: {score}")


def build_audit_row(
    *,
    index: int,
    record: dict[str, Any],
    answer: dict[str, Any],
    faithfulness: dict[str, Any],
    correctness: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "audit_id": f"audit_{index:03d}",
        "sample_rank": index,
        "condition_id": record["condition_id"],
        "faithfulness_band": record["faithfulness_band"],
        "query_id": record["query_id"],
        "question": record["question"],
        "target_document_codes": json_cell(answer.get("target_document_codes", [])),
        "gold_unit_ids": json_cell(record["gold_unit_ids"]),
        "context_answer_unit_ids": json_cell(record["context_answer_unit_ids"]),
        "context_gold_recall": record["context_gold_recall"],
        "faithfulness_score": record["faithfulness_score"],
        "correctness_score": record["correctness_score"],
        "answer_abstains": record["answer_abstains"],
        "faithfulness_abstention_justified": record["faithfulness_abstention_justified"],
        "correctness_answer_abstains": record["correctness_answer_abstains"],
        "correctness_abstention_justified": record[
            "correctness_abstention_justified"
        ],
        "either_judge_unjustified_abstention": record[
            "either_judge_unjustified_abstention"
        ],
        "citation_precision": record["citation_precision"],
        "citation_recall": record["citation_recall"],
        "citation_f1": record["citation_f1"],
        "cited_unit_ids": json_cell(record["cited_unit_ids"]),
        "extra_citation_unit_ids": json_cell(record["extra_citation_unit_ids"]),
        "missing_gold_citation_unit_ids": json_cell(
            record["missing_gold_citation_unit_ids"]
        ),
        "faithfulness_unfaithful_claims": json_cell(
            faithfulness.get("unfaithful_claims", [])
        ),
        "faithfulness_supporting_evidence": json_cell(
            faithfulness.get("supporting_evidence", [])
        ),
        "faithfulness_judge_rationale": faithfulness.get("rationale", ""),
        "correctness_material_omissions": json_cell(
            correctness.get("material_omissions", [])
        ),
        "correctness_incorrect_claims": json_cell(
            correctness.get("incorrect_claims", [])
        ),
        "correctness_supporting_gold_evidence": json_cell(
            correctness.get("supporting_gold_evidence", [])
        ),
        "correctness_judge_rationale": correctness.get("rationale", ""),
        "answer_text": record["answer_text"],
        "context_text": answer.get("context_text", ""),
    }
    row.update({column: "" for column in MANUAL_COLUMNS})
    return row


def load_answers(answers_dir: Path) -> list[dict[str, Any]]:
    answers = []
    for path in sorted(answers_dir.glob("*/answers.json")):
        answers.extend(json.loads(path.read_text(encoding="utf-8")))
    return answers


def keyed_answers(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {record_key(record): record for record in records}


def keyed_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {record_key(record): record for record in records}


def record_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["condition_id"], record["query_id"]


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def stringify_counter(counter: Counter[tuple[str, str]]) -> dict[str, int]:
    return {
        f"{condition_id}|{band_id}": count
        for (condition_id, band_id), count in sorted(counter.items())
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a manual judge-audit packet.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--answer-evaluation-records-path",
        type=Path,
        default=Path("data/evaluation/generated_answer_evaluation.json"),
    )
    parser.add_argument("--answers-dir", type=Path, default=Path("data/generation"))
    parser.add_argument(
        "--faithfulness-scores-path",
        type=Path,
        default=Path("data/evaluation/faithfulness_scores.json"),
    )
    parser.add_argument(
        "--correctness-scores-path",
        type=Path,
        default=Path("data/evaluation/correctness_scores.json"),
    )
    parser.add_argument(
        "--output-csv-path",
        type=Path,
        default=Path("data/audit/manual_judge_audit_sample.csv"),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("data/audit/manual_judge_audit_metadata.json"),
    )
    parser.add_argument("--sample-fraction", type=float)
    parser.add_argument("--sample-size", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, _ = write_manual_audit_outputs(
        config_path=args.config_path,
        answer_evaluation_records_path=args.answer_evaluation_records_path,
        answers_dir=args.answers_dir,
        faithfulness_scores_path=args.faithfulness_scores_path,
        correctness_scores_path=args.correctness_scores_path,
        output_csv_path=args.output_csv_path,
        metadata_path=args.metadata_path,
        sample_fraction=args.sample_fraction,
        sample_size=args.sample_size,
    )
    print(f"Manual audit sample complete: {len(rows)} records.")


if __name__ == "__main__":
    main()
