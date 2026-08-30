from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from enrichment.rq3_context_assembly import validate_rq3_config
from evaluation.generated_answer_evaluation import load_answer_records
from utils.summary_stats import bootstrap_ci, percentile


STRATEGY_ORDER = ["fixed_size", "semantic", "hierarchical"]
MEASUREMENTS = [
    "context_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "generation_cost_usd",
    "query_embedding_seconds",
    "search_seconds",
    "assembly_seconds",
    "generation_seconds",
    "end_to_end_seconds",
]


def write_rq3_cross_chunking_outputs(
    *,
    config_path: Path,
    rq3_answers_dir: Path,
    hierarchical_answers_path: Path,
    rq3_context_records_path: Path,
    hierarchical_context_records_path: Path,
    retrieval_results_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rq3_config = validate_rq3_config(config)
    new_answers = load_answer_records(rq3_answers_dir)
    hierarchical_answers = json.loads(
        hierarchical_answers_path.read_text(encoding="utf-8")
    )
    new_contexts = json.loads(
        rq3_context_records_path.read_text(encoding="utf-8")
    )
    hierarchical_contexts = json.loads(
        hierarchical_context_records_path.read_text(encoding="utf-8")
    )
    retrieval_results = json.loads(retrieval_results_path.read_text(encoding="utf-8"))

    answers_by_strategy = collect_answers_by_strategy(
        new_answers=new_answers,
        hierarchical_answers=hierarchical_answers,
        rq3_config=rq3_config,
    )
    contexts_by_strategy = collect_contexts_by_strategy(
        new_contexts=new_contexts,
        hierarchical_contexts=hierarchical_contexts,
        rq3_config=rq3_config,
    )
    validate_comparison_inputs(
        answers_by_strategy=answers_by_strategy,
        contexts_by_strategy=contexts_by_strategy,
        rq3_config=rq3_config,
        configured_model=config["models"]["generation"]["model"],
    )
    rows = build_per_query_rows(
        answers_by_strategy=answers_by_strategy,
        contexts_by_strategy=contexts_by_strategy,
        retrieval_results=retrieval_results,
        rq3_config=rq3_config,
    )
    strategy_summary = summarize_strategies(rows)
    contrasts = paired_contrasts(
        rows,
        iterations=config["analysis"]["bootstrap_iterations"],
        confidence=config["analysis"]["bootstrap_ci"],
        random_seed=config["random_seed"],
    )
    summary = {
        "analysis": "rq3_cross_chunking_minimal_extension",
        "status": "post_experiment_descriptive_extension",
        "hierarchical_data_reused": True,
        "latency_comparison_concurrent": False,
        "latency_caveat": (
            "Fixed-size and semantic generations were run later than the reused "
            "hierarchical generation; latency comparisons are descriptive."
        ),
        "search_scope": rq3_config["search_scope"],
        "top_k": rq3_config["top_k"],
        "query_count": rq3_config["expected_query_count"],
        "new_generation_count": len(new_answers),
        "strategy_summary": strategy_summary,
        "paired_contrasts": contrasts,
    }

    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    write_csv(tables_dir / "efficiency_per_query.csv", rows)
    write_csv(
        tables_dir / "efficiency_by_strategy.csv",
        flatten_strategy_summary(strategy_summary),
    )
    write_csv(tables_dir / "paired_efficiency_contrasts.csv", contrasts)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def collect_answers_by_strategy(
    *,
    new_answers: list[dict[str, Any]],
    hierarchical_answers: list[dict[str, Any]],
    rq3_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_condition = {
        condition["id"]: condition for condition in rq3_config["new_conditions"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for answer in new_answers:
        condition = by_condition.get(answer["condition_id"])
        if condition is None:
            raise RuntimeError(
                f"Unexpected answer condition in RQ3 directory: {answer['condition_id']}"
            )
        grouped[condition["strategy"]].append(answer)

    reference = rq3_config["hierarchical_reference"]
    for answer in hierarchical_answers:
        if answer["condition_id"] != reference["condition_id"]:
            raise RuntimeError(
                f"Hierarchical reference file contains {answer['condition_id']}"
            )
        grouped[reference["strategy"]].append(answer)
    return dict(grouped)


def collect_contexts_by_strategy(
    *,
    new_contexts: list[dict[str, Any]],
    hierarchical_contexts: list[dict[str, Any]],
    rq3_config: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    by_condition = {
        condition["id"]: condition for condition in rq3_config["new_conditions"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in new_contexts:
        condition = by_condition.get(record["condition_id"])
        if condition is None:
            raise RuntimeError(f"Unexpected RQ3 context: {record['condition_id']}")
        grouped[condition["strategy"]].append(record)
    reference = rq3_config["hierarchical_reference"]
    grouped[reference["strategy"]] = [
        record
        for record in hierarchical_contexts
        if record["condition_id"] == reference["condition_id"]
    ]
    return dict(grouped)


def validate_comparison_inputs(
    *,
    answers_by_strategy: dict[str, list[dict[str, Any]]],
    contexts_by_strategy: dict[str, list[dict[str, Any]]],
    rq3_config: dict[str, Any],
    configured_model: str,
) -> None:
    expected_count = rq3_config["expected_query_count"]
    expected_query_ids: set[str] | None = None
    prompt_hashes: set[str] = set()
    for strategy in STRATEGY_ORDER:
        answers = answers_by_strategy.get(strategy, [])
        contexts = contexts_by_strategy.get(strategy, [])
        if len(answers) != expected_count or len(contexts) != expected_count:
            raise RuntimeError(
                f"RQ3 {strategy} requires {expected_count} answers and contexts; "
                f"found {len(answers)} and {len(contexts)}"
            )
        answer_query_ids = {answer["query_id"] for answer in answers}
        context_query_ids = {record["query_id"] for record in contexts}
        if len(answer_query_ids) != expected_count or answer_query_ids != context_query_ids:
            raise RuntimeError(f"RQ3 {strategy} has incomplete or duplicate query IDs")
        if expected_query_ids is None:
            expected_query_ids = answer_query_ids
        elif answer_query_ids != expected_query_ids:
            raise RuntimeError("RQ3 strategies do not cover the same queries")
        for answer in answers:
            if answer["generation_model"] != configured_model:
                raise RuntimeError(f"RQ3 {strategy} used a different generation model")
            if answer["search_scope"] != rq3_config["search_scope"]:
                raise RuntimeError(f"RQ3 {strategy} used a different search scope")
            if answer["base_top_k"] != rq3_config["top_k"]:
                raise RuntimeError(f"RQ3 {strategy} used a different top-k")
            if not answer.get("usage") or answer.get("estimated_cost_usd") is None:
                raise RuntimeError(f"RQ3 {strategy} is missing usage or cost data")
            prompt_hashes.add(answer["prompt_sha256"])
    if len(prompt_hashes) != 1:
        raise RuntimeError("RQ3 strategies did not use the same generation prompt")


def build_per_query_rows(
    *,
    answers_by_strategy: dict[str, list[dict[str, Any]]],
    contexts_by_strategy: dict[str, list[dict[str, Any]]],
    retrieval_results: list[dict[str, Any]],
    rq3_config: dict[str, Any],
) -> list[dict[str, Any]]:
    config_by_strategy = {
        condition["strategy"]: condition["config_id"]
        for condition in rq3_config["new_conditions"]
    }
    reference = rq3_config["hierarchical_reference"]
    config_by_strategy[reference["strategy"]] = reference["config_id"]
    retrieval_by_key = {
        (record["config_id"], record["query_id"]): record
        for record in retrieval_results
        if record["search_scope"] == rq3_config["search_scope"]
    }
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGY_ORDER:
        answers = {answer["query_id"]: answer for answer in answers_by_strategy[strategy]}
        contexts = {
            record["query_id"]: record for record in contexts_by_strategy[strategy]
        }
        config_id = config_by_strategy[strategy]
        for query_id in sorted(answers):
            answer = answers[query_id]
            context = contexts[query_id]
            retrieval = retrieval_by_key.get((config_id, query_id))
            if retrieval is None:
                raise RuntimeError(f"Missing RQ3 retrieval timing for {config_id}/{query_id}")
            usage = answer["usage"]
            embedding_seconds = retrieval["query_embedding_seconds"]
            search_seconds = retrieval["search_seconds"]
            assembly_seconds = context.get("assembly_seconds")
            generation_seconds = answer["elapsed_seconds"]
            if None in (
                embedding_seconds,
                search_seconds,
                assembly_seconds,
                generation_seconds,
            ):
                raise RuntimeError(f"Incomplete RQ3 latency data for {strategy}/{query_id}")
            rows.append(
                {
                    "query_id": query_id,
                    "strategy": strategy,
                    "config_id": config_id,
                    "condition_id": answer["condition_id"],
                    "context_tokens": answer["context_token_count"],
                    "prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "generation_cost_usd": answer["estimated_cost_usd"],
                    "query_embedding_seconds": embedding_seconds,
                    "search_seconds": search_seconds,
                    "assembly_seconds": assembly_seconds,
                    "generation_seconds": generation_seconds,
                    "end_to_end_seconds": (
                        embedding_seconds
                        + search_seconds
                        + assembly_seconds
                        + generation_seconds
                    ),
                    "generated_at_utc": answer["generated_at_utc"],
                    "generation_model": answer["generation_model"],
                    "prompt_sha256": answer["prompt_sha256"],
                }
            )
    return rows


def summarize_strategies(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)
    return {
        strategy: {
            "query_count": len(grouped[strategy]),
            **{
                measurement: numeric_summary(
                    [float(row[measurement]) for row in grouped[strategy]]
                )
                for measurement in MEASUREMENTS
            },
        }
        for strategy in STRATEGY_ORDER
    }


def paired_contrasts(
    rows: list[dict[str, Any]],
    *,
    iterations: int,
    confidence: float,
    random_seed: int,
) -> list[dict[str, Any]]:
    by_key = {
        (row["strategy"], row["query_id"]): row
        for row in rows
    }
    query_ids = sorted(
        row["query_id"] for row in rows if row["strategy"] == "hierarchical"
    )
    contrasts: list[dict[str, Any]] = []
    for strategy in ("fixed_size", "semantic"):
        for metric in MEASUREMENTS:
            treatment = [float(by_key[(strategy, query_id)][metric]) for query_id in query_ids]
            reference = [
                float(by_key[("hierarchical", query_id)][metric])
                for query_id in query_ids
            ]
            differences = [left - right for left, right in zip(treatment, reference)]
            lower, upper = bootstrap_ci(
                differences,
                iterations=iterations,
                confidence=confidence,
                random_seed=random_seed,
            )
            reference_mean = sum(reference) / len(reference)
            contrasts.append(
                {
                    "contrast": f"{strategy} - hierarchical",
                    "metric": metric,
                    "query_count": len(differences),
                    "comparison_mean": sum(treatment) / len(treatment),
                    "hierarchical_mean": reference_mean,
                    "mean_paired_difference": sum(differences) / len(differences),
                    "median_paired_difference": percentile(differences, 50),
                    "bootstrap_ci_lower": lower,
                    "bootstrap_ci_upper": upper,
                    "relative_mean_difference": (
                        (sum(treatment) / len(treatment)) / reference_mean - 1
                        if reference_mean
                        else None
                    ),
                    "latency_comparison_concurrent": False
                    if metric.endswith("seconds")
                    else "",
                }
            )
    return contrasts


def numeric_summary(values: list[float]) -> dict[str, float]:
    return {
        "total": sum(values),
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "min": min(values),
        "max": max(values),
    }


def flatten_strategy_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGY_ORDER:
        row: dict[str, Any] = {
            "strategy": strategy,
            "query_count": summary[strategy]["query_count"],
        }
        for measurement in MEASUREMENTS:
            for statistic, value in summary[strategy][measurement].items():
                row[f"{measurement}_{statistic}"] = value
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty RQ3 table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
