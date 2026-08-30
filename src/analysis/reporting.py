from __future__ import annotations

import argparse
import csv
import html
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from scipy import stats
import yaml

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.gold_text_coverage import (
    gold_text_coverage_band_table,
    gold_text_coverage_table,
)
from utils.summary_stats import (  # noqa: F401  (re-exported for callers/tests)
    average_ranks,
    bootstrap_ci,
    holm_adjusted_p_values,
    percentile,
    scipy_wilcoxon_method,
    wilcoxon_signed_rank,
)


METRIC_KEYS = [
    "precision",
    "recall",
    "f1",
    "r_precision",
    "mrr",
    "ndcg",
    "retrieved_answer_unit_count",
    "relevant_retrieved_count",
]
SENSITIVITY_CONFIG_ORDER = [
    "fs_64_12",
    "fs_128_25",
    "fs_256_50",
    "sem_50_64",
    "sem_50_128",
    "sem_50_256",
    "sem_70_64",
    "sem_70_128",
    "sem_70_256",
    "hier_article",
    "hier_paragraph",
    "hier_subparagraph",
    "hier_paragraph_contextualized",
]
RQ1_STRATEGY_ORDER = ["fixed_size", "semantic", "hierarchical"]
RQ1_FALLBACK_CONFIRMATORY_CONFIGS = ["fs_256_50", "sem_50_256", "hier_paragraph"]
RQ1_CONFIRMATORY_SCOPE = "all_acts"
RQ1_PRIMARY_TOP_K = 10
RQ1_PRIMARY_METRIC = "f1"
RQ1_PRIMARY_OUTCOME = "gold_f1_at_10"
RQ1_PRIMARY_METRIC_LABEL = "F1@10"
RQ2_ENRICHMENT_CONDITIONS = [
    "no_enrichment",
    "parent_only",
    "xref_only",
    "combined",
]
RQ2_CONFIRMATORY_ENRICHMENT_CONTRASTS = [
    ("combined", "no_enrichment"),
    ("parent_only", "no_enrichment"),
    ("combined", "parent_only"),
]
# Contrasts isolated by the Stage 1c xref power gate.
RQ2_XREF_DEPENDENT_CONFIRMATORY_CONTRASTS = [
    ("combined", "parent_only"),
]
RQ2_EXPLORATORY_XREF_CONTRASTS = [
    ("xref_only", "no_enrichment"),
    ("combined", "xref_only"),
]
RQ2_VOLUME_CONTRAST = ("combined", "volume_matched")
XREF_GATE_ALLOWED_DECISION = "xref_contrasts_confirmatory_allowed"
CONFIRMATORY_QUERY_COUNT = 72
# MiniLM EDU+AMR top-10 anchors from Sovrano et al. (2025), Table 7.
PUBLISHED_DISCOLQA_BASELINE_BY_SCOPE = {
    "all_acts": {
        "system": "DiscoLQA (MiniLM EDU+AMR, all norms, published)",
        "config_id": "published_minilm_edu_amr_all_norms",
        "published_scope_label": "all-norm",
        "precision": 0.688,
        "recall": "",
        "f1": 0.380,
        "ndcg": 0.459,
        "mrr": 0.713,
    },
    "target_acts": {
        "system": "DiscoLQA (MiniLM EDU+AMR, target norms, published)",
        "config_id": "published_minilm_edu_amr_target_norms",
        "published_scope_label": "target-norm",
        "precision": 0.726,
        "recall": "",
        "f1": 0.413,
        "ndcg": 0.506,
        "mrr": 0.755,
    },
}
EXTERNAL_BASELINE_SCOPES = ("all_acts", "target_acts")
# Q4PIL knowledge-graph top-5 anchor from Sovrano et al. (2021).
PUBLISHED_Q4PIL_BASELINE = {
    "system": "Sovrano et al. (2021) KG baseline (Q4PIL, published)",
    "config_id": "published_q4pil_kg_baseline",
    "precision": 0.4517,
    "recall": 0.3758,
    "f1": 0.3805,
}
Q4PIL_SEARCH_SCOPE = "pil_acts"
Q4PIL_TOP_K = 5
CONDITION_ORDER = [
    "no_enrichment",
    "parent_only",
    "xref_only",
    "combined",
    "volume_matched",
]
COLORS = {
    "faithfulness": "#2563eb",
    "correctness": "#16a34a",
    "context": "#0f766e",
    "cost": "#c2410c",
    "latency": "#7c3aed",
    "combined": "#2563eb",
    "no_enrichment": "#64748b",
    "parent_only": "#16a34a",
    "volume_matched": "#c2410c",
    "xref_only": "#9333ea",
}


def write_stage8_outputs(
    *,
    config_path: Path | None = None,
    retrieval_metrics_path: Path,
    retrieval_results_path: Path,
    chunk_characteristics_path: Path,
    chunks_dir: Path,
    context_coverage_metrics_path: Path,
    context_assembly_records_path: Path,
    legal_units_path: Path,
    generation_metrics_path: Path,
    generated_answer_evaluation_path: Path,
    output_dir: Path,
    xref_eligibility_report_path: Path | None = None,
) -> dict[str, Any]:
    config = read_yaml(config_path) if config_path is not None else {}
    xref_gate_triggered = resolve_xref_gate_triggered(xref_eligibility_report_path)
    retrieval_metrics = read_json(retrieval_metrics_path)
    retrieval_results = read_json(retrieval_results_path)
    chunk_characteristics = read_json(chunk_characteristics_path)
    context_coverage_metrics = read_json(context_coverage_metrics_path)
    context_records = read_json(context_assembly_records_path)
    legal_units = read_json(legal_units_path)
    generation_metrics = read_json(generation_metrics_path)
    answer_evaluations = read_json(generated_answer_evaluation_path)

    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    retrieval_summary_rows = retrieval_summary_table(retrieval_metrics)
    retrieval_specificity_rows = retrieval_by_group_table(
        retrieval_metrics,
        group_key="by_specificity",
        group_column="specificity",
    )
    retrieval_act_rows = retrieval_by_group_table(
        retrieval_metrics,
        group_key="by_target_document",
        group_column="target_document",
    )
    retrieval_query_rows = retrieval_query_table(retrieval_results)
    rq1_confirmatory_config_ids = rq1_confirmatory_config_ids_from_config(config)
    paired_difference_rows_by_scope = {
        scope: paired_retrieval_differences(
            retrieval_results,
            hierarchy_config=rq1_confirmatory_config_ids[2],
            baseline_configs=tuple(rq1_confirmatory_config_ids[:2]),
            search_scope=scope,
        )
        for scope in ("all_acts", "target_acts")
    }
    chunk_sensitivity_rows = chunk_size_sensitivity_table(
        retrieval_metrics=retrieval_metrics,
        chunk_characteristics=chunk_characteristics,
        chunks_dir=chunks_dir,
    )
    answer_quality_rows = answer_quality_table(generation_metrics, answer_evaluations)
    abstention_rows = abstention_table(generation_metrics)
    context_coverage_rows = context_coverage_table(context_coverage_metrics)
    gold_text_coverage_rows = gold_text_coverage_table(
        context_records, legal_units, answer_evaluations
    )
    gold_text_coverage_band_rows = gold_text_coverage_band_table(
        context_records, legal_units, answer_evaluations
    )
    efficiency_rows = efficiency_table(
        generation_metrics,
        retrieval_results=retrieval_results,
        context_records=context_records,
    )
    zero_case_rows = zero_correctness_cases(answer_evaluations)
    provider_audit_rows = provider_audit_table(answer_evaluations)
    anti_gaming_tolerance = (
        config.get("analysis", {})
        .get("anti_gaming", {})
        .get("max_unjustified_abstention_increase", 0.10)
    )
    anti_gaming_rows = anti_gaming_gate_table(
        answer_evaluations,
        contrasts=RQ2_CONFIRMATORY_ENRICHMENT_CONTRASTS,
        max_unjustified_abstention_increase=anti_gaming_tolerance,
    )
    external_baseline_rows = external_baseline_table(
        retrieval_metrics,
        config_ids=rq1_confirmatory_config_ids,
    )
    external_modified_ir_rows = external_baseline_modified_ir_table(
        retrieval_results,
        config_ids=rq1_confirmatory_config_ids,
    )
    external_q4pil_rows = external_baseline_q4pil_table(
        retrieval_results,
        config_ids=rq1_confirmatory_config_ids,
    )
    bootstrap_settings = bootstrap_settings_from_config(config)
    rq1_inferential_rows = rq1_inferential_table(
        retrieval_query_rows,
        config_ids=rq1_confirmatory_config_ids,
        bootstrap_settings=bootstrap_settings,
        expected_query_count=CONFIRMATORY_QUERY_COUNT,
    )
    rq2_inferential_rows = rq2_inferential_table(
        answer_evaluations,
        bootstrap_settings=bootstrap_settings,
        xref_gate_triggered=xref_gate_triggered,
        expected_query_count=CONFIRMATORY_QUERY_COUNT,
    )

    write_csv(tables_dir / "retrieval_summary.csv", retrieval_summary_rows)
    write_csv(tables_dir / "retrieval_by_specificity.csv", retrieval_specificity_rows)
    write_csv(tables_dir / "retrieval_by_legal_act.csv", retrieval_act_rows)
    write_csv(tables_dir / "retrieval_query_metrics.csv", retrieval_query_rows)
    write_csv(
        tables_dir / "paired_retrieval_differences_all_acts.csv",
        paired_difference_rows_by_scope["all_acts"],
    )
    write_csv(
        tables_dir / "paired_retrieval_differences_target_acts.csv",
        paired_difference_rows_by_scope["target_acts"],
    )
    write_csv(tables_dir / "chunk_size_sensitivity.csv", chunk_sensitivity_rows)
    write_csv(tables_dir / "answer_quality_by_condition.csv", answer_quality_rows)
    write_csv(
        tables_dir / "gold_text_coverage_by_condition.csv", gold_text_coverage_rows
    )
    write_csv(
        tables_dir / "gold_text_coverage_bands.csv", gold_text_coverage_band_rows
    )
    write_csv(tables_dir / "abstention_by_condition.csv", abstention_rows)
    write_csv(tables_dir / "context_coverage_by_condition.csv", context_coverage_rows)
    write_csv(tables_dir / "efficiency_by_condition.csv", efficiency_rows)
    write_csv(tables_dir / "provider_audit.csv", provider_audit_rows)
    write_csv(tables_dir / "rq2_anti_gaming_gate.csv", anti_gaming_rows)
    write_csv(tables_dir / "zero_correctness_cases.csv", zero_case_rows)
    write_csv(tables_dir / "external_baseline.csv", external_baseline_rows)
    write_csv(
        tables_dir / "external_baseline_modified_ir.csv",
        external_modified_ir_rows,
    )
    write_csv(tables_dir / "external_baseline_q4pil.csv", external_q4pil_rows)
    write_csv(tables_dir / "rq1_confirmatory_statistics.csv", rq1_inferential_rows)
    write_csv(tables_dir / "rq2_confirmatory_statistics.csv", rq2_inferential_rows)

    for scope in ("all_acts", "target_acts"):
        write_rq1_metric_chart(
            figures_dir / f"rq1_{scope}_f1_at_10.svg",
            retrieval_summary_rows,
            metric_key=RQ1_PRIMARY_METRIC,
            metric_label=RQ1_PRIMARY_METRIC_LABEL,
            search_scope=scope,
        )
        write_rq1_metric_chart(
            figures_dir / f"rq1_{scope}_recall_at_10.svg",
            retrieval_summary_rows,
            metric_key="recall",
            metric_label="Recall@10",
            search_scope=scope,
        )
        write_rq1_distribution_chart(
            figures_dir / f"rq1_{scope}_f1_at_10_distribution.svg",
            retrieval_query_rows,
            metric_key=RQ1_PRIMARY_METRIC,
            metric_label=RQ1_PRIMARY_METRIC_LABEL,
            search_scope=scope,
        )
        write_rq1_distribution_chart(
            figures_dir / f"rq1_{scope}_recall_at_10_distribution.svg",
            retrieval_query_rows,
            metric_key="recall",
            metric_label="Recall@10",
            search_scope=scope,
        )
    write_paired_difference_chart(
        figures_dir / "rq1_hierarchical_paired_differences.svg",
        paired_difference_rows_by_scope["all_acts"],
    )
    write_chunk_size_diagnostic_chart(
        figures_dir / "rq1_chunk_size_answer_unit_diagnostic.svg",
        chunk_sensitivity_rows,
    )
    write_rq2_quality_chart(
        figures_dir / "rq2_answer_quality_by_condition.svg",
        answer_quality_rows,
    )
    write_rq2_abstention_chart(
        figures_dir / "rq2_abstention_by_condition.svg",
        abstention_rows,
    )
    write_context_scatter(
        figures_dir / "rq2_context_recall_vs_tokens.svg",
        context_records,
    )
    write_rq3_cost_chart(
        figures_dir / "rq3_cost_by_condition.svg",
        efficiency_rows,
    )
    write_rq3_cost_latency_chart(
        figures_dir / "rq3_cost_latency_by_condition.svg",
        efficiency_rows,
    )

    summary = build_summary(
        retrieval_summary_rows=retrieval_summary_rows,
        answer_quality_rows=answer_quality_rows,
        context_coverage_rows=context_coverage_rows,
        efficiency_rows=efficiency_rows,
        zero_case_rows=zero_case_rows,
        rq1_inferential_rows=rq1_inferential_rows,
        rq2_inferential_rows=rq2_inferential_rows,
        rq1_confirmatory_config_ids=rq1_confirmatory_config_ids,
        anti_gaming_rows=anti_gaming_rows,
        output_dir=output_dir,
    )
    write_json(output_dir / "summary.json", summary)
    return summary


def retrieval_summary_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id, config_metrics in sorted(metrics["configs"].items()):
        for search_scope, scope_metrics in sorted(config_metrics.items()):
            for k, values in sorted(scope_metrics["macro"].items(), key=int_key):
                rows.append(
                    {
                        "config_id": config_id,
                        "search_scope": search_scope,
                        "top_k": int(k),
                        "query_count": scope_metrics["query_count"],
                        **metric_columns(values),
                    }
                )
    return rows


def retrieval_by_group_table(
    metrics: dict[str, Any],
    *,
    group_key: str,
    group_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id, config_metrics in sorted(metrics["configs"].items()):
        for search_scope, scope_metrics in sorted(config_metrics.items()):
            for group_value, group_metrics in sorted(scope_metrics[group_key].items()):
                for k, values in sorted(group_metrics["metrics"].items(), key=int_key):
                    rows.append(
                        {
                            "config_id": config_id,
                            "search_scope": search_scope,
                            group_column: group_value,
                            "top_k": int(k),
                            "query_count": group_metrics["query_count"],
                            **metric_columns(values),
                        }
                    )
    return rows


def retrieval_query_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (
            item["config_id"],
            item["search_scope"],
            item["query_id"],
        ),
    ):
        for k, values in sorted(record["metrics_by_k"].items(), key=int_key):
            rows.append(
                {
                    "config_id": record["config_id"],
                    "strategy": record["strategy"],
                    "search_scope": record["search_scope"],
                    "query_id": record["query_id"],
                    "specificity": record["specificity"],
                    "target_document_codes": ";".join(
                        record["target_document_codes"]
                    ),
                    "gold_unit_count": len(record["gold_unit_ids"]),
                    "top_k": int(k),
                    **metric_columns(values),
                }
            )
    return rows


def paired_retrieval_differences(
    records: list[dict[str, Any]],
    *,
    hierarchy_config: str = "hier_paragraph",
    baseline_configs: tuple[str, ...] = ("fs_256_50", "sem_50_256"),
    search_scope: str = "all_acts",
    top_k: str = "10",
) -> list[dict[str, Any]]:
    records_by_key = {
        (record["config_id"], record["search_scope"], record["query_id"]): record
        for record in records
    }
    rows: list[dict[str, Any]] = []
    for (config_id, scope, query_id), hierarchy_record in sorted(records_by_key.items()):
        if config_id != hierarchy_config or scope != search_scope:
            continue
        hierarchy_metrics = hierarchy_record["metrics_by_k"][top_k]
        for baseline_config in baseline_configs:
            baseline = records_by_key.get((baseline_config, search_scope, query_id))
            if baseline is None:
                continue
            baseline_metrics = baseline["metrics_by_k"][top_k]
            row = {
                "query_id": query_id,
                "baseline_config_id": baseline_config,
                "hierarchy_config_id": hierarchy_config,
                "search_scope": search_scope,
                "top_k": int(top_k),
                "specificity": hierarchy_record["specificity"],
                "target_document_codes": ";".join(
                    hierarchy_record["target_document_codes"]
                ),
            }
            for metric in ("precision", "recall", "f1", "r_precision", "mrr", "ndcg"):
                row[f"baseline_{metric}"] = baseline_metrics[metric]
                row[f"hierarchy_{metric}"] = hierarchy_metrics[metric]
                row[f"delta_{metric}"] = (
                    hierarchy_metrics[metric] - baseline_metrics[metric]
                )
            rows.append(row)
    return rows


def rq1_confirmatory_config_ids_from_config(config: dict[str, Any]) -> list[str]:
    chunking = config.get("chunking")
    if not chunking:
        return RQ1_FALLBACK_CONFIRMATORY_CONFIGS

    confirmatory_by_strategy: dict[str, list[str]] = {
        strategy: [] for strategy in RQ1_STRATEGY_ORDER
    }
    family_by_strategy = {
        "fixed_size": "fixed_size",
        "semantic": "semantic",
        "hierarchical": "hierarchical",
    }
    for strategy in RQ1_STRATEGY_ORDER:
        family = family_by_strategy[strategy]
        for chunk_config in chunking.get(family, {}).get("configs", []):
            if chunk_config.get("confirmatory", False):
                confirmatory_by_strategy[strategy].append(chunk_config["id"])

    confirmatory_config_ids: list[str] = []
    for strategy in RQ1_STRATEGY_ORDER:
        configs = confirmatory_by_strategy[strategy]
        if len(configs) != 1:
            raise ValueError(
                f"Expected exactly one confirmatory RQ1 config for {strategy}, "
                f"found {len(configs)}: {configs}"
            )
        confirmatory_config_ids.append(configs[0])
    return confirmatory_config_ids


def ordered_sensitivity_configs(metrics: dict[str, Any]) -> list[str]:
    config_ids = set(metrics["configs"])
    ordered = [
        config_id for config_id in SENSITIVITY_CONFIG_ORDER if config_id in config_ids
    ]
    return ordered + sorted(config_ids.difference(ordered))


def chunk_size_sensitivity_table(
    *,
    retrieval_metrics: dict[str, Any],
    chunk_characteristics: dict[str, Any],
    chunks_dir: Path,
    top_k: str = "10",
    search_scope: str = "target_acts",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id in ordered_sensitivity_configs(retrieval_metrics):
        chunk_stats = chunk_file_stats(discover_chunk_file(chunks_dir, config_id))
        characteristics = chunk_characteristics["configs"][config_id]
        retrieval = retrieval_metrics["configs"][config_id][search_scope]["macro"][
            top_k
        ]
        rows.append(
            {
                "config_id": config_id,
                "strategy": strategy_from_config_id(config_id),
                "chunk_count": characteristics["chunk_count"],
                "mean_chunk_tokens": chunk_stats["mean_chunk_tokens"],
                "median_chunk_tokens": characteristics["token_count"]["median"],
                "p95_chunk_tokens": characteristics["token_count"]["p95"],
                "max_chunk_tokens": characteristics["token_count"]["max"],
                "mean_answer_units_per_chunk": chunk_stats[
                    "mean_answer_units_per_chunk"
                ],
                "search_scope": search_scope,
                "top_k": int(top_k),
                **metric_columns(retrieval),
            }
        )
    return rows


def answer_quality_table(
    generation_metrics: dict[str, Any],
    answer_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    distributions = score_distributions_by_condition(answer_evaluations)
    rows: list[dict[str, Any]] = []
    for condition_id in ordered_conditions(generation_metrics["conditions"]):
        values = generation_metrics["conditions"][condition_id]
        condition_distribution = distributions.get(condition_id, {})
        rows.append(
            {
                "condition_id": condition_id,
                "query_count": values["query_count"],
                "faithfulness_mean": values["faithfulness"]["mean"],
                **distribution_columns(
                    "faithfulness_",
                    condition_distribution.get("faithfulness_score", []),
                ),
                "faithfulness_min": values["faithfulness"]["min"],
                "faithfulness_max": values["faithfulness"]["max"],
                # Keep anti-gaming outcomes beside faithfulness.
                "unfaithful_response_rate": values["faithfulness"][
                    "unfaithful_response_rate"
                ],
                "faithfulness_judge_abstention_rate": values["faithfulness"][
                    "answer_abstention_rate"
                ],
                "faithfulness_judge_unjustified_abstention_rate": values[
                    "faithfulness"
                ]["unjustified_abstention_rate"],
                "correctness_mean": values["correctness"]["mean"],
                **distribution_columns(
                    "correctness_",
                    condition_distribution.get("correctness_score", []),
                ),
                "correctness_min": values["correctness"]["min"],
                "correctness_max": values["correctness"]["max"],
                "citation_precision_mean": values["citation_coverage"][
                    "mean_precision"
                ],
                "citation_recall_mean": values["citation_coverage"]["mean_recall"],
                "citation_f1_mean": values["citation_coverage"]["mean_f1"],
            }
        )
    return rows


def abstention_table(generation_metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_id in ordered_conditions(generation_metrics["conditions"]):
        values = generation_metrics["conditions"][condition_id]
        rows.append(
            {
                "condition_id": condition_id,
                "faithfulness_judge_abstention_rate": values["faithfulness"][
                    "answer_abstention_rate"
                ],
                "faithfulness_judge_unjustified_abstention_rate": values[
                    "faithfulness"
                ]["unjustified_abstention_rate"],
                "correctness_judge_abstention_rate": values["correctness"][
                    "answer_abstention_rate"
                ],
                "correctness_judge_unjustified_abstention_rate": values[
                    "correctness"
                ]["unjustified_abstention_rate"],
            }
        )
    return rows


def context_coverage_table(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for condition_id in ordered_conditions(metrics["conditions"]):
        values = metrics["conditions"][condition_id]
        row = {
            "condition_id": condition_id,
            "query_count": values["query_count"],
            **prefix_columns("macro_", values["macro"]),
            "context_token_min": values["context_token_count"]["min"],
            "context_token_median": values["context_token_count"]["median"],
            "context_token_p95": values["context_token_count"]["p95"],
            "context_token_max": values["context_token_count"]["max"],
        }
        rows.append(row)
    return rows


def efficiency_table(
    generation_metrics: dict[str, Any],
    *,
    retrieval_results: list[dict[str, Any]] | None = None,
    context_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pipeline_latency = pipeline_latency_by_condition(
        generation_metrics=generation_metrics,
        retrieval_results=retrieval_results,
        context_records=context_records,
    )
    rows: list[dict[str, Any]] = []
    for condition_id in ordered_conditions(generation_metrics["conditions"]):
        values = generation_metrics["conditions"][condition_id]
        rows.append(
            {
                "condition_id": condition_id,
                **pipeline_latency.get(condition_id, EMPTY_PIPELINE_LATENCY_COLUMNS),
                **prefix_columns("generation_", values["generation_efficiency"]),
                **prefix_columns(
                    "faithfulness_judge_",
                    values["faithfulness_judge_efficiency"],
                ),
                **prefix_columns(
                    "correctness_judge_",
                    values["correctness_judge_efficiency"],
                ),
                **prefix_columns("total_judge_", values["judge_efficiency"]),
            }
        )
    return rows


PIPELINE_LATENCY_COLUMNS = [
    "query_embedding_mean_seconds",
    "search_mean_seconds",
    "assembly_mean_seconds",
    "end_to_end_mean_seconds",
    "end_to_end_median_seconds",
    "end_to_end_p95_seconds",
    "end_to_end_query_count",
]
EMPTY_PIPELINE_LATENCY_COLUMNS = {column: "" for column in PIPELINE_LATENCY_COLUMNS}


def pipeline_latency_by_condition(
    *,
    generation_metrics: dict[str, Any],
    retrieval_results: list[dict[str, Any]] | None,
    context_records: list[dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Combine Stage 4–6 timings by condition; return {} if any are missing."""
    generation_per_query = generation_metrics.get("generation_per_query")
    if not generation_per_query or not retrieval_results or not context_records:
        return {}
    base_config_id = context_records[0]["base_config_id"]
    base_search_scope = context_records[0]["search_scope"]
    retrieval_seconds_by_query: dict[str, dict[str, float]] = {
        record["query_id"]: {
            "embedding": record["query_embedding_seconds"],
            "search": record["search_seconds"],
        }
        for record in retrieval_results
        if record["config_id"] == base_config_id
        and record["search_scope"] == base_search_scope
        and record.get("query_embedding_seconds") is not None
        and record.get("search_seconds") is not None
    }
    assembly_seconds_by_key: dict[tuple[str, str], float] = {
        (record["condition_id"], record["query_id"]): record["assembly_seconds"]
        for record in context_records
        if record.get("assembly_seconds") is not None
    }

    columns_by_condition: dict[str, dict[str, Any]] = {}
    for condition_id, per_query in generation_per_query.items():
        embedding: list[float] = []
        search: list[float] = []
        assembly: list[float] = []
        end_to_end: list[float] = []
        for query_id, generation in per_query.items():
            retrieval = retrieval_seconds_by_query.get(query_id)
            assembly_seconds = assembly_seconds_by_key.get((condition_id, query_id))
            generation_seconds = generation.get("elapsed_seconds")
            if (
                retrieval is None
                or assembly_seconds is None
                or generation_seconds is None
            ):
                continue
            embedding.append(retrieval["embedding"])
            search.append(retrieval["search"])
            assembly.append(assembly_seconds)
            end_to_end.append(
                retrieval["embedding"]
                + retrieval["search"]
                + assembly_seconds
                + generation_seconds
            )
        if not end_to_end:
            continue
        columns_by_condition[condition_id] = {
            "query_embedding_mean_seconds": mean(embedding),
            "search_mean_seconds": mean(search),
            "assembly_mean_seconds": mean(assembly),
            "end_to_end_mean_seconds": mean(end_to_end),
            "end_to_end_median_seconds": percentile(end_to_end, 50),
            "end_to_end_p95_seconds": percentile(end_to_end, 95),
            "end_to_end_query_count": len(end_to_end),
        }
    return columns_by_condition


def anti_gaming_gate_table(
    records: list[dict[str, Any]],
    *,
    contrasts: list[tuple[str, str]],
    max_unjustified_abstention_increase: float,
) -> list[dict[str, Any]]:
    # Apply the registered RQ2 anti-gaming rule.
    faithfulness = answer_metric_values_by_condition(records, "faithfulness_score")
    correctness = answer_metric_values_by_condition(records, "correctness_score")
    unjustified = answer_metric_values_by_condition(
        records, "either_judge_unjustified_abstention"
    )

    def condition_mean(values_by_condition, condition_id):
        values = values_by_condition[condition_id]
        return sum(values.values()) / len(values)

    rows: list[dict[str, Any]] = []
    for condition_id, baseline_id in contrasts:
        if condition_id not in faithfulness or baseline_id not in faithfulness:
            continue
        faithfulness_delta = condition_mean(faithfulness, condition_id) - condition_mean(
            faithfulness, baseline_id
        )
        correctness_delta = condition_mean(correctness, condition_id) - condition_mean(
            correctness, baseline_id
        )
        unjustified_delta = condition_mean(unjustified, condition_id) - condition_mean(
            unjustified, baseline_id
        )
        faithfulness_improves = faithfulness_delta > 0
        correctness_improves = correctness_delta > 0
        abstention_within_tolerance = (
            unjustified_delta <= max_unjustified_abstention_increase
        )
        rows.append(
            {
                "rq": "RQ2",
                "contrast": f"{condition_id} - {baseline_id}",
                "faithfulness_mean_delta": faithfulness_delta,
                "correctness_mean_delta": correctness_delta,
                "either_judge_unjustified_abstention_rate_delta": unjustified_delta,
                "max_unjustified_abstention_increase": (
                    max_unjustified_abstention_increase
                ),
                "faithfulness_improves": faithfulness_improves,
                "correctness_improves": correctness_improves,
                "either_judge_unjustified_abstention_within_tolerance": (
                    abstention_within_tolerance
                ),
                "interpretation_gate_passed": (
                    faithfulness_improves
                    and correctness_improves
                    and abstention_within_tolerance
                ),
            }
        )
    return rows


def provider_audit_table(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        for role, key in (
            ("generation", "generation_provider"),
            ("faithfulness_judge", "faithfulness_judge_provider"),
            ("correctness_judge", "correctness_judge_provider"),
        ):
            counts[(record["condition_id"], role, str(record.get(key)))] += 1
    return [
        {
            "condition_id": condition_id,
            "role": role,
            "provider": provider,
            "call_count": count,
        }
        for (condition_id, role, provider), count in sorted(counts.items())
    ]


def zero_correctness_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record["correctness_score"] != 0 and record["faithfulness_score"] != 0:
            continue
        rows.append(
            {
                "condition_id": record["condition_id"],
                "query_id": record["query_id"],
                "question": record["question"],
                "faithfulness_score": record["faithfulness_score"],
                "correctness_score": record["correctness_score"],
                "context_gold_recall": record["context_gold_recall"],
                "answer_abstains": record["answer_abstains"],
                "either_judge_unjustified_abstention": record[
                    "either_judge_unjustified_abstention"
                ],
                "missing_gold_citation_unit_ids": ";".join(
                    record["missing_gold_citation_unit_ids"]
                ),
                "extra_citation_unit_ids": ";".join(
                    record["extra_citation_unit_ids"]
                ),
            }
        )
    return sorted(rows, key=lambda row: (row["query_id"], row["condition_id"]))


def external_baseline_table(
    retrieval_metrics: dict[str, Any],
    *,
    config_ids: list[str] | None = None,
    search_scopes: tuple[str, ...] = EXTERNAL_BASELINE_SCOPES,
    top_k: str = "10",
) -> list[dict[str, Any]]:
    """Report standard IR metrics; published rows are for orientation only."""
    rows: list[dict[str, Any]] = []
    if config_ids is None:
        config_ids = RQ1_FALLBACK_CONFIRMATORY_CONFIGS
    for search_scope in search_scopes:
        anchor = PUBLISHED_DISCOLQA_BASELINE_BY_SCOPE[search_scope]
        for config_id in config_ids:
            metrics = retrieval_metrics["configs"][config_id][search_scope][
                "macro"
            ][top_k]
            rows.append(
                {
                    "system": config_label(config_id),
                    "config_id": config_id,
                    "source": "this_experiment",
                    "search_scope": search_scope,
                    "top_k": int(top_k),
                    "metric_basis": "standard_ir_on_pinned_dataset",
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "ndcg": metrics["ndcg"],
                    "mrr": metrics["mrr"],
                    "note": "standard IR metrics computed on pinned 246-answer Q4EU gold set",
                }
            )
        rows.append(
            {
                "system": anchor["system"],
                "config_id": anchor["config_id"],
                "source": "paper",
                "search_scope": search_scope,
                "top_k": int(top_k),
                "metric_basis": "paper_reported_released_script_convention",
                "precision": anchor["precision"],
                "recall": anchor["recall"],
                "f1": anchor["f1"],
                "ndcg": anchor["ndcg"],
                "mrr": anchor["mrr"],
                "note": (
                    f"published Table 7 top10 {anchor['published_scope_label']} "
                    "MiniLM EDU+AMR. NOT definition-comparable to the standard-IR "
                    "study rows above (published values follow DiscoLQA's released "
                    "-script convention); see external_baseline_modified_ir.csv for "
                    "the definition-matched comparison"
                ),
            }
        )
    return rows


def external_baseline_modified_ir_table(
    retrieval_results: list[dict[str, Any]],
    *,
    config_ids: list[str] | None = None,
    search_scopes: tuple[str, ...] = EXTERNAL_BASELINE_SCOPES,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Compare systems using DiscoLQA's released metric convention."""
    rows: list[dict[str, Any]] = []
    if config_ids is None:
        config_ids = RQ1_FALLBACK_CONFIRMATORY_CONFIGS
    for search_scope in search_scopes:
        anchor = PUBLISHED_DISCOLQA_BASELINE_BY_SCOPE[search_scope]
        for config_id in config_ids:
            records = [
                record
                for record in retrieval_results
                if record["config_id"] == config_id
                and record["search_scope"] == search_scope
            ]
            metrics = average_metric_dict(
                [
                    released_script_modified_ir_metrics(record=record, top_k=top_k)
                    for record in records
                ]
            )
            rows.append(
                {
                    "system": config_label(config_id),
                    "config_id": config_id,
                    "source": "this_experiment",
                    "search_scope": search_scope,
                    "top_k": top_k,
                    "metric_basis": "released_evaluate_py_modified_ir",
                    "modified_precision": metrics["precision"],
                    "modified_recall": metrics["recall"],
                    "modified_f1": metrics["f1"],
                    "modified_ndcg": metrics["ndcg"],
                    "modified_mrr": metrics["mrr"],
                    "note": "primary external comparison: this study's retrieval re-scored with DiscoLQA's released evaluate.py convention on the pinned 246-answer gold set, so study and DiscoLQA rows share one ruler",
                }
            )
        rows.append(
            {
                "system": anchor["system"],
                "config_id": anchor["config_id"],
                "source": "paper",
                "search_scope": search_scope,
                "top_k": top_k,
                "metric_basis": "paper_reported_values",
                "modified_precision": anchor["precision"],
                "modified_recall": anchor["recall"],
                "modified_f1": anchor["f1"],
                "modified_ndcg": anchor["ndcg"],
                "modified_mrr": anchor["mrr"],
                "note": (
                    f"published Table 7 top10 {anchor['published_scope_label']} "
                    "MiniLM EDU+AMR; released-script convention; approximate "
                    "contextual comparison (paper reports 225 answers, pinned "
                    "repo has 246)"
                ),
            }
        )
    return rows


def external_baseline_q4pil_table(
    retrieval_results: list[dict[str, Any]],
    *,
    config_ids: list[str] | None = None,
    search_scope: str = Q4PIL_SEARCH_SCOPE,
    top_k: int = Q4PIL_TOP_K,
) -> list[dict[str, Any]]:
    """Compare the Q4PIL subset with Sovrano et al. (2021) at top-5."""
    rows: list[dict[str, Any]] = []
    if config_ids is None:
        config_ids = RQ1_FALLBACK_CONFIRMATORY_CONFIGS
    for config_id in config_ids:
        records = [
            record
            for record in retrieval_results
            if record["config_id"] == config_id
            and record["search_scope"] == search_scope
        ]
        if not records:
            continue
        metrics = average_metric_dict(
            [
                released_script_modified_ir_metrics(record=record, top_k=top_k)
                for record in records
            ]
        )
        rows.append(
            {
                "system": config_label(config_id),
                "config_id": config_id,
                "source": "this_experiment",
                "search_scope": search_scope,
                "top_k": top_k,
                "query_count": len(records),
                "metric_basis": "released_evaluate_py_modified_ir",
                "modified_precision": metrics["precision"],
                "modified_recall": metrics["recall"],
                "modified_f1": metrics["f1"],
                "note": (
                    "Q4PIL subset: retrieval restricted to the three PIL acts "
                    "(B/RI/RII), scored with DiscoLQA's released evaluate.py "
                    "convention on the pinned gold set (83 raw expected "
                    "answers vs 65 in the 2021 paper)"
                ),
            }
        )
    rows.append(
        {
            "system": PUBLISHED_Q4PIL_BASELINE["system"],
            "config_id": PUBLISHED_Q4PIL_BASELINE["config_id"],
            "source": "paper",
            "search_scope": search_scope,
            "top_k": top_k,
            "query_count": 17,
            "metric_basis": "paper_reported_values",
            "modified_precision": PUBLISHED_Q4PIL_BASELINE["precision"],
            "modified_recall": PUBLISHED_Q4PIL_BASELINE["recall"],
            "modified_f1": PUBLISHED_Q4PIL_BASELINE["f1"],
            "note": (
                "published aggregate top-5 values (P 45.17%, R 37.58%, "
                "F1 38.05%) for the Sovrano et al. (2020) knowledge-graph "
                "baseline on Q4PIL; modified metric convention; approximate "
                "contextual comparison (paper gold set has 65 expected "
                "answers, pinned snapshot 83)"
            ),
        }
    )
    return rows


def released_script_modified_ir_metrics(
    *,
    record: dict[str, Any],
    top_k: int,
) -> dict[str, float]:
    gold = set(record["gold_unit_ids"])
    given = flattened_answer_units(record["retrieved_chunks"])[:top_k]
    relevant_unique = len(set(given).intersection(gold))
    relevant_flags = [unit_id in gold for unit_id in given]
    relevant_total = sum(relevant_flags)
    precision_denominator = min(top_k, len(gold))
    recall_denominator = min(top_k, len(given))
    precision = (
        relevant_unique / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = relevant_total / recall_denominator if recall_denominator else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ndcg = modified_ndcg_from_flags(relevant_flags)
    mrr = modified_reciprocal_rank(relevant_flags)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ndcg": ndcg,
        "mrr": mrr,
    }


def flattened_answer_units(chunks: list[dict[str, Any]]) -> list[str]:
    units: list[str] = []
    for chunk in chunks:
        units.extend(chunk["answer_unit_ids"])
    return units


def config_label(config_id: str) -> str:
    if config_id.startswith("fs_"):
        parts = config_id.split("_")
        if len(parts) >= 3:
            return f"Fixed size, {parts[1]} tokens"
    if config_id.startswith("sem_"):
        parts = config_id.split("_")
        if len(parts) >= 3:
            return f"Semantic, p{parts[1]} max {parts[2]} tokens"
    if config_id == "hier_article":
        return "Hierarchical, article/recital leaf"
    if config_id == "hier_paragraph":
        return "Hierarchical, paragraph/recital leaf"
    if config_id == "hier_subparagraph":
        return "Hierarchical, subparagraph/recital leaf"
    if config_id == "hier_paragraph_contextualized":
        return "Hierarchical, paragraph leaf with structural embedding context"
    return config_id


def modified_ndcg_from_flags(flags: list[bool]) -> float:
    dcg = discounted_gain(flags)
    ideal = discounted_gain(sorted(flags, reverse=True))
    return dcg / ideal if ideal else 0.0


def discounted_gain(flags: list[bool]) -> float:
    return sum(
        (1.0 if flag else 0.0) / math.log2(index + 1)
        for index, flag in enumerate(flags, start=1)
    )


def modified_reciprocal_rank(flags: list[bool]) -> float:
    for index, flag in enumerate(flags, start=1):
        if flag:
            return 1 / index
    return 0.0


def rq1_inferential_table(
    rows: list[dict[str, Any]],
    *,
    config_ids: list[str] | None = None,
    bootstrap_settings: dict[str, Any] | None = None,
    expected_query_count: int | None = None,
) -> list[dict[str, Any]]:
    if config_ids is None:
        config_ids = RQ1_FALLBACK_CONFIRMATORY_CONFIGS
    values_by_config = metric_values_by_key(
        [
            row
            for row in rows
            if row["search_scope"] == RQ1_CONFIRMATORY_SCOPE
            and int(row["top_k"]) == RQ1_PRIMARY_TOP_K
            and row["config_id"] in config_ids
        ],
        item_key="config_id",
        outcome_key=RQ1_PRIMARY_METRIC,
    )
    assert_complete_paired_queries(
        values_by_config,
        config_ids,
        table_name="RQ1 inferential table",
        expected_query_count=expected_query_count,
    )
    aligned = aligned_values(values_by_config, config_ids)
    fixed_config, semantic_config, hierarchical_config = config_ids
    output = [
        friedman_row(
            rq="RQ1",
            outcome=RQ1_PRIMARY_OUTCOME,
            family="confirmatory_chunking",
            labels=config_ids,
            aligned=aligned,
        )
    ]
    output.extend(
        pairwise_rows(
            rq="RQ1",
            outcome=RQ1_PRIMARY_OUTCOME,
            family="confirmatory_chunking",
            values_by_key=values_by_config,
            contrasts=[
                (hierarchical_config, fixed_config),
                (hierarchical_config, semantic_config),
                (semantic_config, fixed_config),
            ],
            holm_family="rq1_confirmatory",
            bootstrap_settings=bootstrap_settings,
        )
    )
    return output


def resolve_xref_gate_triggered(report_path: Path | None) -> bool:
    # Only the explicit Stage 1c-ii decision opens the confirmatory gate.
    if report_path is None or not report_path.exists():
        return True
    report = read_json(report_path)
    return report.get("xref_power_decision") != XREF_GATE_ALLOWED_DECISION


def rq2_inferential_table(
    records: list[dict[str, Any]],
    *,
    bootstrap_settings: dict[str, Any] | None = None,
    xref_gate_triggered: bool = True,
    expected_query_count: int | None = None,
) -> list[dict[str, Any]]:
    if xref_gate_triggered:
        confirmatory_contrasts = [
            contrast
            for contrast in RQ2_CONFIRMATORY_ENRICHMENT_CONTRASTS
            if contrast not in RQ2_XREF_DEPENDENT_CONFIRMATORY_CONTRASTS
        ]
        exploratory_contrasts = (
            RQ2_XREF_DEPENDENT_CONFIRMATORY_CONTRASTS + RQ2_EXPLORATORY_XREF_CONTRASTS
        )
    else:
        confirmatory_contrasts = RQ2_CONFIRMATORY_ENRICHMENT_CONTRASTS
        exploratory_contrasts = RQ2_EXPLORATORY_XREF_CONTRASTS

    output: list[dict[str, Any]] = []
    for outcome in ("faithfulness_score",):
        values_by_condition = answer_metric_values_by_condition(records, outcome)
        assert_complete_paired_queries(
            values_by_condition,
            RQ2_ENRICHMENT_CONDITIONS + [RQ2_VOLUME_CONTRAST[1]],
            table_name=f"RQ2 inferential table ({outcome})",
            expected_query_count=expected_query_count,
        )
        aligned = aligned_values(values_by_condition, RQ2_ENRICHMENT_CONDITIONS)
        output.append(
            friedman_row(
                rq="RQ2",
                outcome=outcome,
                family="enrichment_family",
                labels=RQ2_ENRICHMENT_CONDITIONS,
                aligned=aligned,
            )
        )
        output.extend(
            pairwise_rows(
                rq="RQ2",
                outcome=outcome,
                family="enrichment_confirmatory",
                values_by_key=values_by_condition,
                contrasts=confirmatory_contrasts,
                holm_family=f"rq2_{outcome}_enrichment_confirmatory",
                bootstrap_settings=bootstrap_settings,
            )
        )
        output.extend(
            pairwise_rows(
                rq="RQ2",
                outcome=outcome,
                family="xref_exploratory",
                values_by_key=values_by_condition,
                contrasts=exploratory_contrasts,
                holm_family="not_applicable",
                bootstrap_settings=bootstrap_settings,
                notes="exploratory xref contrast; outside confirmatory Holm family",
            )
        )
        output.extend(
            pairwise_rows(
                rq="RQ2",
                outcome=outcome,
                family="volume_control",
                values_by_key=values_by_condition,
                contrasts=[RQ2_VOLUME_CONTRAST],
                holm_family="not_applicable",
                bootstrap_settings=bootstrap_settings,
                notes="volume control reported outside enrichment Holm family",
            )
        )
    return output


def friedman_row(
    *,
    rq: str,
    outcome: str,
    family: str,
    labels: list[str],
    aligned: list[list[float]],
) -> dict[str, Any]:
    # Descriptive only; confirmatory inference uses the Holm family.
    result = friedman_test(aligned)
    return {
        "rq": rq,
        "outcome": outcome,
        "family": family,
        "comparison": " vs ".join(labels),
        "test": result["test"],
        "method": result["method"],
        "n": len(aligned),
        "effective_n": len(aligned),
        "statistic": result["statistic"],
        "p_value": result["p_value"],
        "p_value_adjusted_holm": "",
        "kendalls_w": result["kendalls_w"],
        "rank_biserial": "",
        "mean_difference": "",
        "median_difference": "",
        "ci_lower": "",
        "ci_upper": "",
        "notes": "descriptive omnibus summary; not a gate for the contrast family",
    }


def pairwise_rows(
    *,
    rq: str,
    outcome: str,
    family: str,
    values_by_key: dict[str, dict[str, float]],
    contrasts: list[tuple[str, str]],
    holm_family: str,
    bootstrap_settings: dict[str, Any] | None = None,
    notes: str | None = None,
) -> list[dict[str, Any]]:
    settings = bootstrap_settings or bootstrap_settings_from_config({})
    rows: list[dict[str, Any]] = []
    p_values: list[float] = []
    for left, right in contrasts:
        paired = paired_values(values_by_key[left], values_by_key[right])
        differences = [
            round(left_value - right_value, 12)
            for left_value, right_value in paired
        ]
        test = wilcoxon_signed_rank(differences)
        ci_lower, ci_upper = bootstrap_ci(
            differences,
            iterations=settings["iterations"],
            confidence=settings["confidence"],
            random_seed=settings["random_seed"],
        )
        p_values.append(test["p_value"])
        rows.append(
            {
                "rq": rq,
                "outcome": outcome,
                "family": family,
                "comparison": f"{left} - {right}",
                "test": test["test"],
                "method": test["method"],
                "n": len(paired),
                "effective_n": test["effective_n"],
                "statistic": test["statistic"],
                "p_value": test["p_value"],
                "p_value_adjusted_holm": "",
                "kendalls_w": "",
                "rank_biserial": test["rank_biserial"],
                "mean_difference": mean(differences),
                "median_difference": median(differences),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "notes": notes
                if notes is not None
                else "Holm-adjusted within confirmatory family",
            }
        )
    adjusted = holm_adjusted_p_values(p_values)
    for row, p_value in zip(rows, adjusted):
        row["p_value_adjusted_holm"] = (
            p_value if holm_family != "not_applicable" else ""
        )
    return rows


def friedman_test(aligned: list[list[float]]) -> dict[str, Any]:
    n = len(aligned)
    if n == 0:
        return {
            "test": "scipy_friedman_chi_square",
            "method": "scipy.stats.friedmanchisquare",
            "statistic": 0.0,
            "p_value": 1.0,
            "kendalls_w": 0.0,
        }
    k = len(aligned[0])
    columns = [[row[index] for row in aligned] for index in range(k)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = stats.friedmanchisquare(*columns)
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    if math.isnan(statistic) or math.isnan(p_value):
        statistic = 0.0
        p_value = 1.0
    kendalls_w = statistic / (n * (k - 1)) if n and k > 1 else 0.0
    return {
        "test": "scipy_friedman_chi_square",
        "method": "scipy.stats.friedmanchisquare",
        "statistic": statistic,
        "p_value": p_value,
        "kendalls_w": kendalls_w,
    }


def bootstrap_settings_from_config(config: dict[str, Any]) -> dict[str, Any]:
    analysis_config = config.get("analysis", {})
    return {
        "iterations": int(analysis_config.get("bootstrap_iterations", 10000)),
        "confidence": float(analysis_config.get("bootstrap_ci", 0.95)),
        "random_seed": int(config.get("random_seed", 42)),
    }


def metric_values_by_key(
    rows: list[dict[str, Any]],
    *,
    item_key: str,
    outcome_key: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        values.setdefault(row[item_key], {})[row["query_id"]] = float(row[outcome_key])
    return values


def answer_metric_values_by_condition(
    records: list[dict[str, Any]],
    outcome_key: str,
) -> dict[str, dict[str, float]]:
    values: dict[str, dict[str, float]] = {}
    for record in records:
        values.setdefault(record["condition_id"], {})[record["query_id"]] = float(
            record[outcome_key]
        )
    return values


def assert_complete_paired_queries(
    values_by_key: dict[str, dict[str, float]],
    labels: list[str],
    *,
    table_name: str,
    expected_query_count: int | None = None,
) -> None:
    # Paired inference requires identical query sets.
    query_sets = {label: set(values_by_key.get(label, {})) for label in labels}
    all_query_ids = set.union(*query_sets.values())
    problems = [
        f"{label} is missing {sorted(all_query_ids - query_ids)}"
        for label, query_ids in query_sets.items()
        if query_ids != all_query_ids
    ]
    if expected_query_count is not None and len(all_query_ids) != expected_query_count:
        problems.append(
            f"expected {expected_query_count} queries, found {len(all_query_ids)}"
        )
    if problems:
        raise ValueError(
            f"{table_name}: incomplete paired query data: " + "; ".join(problems)
        )


def aligned_values(
    values_by_key: dict[str, dict[str, float]],
    labels: list[str],
) -> list[list[float]]:
    shared_query_ids = sorted(set.intersection(*(set(values_by_key[label]) for label in labels)))
    return [[values_by_key[label][query_id] for label in labels] for query_id in shared_query_ids]


def paired_values(
    left: dict[str, float],
    right: dict[str, float],
) -> list[tuple[float, float]]:
    shared_query_ids = sorted(set(left) & set(right))
    return [(left[query_id], right[query_id]) for query_id in shared_query_ids]


def build_summary(
    *,
    retrieval_summary_rows: list[dict[str, Any]],
    answer_quality_rows: list[dict[str, Any]],
    context_coverage_rows: list[dict[str, Any]],
    efficiency_rows: list[dict[str, Any]],
    zero_case_rows: list[dict[str, Any]],
    rq1_inferential_rows: list[dict[str, Any]],
    rq2_inferential_rows: list[dict[str, Any]],
    rq1_confirmatory_config_ids: list[str],
    anti_gaming_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    target_retrieval = [
        row
        for row in retrieval_summary_rows
        if row["search_scope"] == "target_acts" and row["top_k"] == 10
    ]
    best_primary_metric = max(
        target_retrieval,
        key=lambda row: row[RQ1_PRIMARY_METRIC],
    )
    best_recall = max(target_retrieval, key=lambda row: row["recall"])
    best_faithfulness = max(
        answer_quality_rows,
        key=lambda row: row["faithfulness_mean"],
    )
    return {
        "output_dir": str(output_dir),
        "retrieval": {
            "rq1_confirmatory_config_ids": rq1_confirmatory_config_ids,
            "rq1_confirmatory_scope": RQ1_CONFIRMATORY_SCOPE,
            "rq1_confirmatory_top_k": RQ1_PRIMARY_TOP_K,
            "rq1_confirmatory_metric": RQ1_PRIMARY_METRIC,
            "best_target_acts_f1_at_10": {
                "config_id": best_primary_metric["config_id"],
                "primary_metric": RQ1_PRIMARY_METRIC,
                "f1": best_primary_metric["f1"],
                "precision": best_primary_metric["precision"],
                "recall": best_primary_metric["recall"],
                "ndcg": best_primary_metric["ndcg"],
                "mrr": best_primary_metric["mrr"],
            },
            "best_target_acts_recall_at_10": {
                "config_id": best_recall["config_id"],
                "recall": best_recall["recall"],
                "precision": best_recall["precision"],
                "f1": best_recall["f1"],
                "ndcg": best_recall["ndcg"],
                "mrr": best_recall["mrr"],
            },
            "target_acts_top_10_config_count": len(target_retrieval),
        },
        "generation": {
            "best_faithfulness_condition": {
                "condition_id": best_faithfulness["condition_id"],
                "faithfulness_mean": best_faithfulness["faithfulness_mean"],
                "correctness_mean": best_faithfulness["correctness_mean"],
                "note": "ungated best-faithfulness; see rq2_anti_gaming_gate",
            },
            "rq2_anti_gaming_gate": {
                "registered_rule": (
                    "a contrast is interpreted as better only if faithfulness "
                    "and correctness mean deltas are both > 0 and the "
                    "unjustified-abstention rate (an abstention flagged "
                    "unjustified by either judge) increases by at most the "
                    "registered tolerance"
                ),
                "contrasts": anti_gaming_rows,
                "gate_passed_contrasts": [
                    row["contrast"]
                    for row in anti_gaming_rows
                    if row["interpretation_gate_passed"]
                ],
            },
            "zero_correctness_case_count": len(zero_case_rows),
        },
        "inferential": {
            "rq1_omnibus": next(
                row
                for row in rq1_inferential_rows
                if row["test"] == "scipy_friedman_chi_square"
            ),
            "rq2_faithfulness_omnibus": next(
                row
                for row in rq2_inferential_rows
                if row["test"] == "scipy_friedman_chi_square"
                and row["outcome"] == "faithfulness_score"
            ),
        },
        "context_coverage": {
            row["condition_id"]: {
                "context_gold_recall": row["macro_context_gold_recall"],
                "context_token_median": row["context_token_median"],
            }
            for row in context_coverage_rows
        },
        "efficiency": {
            row["condition_id"]: {
                "generation_total_cost_usd": row["generation_total_cost_usd"],
                "total_judge_cost_usd": row["total_judge_total_cost_usd"],
                "generation_total_elapsed_seconds": row[
                    "generation_total_elapsed_seconds"
                ],
                "total_judge_elapsed_seconds": row[
                    "total_judge_total_elapsed_seconds"
                ],
            }
            for row in efficiency_rows
        },
    }


def scope_title_label(search_scope: str) -> str:
    return "All-Act" if search_scope == "all_acts" else "Target-Act"


def write_rq1_metric_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    metric_label: str,
    search_scope: str = "all_acts",
) -> None:
    chart_rows = [
        {
            "label": row["config_id"],
            "value": row[metric_key],
        }
        for row in rows
        if row["search_scope"] == search_scope and row["top_k"] == 10
    ]
    write_bar_svg(
        path,
        title=(
            f"RQ1 {scope_title_label(search_scope)} {metric_label} "
            "by Chunking Configuration"
        ),
        rows=sorted(chart_rows, key=lambda row: row["value"], reverse=True),
        value_key="value",
        color="#2563eb",
        value_max=1.0,
    )


def write_rq1_distribution_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric_key: str,
    metric_label: str,
    search_scope: str = "all_acts",
) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row["search_scope"] == search_scope and row["top_k"] == 10:
            grouped.setdefault(row["config_id"], []).append(float(row[metric_key]))
    labels = sorted(grouped)
    width = max(980, 110 + 96 * len(labels))
    height = 560
    margin_left, margin_right, margin_top, margin_bottom = 74, 28, 60, 126
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_step = plot_width / len(labels)
    parts: list[str] = []
    for index, label in enumerate(labels):
        values = sorted(grouped[label])
        x = margin_left + index * group_step + group_step / 2
        q1 = quantile(values, 0.25)
        median = quantile(values, 0.5)
        q3 = quantile(values, 0.75)
        y_min = value_to_y(values[0], 0.0, 1.0, margin_top, plot_height)
        y_max = value_to_y(values[-1], 0.0, 1.0, margin_top, plot_height)
        y_q1 = value_to_y(q1, 0.0, 1.0, margin_top, plot_height)
        y_median = value_to_y(median, 0.0, 1.0, margin_top, plot_height)
        y_q3 = value_to_y(q3, 0.0, 1.0, margin_top, plot_height)
        box_width = min(42, group_step * 0.48)
        parts.append(
            f'<line x1="{x:.2f}" y1="{y_max:.2f}" x2="{x:.2f}" y2="{y_min:.2f}" stroke="#475569"/>'
            f'<rect x="{x - box_width/2:.2f}" y="{y_q3:.2f}" width="{box_width:.2f}" height="{y_q1 - y_q3:.2f}" fill="#bfdbfe" stroke="#1d4ed8"/>'
            f'<line x1="{x - box_width/2:.2f}" y1="{y_median:.2f}" x2="{x + box_width/2:.2f}" y2="{y_median:.2f}" stroke="#1e3a8a" stroke-width="2"/>'
        )
        for point_index, value in enumerate(values):
            jitter = ((point_index % 7) - 3) * 2.3
            y = value_to_y(value, 0.0, 1.0, margin_top, plot_height)
            parts.append(
                f'<circle cx="{x + jitter:.2f}" cy="{y:.2f}" r="2.1" fill="#2563eb" fill-opacity="0.36"/>'
            )
        parts.append(
            f'<text x="{x:.2f}" y="{height - margin_bottom + 20}" font-family="Arial" font-size="11" text-anchor="end" transform="rotate(-45 {x:.2f} {height - margin_bottom + 20})">{escape(label)}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_left}" y="34" font-family="Arial" font-size="20" font-weight="700">{escape(f"RQ1 {scope_title_label(search_scope)} {metric_label} Distribution")}</text>
<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#334155"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#334155"/>
<text x="24" y="{height/2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 24 {height/2})">{escape(metric_label)}</text>
<text x="{margin_left-10}" y="{height-margin_bottom+4}" font-family="Arial" font-size="11" text-anchor="end">0.0</text>
<text x="{margin_left-10}" y="{margin_top+4}" font-family="Arial" font-size="11" text-anchor="end">1.0</text>
{''.join(parts)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_paired_difference_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metric_key: str = RQ1_PRIMARY_METRIC,
    metric_label: str = RQ1_PRIMARY_METRIC_LABEL,
) -> None:
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(row["baseline_config_id"], []).append(
            row[f"delta_{metric_key}"]
        )
    chart_rows = [
        {"label": f"hier - {baseline}", "value": mean(values)}
        for baseline, values in sorted(grouped.items())
    ]
    write_bar_svg(
        path,
        title=f"RQ1 Paired {metric_label} Difference: Hierarchical vs Baselines",
        rows=chart_rows,
        value_key="value",
        color="#9333ea",
        value_min=min(0.0, *(row["value"] for row in chart_rows)),
        value_max=max(0.0, *(row["value"] for row in chart_rows)),
    )


def write_chunk_size_diagnostic_chart(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    width, height = 900, 560
    margin = 76
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    max_tokens = max(float(row["mean_chunk_tokens"]) for row in rows) * 1.08
    max_units = max(float(row["mean_answer_units_per_chunk"]) for row in rows) * 1.16
    strategy_colors = {
        "fixed_size": "#2563eb",
        "semantic": "#16a34a",
        "hierarchical": "#9333ea",
    }
    points: list[str] = []
    for row in rows:
        x = margin + plot_width * float(row["mean_chunk_tokens"]) / max_tokens
        y = margin + plot_height * (
            1 - float(row["mean_answer_units_per_chunk"]) / max_units
        )
        radius = 5 + 18 * float(row["recall"])
        color = strategy_colors.get(row["strategy"], "#334155")
        label = row["config_id"]
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity="0.65" stroke="#0f172a"/>'
            f'<text x="{x + radius + 5:.2f}" y="{y + 4:.2f}" font-family="Arial" font-size="12">{escape(label)}</text>'
            f'<title>{escape(label)} recall={float(row["recall"]):.3f}, mean tokens={float(row["mean_chunk_tokens"]):.1f}, mean answer units={float(row["mean_answer_units_per_chunk"]):.2f}</title>'
        )
    legend = legend_svg(
        [(label, color) for label, color in strategy_colors.items()],
        x=margin,
        y=height - 30,
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-family="Arial" font-size="20" font-weight="700">{escape("RQ1 Chunk Size and Answer-Unit Diagnostic")}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="{height-18}" font-family="Arial" font-size="13" text-anchor="middle">Mean chunk tokens</text>
<text x="24" y="{height/2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 24 {height/2})">Mean answer units per chunk</text>
<text x="{margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">0</text>
<text x="{width-margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">{max_tokens:.0f}</text>
<text x="{margin-10}" y="{height-margin+4}" font-family="Arial" font-size="11" text-anchor="end">0</text>
<text x="{margin-10}" y="{margin+4}" font-family="Arial" font-size="11" text-anchor="end">{max_units:.1f}</text>
{''.join(points)}
{legend}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_rq2_quality_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_bar_svg(
        path,
        title="RQ2 Generated-Answer Quality by Enrichment Condition",
        rows=rows,
        label_key="condition_id",
        series=[
            ("faithfulness_mean", "faithfulness", COLORS["faithfulness"]),
            ("correctness_mean", "correctness", COLORS["correctness"]),
        ],
        value_max=1.0,
    )


def write_rq2_abstention_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    write_grouped_bar_svg(
        path,
        title="RQ2 Abstention Rates by Enrichment Condition",
        rows=rows,
        label_key="condition_id",
        series=[
            (
                "faithfulness_judge_abstention_rate",
                "faithfulness-judge abstention",
                "#2563eb",
            ),
            (
                "correctness_judge_unjustified_abstention_rate",
                "correctness-judge unjustified abstention",
                "#dc2626",
            ),
        ],
        value_max=0.2,
    )


def write_context_scatter(path: Path, records: list[dict[str, Any]]) -> None:
    width, height = 980, 560
    margin = 72
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    max_tokens = max(record["context_token_count"] for record in records)
    points: list[str] = []
    for record in records:
        x = margin + plot_width * record["context_token_count"] / max_tokens
        y = margin + plot_height * (1 - record["context_gold_recall"])
        color = COLORS.get(record["condition_id"], "#334155")
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}" '
            f'fill-opacity="0.62"><title>{escape(record["condition_id"])} '
            f'{escape(record["query_id"])} recall={record["context_gold_recall"]:.3f} '
            f'tokens={record["context_token_count"]}</title></circle>'
        )
    legend = legend_svg(
        [(condition, COLORS.get(condition, "#334155")) for condition in CONDITION_ORDER],
        x=margin,
        y=height - 36,
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-family="Arial" font-size="20" font-weight="700">{escape("RQ2 Context Gold Recall vs Context Tokens")}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="{height-18}" font-family="Arial" font-size="13" text-anchor="middle">Context tokens</text>
<text x="22" y="{height/2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 22 {height/2})">Context gold recall</text>
<text x="{margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">0</text>
<text x="{width-margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">{max_tokens}</text>
<text x="{margin-10}" y="{height-margin+4}" font-family="Arial" font-size="11" text-anchor="end">0.0</text>
<text x="{margin-10}" y="{margin+4}" font-family="Arial" font-size="11" text-anchor="end">1.0</text>
{''.join(points)}
{legend}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_rq3_cost_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    chart_rows = [
        {
            "condition_id": row["condition_id"],
            "generation_total_cost_usd": row["generation_total_cost_usd"],
            "total_judge_total_cost_usd": row["total_judge_total_cost_usd"],
        }
        for row in rows
    ]
    write_grouped_bar_svg(
        path,
        title="RQ3 API Cost by Condition",
        rows=chart_rows,
        label_key="condition_id",
        series=[
            ("generation_total_cost_usd", "generation cost", COLORS["cost"]),
            ("total_judge_total_cost_usd", "judge cost", COLORS["latency"]),
        ],
        value_max=max(
            max(row["generation_total_cost_usd"], row["total_judge_total_cost_usd"])
            for row in chart_rows
        ),
        y_label="USD",
    )


def write_rq3_cost_latency_chart(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 860, 560
    margin = 76
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin
    chart_rows = [
        {
            "condition_id": row["condition_id"],
            "total_cost_usd": float(row["generation_total_cost_usd"])
            + float(row["total_judge_total_cost_usd"]),
            "total_elapsed_seconds": float(row["generation_total_elapsed_seconds"])
            + float(row["total_judge_total_elapsed_seconds"]),
        }
        for row in rows
    ]
    min_cost = min(row["total_cost_usd"] for row in chart_rows) * 0.96
    max_cost = max(row["total_cost_usd"] for row in chart_rows) * 1.04
    min_seconds = min(row["total_elapsed_seconds"] for row in chart_rows) * 0.96
    max_seconds = max(row["total_elapsed_seconds"] for row in chart_rows) * 1.04
    points: list[str] = []
    for row in chart_rows:
        x = margin + plot_width * (
            (row["total_cost_usd"] - min_cost) / (max_cost - min_cost)
        )
        y = margin + plot_height * (
            1
            - (row["total_elapsed_seconds"] - min_seconds)
            / (max_seconds - min_seconds)
        )
        color = COLORS.get(row["condition_id"], "#334155")
        points.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" fill="{color}" fill-opacity="0.72" stroke="#0f172a"/>'
            f'<text x="{x + 12:.2f}" y="{y + 4:.2f}" font-family="Arial" font-size="12">{escape(row["condition_id"])}</text>'
            f'<title>{escape(row["condition_id"])} cost=${row["total_cost_usd"]:.3f}, elapsed={row["total_elapsed_seconds"]:.1f}s</title>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin}" y="34" font-family="Arial" font-size="20" font-weight="700">{escape("RQ3 Total API Cost vs Latency by Condition")}</text>
<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#334155"/>
<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#334155"/>
<text x="{width/2}" y="{height-18}" font-family="Arial" font-size="13" text-anchor="middle">Total API cost (USD)</text>
<text x="24" y="{height/2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 24 {height/2})">Total elapsed seconds</text>
<text x="{margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">${min_cost:.2f}</text>
<text x="{width-margin}" y="{height-margin+20}" font-family="Arial" font-size="11" text-anchor="middle">${max_cost:.2f}</text>
<text x="{margin-10}" y="{height-margin+4}" font-family="Arial" font-size="11" text-anchor="end">{min_seconds:.0f}</text>
<text x="{margin-10}" y="{margin+4}" font-family="Arial" font-size="11" text-anchor="end">{max_seconds:.0f}</text>
{''.join(points)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_bar_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    value_key: str,
    color: str,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> None:
    width = max(760, 100 + 72 * len(rows))
    height = 500
    margin_left, margin_right, margin_top, margin_bottom = 72, 28, 58, 118
    plot_height = height - margin_top - margin_bottom
    zero = value_to_y(0.0, value_min, value_max, margin_top, plot_height)
    bar_width = max(22, (width - margin_left - margin_right) / len(rows) * 0.62)
    step = (width - margin_left - margin_right) / len(rows)
    bars: list[str] = []
    for index, row in enumerate(rows):
        value = float(row[value_key])
        x = margin_left + index * step + (step - bar_width) / 2
        y = value_to_y(max(value, 0.0), value_min, value_max, margin_top, plot_height)
        bar_height = abs(zero - y)
        bar_y = min(y, zero)
        bars.append(
            f'<rect x="{x:.2f}" y="{bar_y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
            f'<text x="{x + bar_width/2:.2f}" y="{bar_y - 6:.2f}" font-family="Arial" font-size="11" text-anchor="middle">{value:.3f}</text>'
            f'<text x="{x + bar_width/2:.2f}" y="{height - margin_bottom + 20}" font-family="Arial" font-size="11" text-anchor="end" transform="rotate(-45 {x + bar_width/2:.2f} {height - margin_bottom + 20})">{escape(str(row["label"]))}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_left}" y="34" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>
<line x1="{margin_left}" y1="{zero:.2f}" x2="{width-margin_right}" y2="{zero:.2f}" stroke="#334155"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#334155"/>
{''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    rows: list[dict[str, Any]],
    label_key: str,
    series: list[tuple[str, str, str]],
    value_max: float,
    y_label: str = "score",
) -> None:
    width = max(820, 130 + 108 * len(rows))
    height = 540
    margin_left, margin_right, margin_top, margin_bottom = 74, 28, 68, 124
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    group_step = plot_width / len(rows)
    bar_width = min(22, group_step / (len(series) + 1))
    bars: list[str] = []
    for index, row in enumerate(rows):
        group_x = margin_left + index * group_step + group_step / 2
        for series_index, (key, _, color) in enumerate(series):
            value = float(row[key] or 0.0)
            x = group_x + (series_index - (len(series) - 1) / 2) * (bar_width + 4)
            bar_height = plot_height * value / value_max if value_max else 0.0
            y = margin_top + plot_height - bar_height
            bars.append(
                f'<rect x="{x - bar_width/2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
                f'<title>{escape(str(row[label_key]))} {escape(key)}={value:.3f}</title>'
            )
        bars.append(
            f'<text x="{group_x:.2f}" y="{height - margin_bottom + 20}" font-family="Arial" font-size="11" text-anchor="end" transform="rotate(-45 {group_x:.2f} {height - margin_bottom + 20})">{escape(str(row[label_key]))}</text>'
        )
    legend = legend_svg([(label, color) for _, label, color in series], margin_left, 44)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{margin_left}" y="28" font-family="Arial" font-size="20" font-weight="700">{escape(title)}</text>
{legend}
<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#334155"/>
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#334155"/>
<text x="24" y="{height/2}" font-family="Arial" font-size="13" text-anchor="middle" transform="rotate(-90 24 {height/2})">{escape(y_label)}</text>
<text x="{margin_left-10}" y="{height-margin_bottom+4}" font-family="Arial" font-size="11" text-anchor="end">0</text>
<text x="{margin_left-10}" y="{margin_top+4}" font-family="Arial" font-size="11" text-anchor="end">{value_max:.2f}</text>
{''.join(bars)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def legend_svg(items: list[tuple[str, str]], x: float, y: float) -> str:
    parts: list[str] = []
    current_x = x
    for label, color in items:
        label_width = 10 + len(label) * 7
        parts.append(
            f'<rect x="{current_x:.2f}" y="{y-10:.2f}" width="10" height="10" fill="{color}"/>'
            f'<text x="{current_x + 16:.2f}" y="{y:.2f}" font-family="Arial" font-size="12">{escape(label)}</text>'
        )
        current_x += label_width + 28
    return "".join(parts)


def chunk_file_stats(path: Path) -> dict[str, float]:
    chunks = read_json(path)
    return {
        "mean_chunk_tokens": mean([chunk["token_count"] for chunk in chunks]),
        "mean_answer_units_per_chunk": mean(
            [
                len(chunk.get("metadata", {}).get("answer_unit_ids", []))
                for chunk in chunks
            ]
        ),
    }


def discover_chunk_file(chunks_dir: Path, config_id: str) -> Path:
    matches = sorted(chunks_dir.rglob(f"{config_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No chunk file found for {config_id} under {chunks_dir}")
    return matches[0]


def strategy_from_config_id(config_id: str) -> str:
    if config_id.startswith("fs_"):
        return "fixed_size"
    if config_id.startswith("sem_"):
        return "semantic"
    if config_id.startswith("hier_"):
        return "hierarchical"
    return "unknown"


def metric_columns(values: dict[str, Any]) -> dict[str, Any]:
    return {key: values.get(key) for key in METRIC_KEYS}


def prefix_columns(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def ordered_conditions(values: dict[str, Any]) -> list[str]:
    known = [condition for condition in CONDITION_ORDER if condition in values]
    unknown = sorted(condition for condition in values if condition not in CONDITION_ORDER)
    return known + unknown


def score_distributions_by_condition(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for record in records:
        condition = grouped.setdefault(record["condition_id"], {})
        for key in (
            "faithfulness_score",
            "correctness_score",
        ):
            condition.setdefault(key, []).append(float(record[key]))
    return grouped


def distribution_columns(prefix: str, values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        f"{prefix}median": median(ordered),
        f"{prefix}q1": quantile(ordered, 0.25),
        f"{prefix}q3": quantile(ordered, 0.75),
        f"{prefix}iqr": quantile(ordered, 0.75) - quantile(ordered, 0.25),
    }


def mean(values: list[float | int]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def average_metric_dict(records: list[dict[str, float]]) -> dict[str, float]:
    if not records:
        return {}
    return {
        key: mean([record[key] for record in records])
        for key in records[0]
    }


def median(values: list[float]) -> float:
    return quantile(sorted(values), 0.5)


def quantile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def value_to_y(
    value: float,
    value_min: float,
    value_max: float,
    margin_top: float,
    plot_height: float,
) -> float:
    if value_max == value_min:
        return margin_top + plot_height
    return margin_top + plot_height * (1 - (value - value_min) / (value_max - value_min))


def int_key(item: tuple[str, Any]) -> int:
    return int(item[0])


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def escape(value: str) -> str:
    return html.escape(value, quote=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 8 analysis reporting.")
    parser.add_argument(
        "--retrieval-metrics-path",
        type=Path,
        default=Path("data/evaluation/retrieval_metrics.json"),
    )
    parser.add_argument(
        "--retrieval-results-path",
        type=Path,
        default=Path("data/retrieval/retrieval_results.json"),
    )
    parser.add_argument(
        "--chunk-characteristics-path",
        type=Path,
        default=Path("data/chunks/chunk_characteristics.json"),
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument(
        "--context-coverage-metrics-path",
        type=Path,
        default=Path("data/evaluation/context_coverage_metrics.json"),
    )
    parser.add_argument(
        "--context-assembly-records-path",
        type=Path,
        default=Path("data/evaluation/context_assembly_records.json"),
    )
    parser.add_argument(
        "--generation-metrics-path",
        type=Path,
        default=Path("data/evaluation/generation_metrics.json"),
    )
    parser.add_argument(
        "--generated-answer-evaluation-path",
        type=Path,
        default=Path("data/evaluation/generated_answer_evaluation.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/results"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_stage8_outputs(
        retrieval_metrics_path=args.retrieval_metrics_path,
        retrieval_results_path=args.retrieval_results_path,
        chunk_characteristics_path=args.chunk_characteristics_path,
        chunks_dir=args.chunks_dir,
        context_coverage_metrics_path=args.context_coverage_metrics_path,
        context_assembly_records_path=args.context_assembly_records_path,
        generation_metrics_path=args.generation_metrics_path,
        generated_answer_evaluation_path=args.generated_answer_evaluation_path,
        output_dir=args.output_dir,
    )
    print(
        "Stage 8 analysis complete: "
        f"{summary['retrieval']['target_acts_top_10_config_count']} retrieval configs, "
        f"{summary['generation']['zero_correctness_case_count']} zero-correctness cases."
    )


if __name__ == "__main__":
    main()
