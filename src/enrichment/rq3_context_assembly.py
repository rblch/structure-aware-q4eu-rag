from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from enrichment.context_assembly import (
    assemble_context,
    assert_base_records_current,
    base_retrieval_candidates,
    load_base_retrieval_records,
    summarize_context_group,
)
from retrieval.index_corpus import file_sha256


CHECKPOINT_TYPE = "rq3_cross_chunking"


def write_rq3_context_assembly_outputs(
    *,
    config_path: Path,
    queries_path: Path,
    chunks_dir: Path,
    retrieval_results_path: Path,
    context_records_path: Path,
    context_metrics_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    rq3_config = validate_rq3_config(config)
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    retrieval_records = json.loads(
        retrieval_results_path.read_text(encoding="utf-8")
    )
    expected_query_count = rq3_config["expected_query_count"]
    if len(queries) != expected_query_count:
        raise RuntimeError(
            f"RQ3 requires {expected_query_count} queries, found {len(queries)}"
        )

    records: list[dict[str, Any]] = []
    source_chunks: dict[str, dict[str, str]] = {}
    for condition in rq3_config["new_conditions"]:
        strategy = condition["strategy"]
        config_id = condition["config_id"]
        chunk_path = chunks_dir / strategy / f"{config_id}.json"
        chunks = json.loads(chunk_path.read_text(encoding="utf-8"))
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        base_records = load_base_retrieval_records(
            retrieval_records,
            base_config_id=config_id,
            search_scope=rq3_config["search_scope"],
        )
        assert_base_records_current(
            base_records,
            queries=queries,
            base_chunk_file=chunk_path,
        )
        if set(base_records) != {query["query_id"] for query in queries}:
            raise RuntimeError(f"RQ3 retrieval records are incomplete for {config_id}")

        source_chunks[condition["id"]] = {
            "config_id": config_id,
            "strategy": strategy,
            "path": str(chunk_path),
            "sha256": file_sha256(chunk_path),
        }
        for query in queries:
            started = time.perf_counter()
            candidates = base_retrieval_candidates(
                base_records[query["query_id"]], rq3_config["top_k"]
            )
            record = assemble_context(
                query=query,
                condition_id=condition["id"],
                candidates=candidates,
                chunks_by_id=chunks_by_id,
                context_budget_tokens=rq3_config["context_budget_tokens"],
                base_config_id=config_id,
                base_top_k=rq3_config["top_k"],
            )
            record["strategy"] = strategy
            record["assembly_seconds"] = time.perf_counter() - started
            records.append(record)

    records.sort(key=lambda record: (record["query_id"], record["condition_id"]))
    assert_rq3_context_records(records, rq3_config)
    serialized_records = json.dumps(records, indent=2) + "\n"
    metrics = {
        "checkpoint_type": CHECKPOINT_TYPE,
        "ready_for_generation": True,
        "rq3_config": rq3_config,
        "record_count": len(records),
        "context_records_sha256": hashlib.sha256(
            serialized_records.encode("utf-8")
        ).hexdigest(),
        "sources": {
            "queries_path": str(queries_path),
            "queries_sha256": file_sha256(queries_path),
            "retrieval_results_path": str(retrieval_results_path),
            "retrieval_results_sha256": file_sha256(retrieval_results_path),
            "chunks": source_chunks,
        },
        "conditions": {
            condition["id"]: summarize_context_group(
                [
                    record
                    for record in records
                    if record["condition_id"] == condition["id"]
                ]
            )
            for condition in rq3_config["new_conditions"]
        },
    }

    context_records_path.parent.mkdir(parents=True, exist_ok=True)
    context_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    context_records_path.write_text(serialized_records, encoding="utf-8")
    context_metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return records, metrics


def validate_rq3_config(config: dict[str, Any]) -> dict[str, Any]:
    rq3_config = config.get("rq3_cross_chunking")
    if not isinstance(rq3_config, dict):
        raise ValueError("rq3_cross_chunking must be configured")
    conditions = rq3_config.get("new_conditions")
    if not isinstance(conditions, list) or len(conditions) != 2:
        raise ValueError("rq3_cross_chunking.new_conditions must contain two entries")
    expected = {"fixed_size": "fs_256_50", "semantic": "sem_50_256"}
    actual = {
        condition.get("strategy"): condition.get("config_id")
        for condition in conditions
    }
    if actual != expected:
        raise ValueError(
            "RQ3 generation must use the confirmatory fixed-size and semantic "
            f"configurations; got {actual}"
        )
    condition_ids = [condition.get("id") for condition in conditions]
    if len(set(condition_ids)) != 2 or any(not value for value in condition_ids):
        raise ValueError("RQ3 condition IDs must be distinct non-empty strings")
    if rq3_config.get("search_scope") != "all_acts":
        raise ValueError("RQ3 search_scope must be all_acts")
    if rq3_config.get("top_k") != 5:
        raise ValueError("RQ3 top_k must be 5 to match hierarchical no_enrichment")
    if rq3_config.get("expected_query_count") != 72:
        raise ValueError("RQ3 expected_query_count must be 72")
    if rq3_config.get("context_budget_tokens") != config["enrichment"][
        "context_budget_tokens"
    ]:
        raise ValueError("RQ3 and enrichment context budgets must match")
    reference = rq3_config.get("hierarchical_reference") or {}
    if reference != {
        "condition_id": "no_enrichment",
        "config_id": "hier_paragraph",
        "strategy": "hierarchical",
    }:
        raise ValueError("RQ3 hierarchical reference must be hier_paragraph/no_enrichment")
    return rq3_config


def assert_rq3_context_records(
    records: list[dict[str, Any]], rq3_config: dict[str, Any]
) -> None:
    expected_query_count = rq3_config["expected_query_count"]
    expected_conditions = {
        condition["id"]: condition for condition in rq3_config["new_conditions"]
    }
    if len(records) != expected_query_count * len(expected_conditions):
        raise RuntimeError(f"RQ3 expected 144 context records, found {len(records)}")
    for condition_id, condition in expected_conditions.items():
        group = [record for record in records if record["condition_id"] == condition_id]
        if len(group) != expected_query_count:
            raise RuntimeError(
                f"RQ3 condition {condition_id} expected {expected_query_count} records, "
                f"found {len(group)}"
            )
        query_ids = {record["query_id"] for record in group}
        if len(query_ids) != expected_query_count:
            raise RuntimeError(f"RQ3 condition {condition_id} has duplicate query IDs")
        for record in group:
            if record["base_config_id"] != condition["config_id"]:
                raise RuntimeError(f"RQ3 context config mismatch for {condition_id}")
            if record["strategy"] != condition["strategy"]:
                raise RuntimeError(f"RQ3 strategy mismatch for {condition_id}")
            if record["search_scope"] != rq3_config["search_scope"]:
                raise RuntimeError(f"RQ3 search-scope mismatch for {condition_id}")
            if record["base_top_k"] != rq3_config["top_k"]:
                raise RuntimeError(f"RQ3 top-k mismatch for {condition_id}")
            if record["context_budget_tokens"] != rq3_config[
                "context_budget_tokens"
            ]:
                raise RuntimeError(f"RQ3 context-budget mismatch for {condition_id}")
            if record["candidate_chunk_count"] != rq3_config["top_k"]:
                raise RuntimeError(f"RQ3 ranking has fewer than top-k chunks for {condition_id}")
            if record["included_chunk_count"] != rq3_config["top_k"]:
                raise RuntimeError(
                    f"RQ3 context budget excluded a top-k chunk for {condition_id}/"
                    f"{record['query_id']}"
                )
            if record["excluded_chunk_count"]:
                raise RuntimeError(f"RQ3 context unexpectedly excluded chunks for {condition_id}")
