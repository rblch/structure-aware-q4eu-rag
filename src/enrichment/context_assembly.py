from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.chunk import chunk_id_for_unit
from chunking.tokenization import DEFAULT_ENCODING, count_tokens
from retrieval.evaluate_retrieval import (
    embed_queries,
    load_chunk_sets,
    ranked_answer_units_from_chunks,
    search_chunks,
)
from retrieval.index_corpus import file_sha256
from utils.summary_stats import percentile


PRIMARY_SEARCH_SCOPE = "all_acts"
ARTICLE_CONFIG_ID = "hier_article"


def write_context_assembly_outputs(
    *,
    config_path: Path,
    queries_path: Path,
    legal_units_path: Path,
    chunks_dir: Path,
    embeddings_dir: Path,
    indices_dir: Path,
    xref_graph_path: Path,
    retrieval_results_path: Path,
    context_records_path: Path,
    coverage_metrics_path: Path,
    xref_eligibility_report_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    xref_graph = json.loads(xref_graph_path.read_text(encoding="utf-8"))
    retrieval_records = json.loads(retrieval_results_path.read_text(encoding="utf-8"))

    enrichment_config = config["enrichment"]
    base_config_id = enrichment_config["base_config_id"]
    base_top_k = enrichment_config["top_k"]
    extended_k = enrichment_config["volume_match_extended_k"]

    chunks_by_id = load_enrichment_chunks(chunks_dir, base_config_id)
    base_records = load_base_retrieval_records(
        retrieval_records,
        base_config_id=base_config_id,
        search_scope=PRIMARY_SEARCH_SCOPE,
    )
    extended_rankings = build_extended_base_rankings(
        config=config,
        queries=queries,
        chunks_dir=chunks_dir,
        embeddings_dir=embeddings_dir,
        indices_dir=indices_dir,
        base_config_id=base_config_id,
        extended_k=extended_k,
    )
    assert_base_records_current(
        base_records,
        queries=queries,
        base_chunk_file=chunks_dir / "hierarchical" / f"{base_config_id}.json",
    )
    assert_extended_rankings_match_base(
        base_records,
        extended_rankings,
        base_top_k=base_top_k,
    )
    assert_stage5_inputs_current(
        chunks_by_id=chunks_by_id,
        xref_graph=xref_graph,
        legal_units_path=legal_units_path,
    )
    edges_by_source = build_edges_by_source(xref_graph)

    records, uncapped_combined_tokens = build_context_records(
        queries=queries,
        conditions=enrichment_config["conditions"],
        chunks_by_id=chunks_by_id,
        base_records=base_records,
        extended_rankings=extended_rankings,
        edges_by_source=edges_by_source,
        base_config_id=base_config_id,
        base_top_k=base_top_k,
        context_budget_tokens=enrichment_config["context_budget_tokens"],
        volume_match_tolerance=enrichment_config["volume_match_tolerance"],
    )
    coverage_metrics = summarize_context_records(records, config)
    coverage_metrics["budget_derivation"] = budget_derivation_record(
        uncapped_combined_tokens=uncapped_combined_tokens,
        configured_budget=enrichment_config["context_budget_tokens"],
    )
    try:
        assert_context_budget_matches_derivation(coverage_metrics["budget_derivation"])
        assert_volume_match_within_tolerance(
            coverage_metrics.get("volume_match"),
            tolerance=enrichment_config["volume_match_tolerance"],
        )
    except RuntimeError:
        # Prevent stale metrics from authorizing Stage 6.
        coverage_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        coverage_metrics_path.write_text(
            json.dumps(coverage_metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    updated_queries, xref_report = apply_retrieval_conditioned_xref_eligibility(
        queries=queries,
        xref_graph=xref_graph,
        base_records=base_records,
        chunks_by_id=chunks_by_id,
        base_config_id=base_config_id,
        base_top_k=base_top_k,
        existing_report=json.loads(
            xref_eligibility_report_path.read_text(encoding="utf-8")
        )
        if xref_eligibility_report_path.exists()
        else None,
    )

    context_records_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    context_records_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    coverage_metrics_path.write_text(
        json.dumps(coverage_metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        json.dumps(updated_queries, indent=2) + "\n",
        encoding="utf-8",
    )
    xref_eligibility_report_path.parent.mkdir(parents=True, exist_ok=True)
    xref_eligibility_report_path.write_text(
        json.dumps(xref_report, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, coverage_metrics, xref_report


def assert_extended_rankings_match_base(
    base_records: dict[str, dict[str, Any]],
    extended_rankings: dict[str, list[dict[str, Any]]],
    *,
    base_top_k: int,
) -> None:
    # Preserve the Stage 4 top-k prefix in the extended ranking.
    mismatched: list[str] = []
    for query_id, base_record in sorted(base_records.items()):
        base_top = [
            retrieved["chunk_id"]
            for retrieved in base_record["retrieved_chunks"][:base_top_k]
        ]
        extended_top = [
            retrieved["chunk_id"]
            for retrieved in extended_rankings.get(query_id, [])[: len(base_top)]
        ]
        if base_top != extended_top:
            mismatched.append(query_id)
    if mismatched:
        raise RuntimeError(
            "Extended volume-match ranking disagrees with the stored Stage 4 "
            f"base ranking for {len(mismatched)} queries "
            f"({mismatched[:5]}); the query embeddings no longer reproduce "
            "the retrieval the other conditions were built from. Restore "
            "data/embeddings/text_cache or rerun the retrieve stage before "
            "assembling contexts."
        )


def assert_stage5_inputs_current(
    *,
    chunks_by_id: dict[str, dict[str, Any]],
    xref_graph: dict[str, Any],
    legal_units_path: Path,
) -> None:
    # Reject inputs produced by different parse runs.
    current_legal_units_sha256 = file_sha256(legal_units_path)
    if xref_graph.get("legal_units_sha256") != current_legal_units_sha256:
        raise RuntimeError(
            "Xref graph was built from a different legal_units.json than the "
            "current one; rerun the parse stage."
        )
    legal_units = json.loads(legal_units_path.read_text(encoding="utf-8"))
    document_hashes = {
        unit["document_code"]: unit["source_text_sha256"] for unit in legal_units
    }
    stale_documents = sorted(
        {
            chunk["metadata"]["document_code"]
            for chunk in chunks_by_id.values()
            if chunk["source_text_sha256"]
            != document_hashes.get(chunk["metadata"]["document_code"])
        }
    )
    if stale_documents:
        raise RuntimeError(
            f"Enrichment chunks for documents {stale_documents} were built "
            "from a different parsed corpus; rerun the chunk stage."
        )


def load_enrichment_chunks(chunks_dir: Path, base_config_id: str) -> dict[str, Any]:
    chunks_by_id: dict[str, Any] = {}
    for config_id in (base_config_id, ARTICLE_CONFIG_ID):
        path = chunks_dir / "hierarchical" / f"{config_id}.json"
        for chunk in json.loads(path.read_text(encoding="utf-8")):
            chunks_by_id[chunk["chunk_id"]] = chunk
    return chunks_by_id


def load_base_retrieval_records(
    retrieval_records: list[dict[str, Any]],
    *,
    base_config_id: str,
    search_scope: str,
) -> dict[str, dict[str, Any]]:
    records = {
        record["query_id"]: record
        for record in retrieval_records
        if record["config_id"] == base_config_id
        and record["search_scope"] == search_scope
    }
    if not records:
        raise ValueError(
            f"No retrieval records found for {base_config_id}/{search_scope}"
        )
    return records


def assert_base_records_current(
    base_records: dict[str, dict[str, Any]],
    *,
    queries: list[dict[str, Any]],
    base_chunk_file: Path,
) -> None:
    # Reject retrieval artifacts from different chunks or queries.
    current_chunk_sha256 = file_sha256(base_chunk_file)
    problems: list[str] = []
    stale_hashes = sorted(
        {
            str(record.get("chunk_file_sha256"))
            for record in base_records.values()
            if record.get("chunk_file_sha256") != current_chunk_sha256
        }
    )
    if stale_hashes:
        problems.append(
            f"rankings were computed from a different {base_chunk_file.name} "
            f"(stored {stale_hashes}, current {current_chunk_sha256})"
        )
    for query in queries:
        record = base_records.get(query["query_id"])
        if record is None:
            problems.append(f"no ranking for query {query['query_id']}")
        elif (
            record["question"] != query["question"]
            or record["gold_unit_ids"] != query["gold_unit_ids"]
        ):
            problems.append(
                f"query {query['query_id']} question/gold set differs from "
                "the one the rankings were computed with"
            )
    if problems:
        raise RuntimeError(
            "Stage 4 rankings are stale; rerun the retrieve stage: "
            + "; ".join(problems[:5])
        )


def build_extended_base_rankings(
    *,
    config: dict[str, Any],
    queries: list[dict[str, Any]],
    chunks_dir: Path,
    embeddings_dir: Path,
    indices_dir: Path,
    base_config_id: str,
    extended_k: int,
) -> dict[str, list[dict[str, Any]]]:
    chunk_sets = load_chunk_sets(config, chunks_dir, embeddings_dir, indices_dir)
    base_sets = [
        chunk_set
        for chunk_set in chunk_sets
        if chunk_set["config_id"] == base_config_id
    ]
    if len(base_sets) != 1:
        raise ValueError(f"Expected exactly one chunk set for {base_config_id}")

    query_embeddings = embed_queries(queries, config)
    rankings: dict[str, list[dict[str, Any]]] = {}
    for query, query_embedding in zip(queries, query_embeddings):
        rankings[query["query_id"]] = search_chunks(
            chunk_set=base_sets[0],
            query_embedding=query_embedding,
            allowed_document_codes=None,
            max_k=extended_k,
        )
    return rankings


def build_edges_by_source(xref_graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in xref_graph["edges"]:
        edges_by_source[edge["source_unit_id"]].append(edge)
    return dict(edges_by_source)


def build_context_records(
    *,
    queries: list[dict[str, Any]],
    conditions: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    base_records: dict[str, dict[str, Any]],
    extended_rankings: dict[str, list[dict[str, Any]]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    base_config_id: str,
    base_top_k: int,
    context_budget_tokens: int,
    volume_match_tolerance: float,
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    uncapped_combined_tokens: dict[str, int] = {}
    conditions_by_id = {condition["id"]: condition for condition in conditions}
    if "combined" not in conditions_by_id:
        raise ValueError("Enrichment config must include a combined condition")

    for query in queries:
        base_record = base_records[query["query_id"]]
        base_candidates = base_retrieval_candidates(base_record, base_top_k)
        combined_candidates = condition_candidates(
            condition=conditions_by_id["combined"],
            base_candidates=base_candidates,
            extended_ranking=extended_rankings[query["query_id"]],
            chunks_by_id=chunks_by_id,
            edges_by_source=edges_by_source,
        )
        combined_preview = assemble_context(
            query=query,
            condition_id="combined",
            candidates=combined_candidates,
            chunks_by_id=chunks_by_id,
            context_budget_tokens=context_budget_tokens,
            base_config_id=base_config_id,
            base_top_k=base_top_k,
            encoding_name=encoding_name,
        )
        # Match the capped combined context; retain uncapped size for p95.
        volume_target_tokens = combined_preview["context_token_count"]
        uncapped_combined = assemble_context(
            query=query,
            condition_id="combined_uncapped_derivation",
            candidates=combined_candidates,
            chunks_by_id=chunks_by_id,
            context_budget_tokens=None,
            base_config_id=base_config_id,
            base_top_k=base_top_k,
            encoding_name=encoding_name,
        )
        uncapped_combined_tokens[query["query_id"]] = uncapped_combined[
            "context_token_count"
        ]

        for condition in conditions:
            # RQ3 times candidate collection and assembly together.
            assembly_started = time.perf_counter()
            candidates = condition_candidates(
                condition=condition,
                base_candidates=base_candidates,
                extended_ranking=extended_rankings[query["query_id"]],
                chunks_by_id=chunks_by_id,
                edges_by_source=edges_by_source,
            )
            stop_at_tokens = (
                volume_target_tokens if condition.get("volume_matched") else None
            )
            record = assemble_context(
                query=query,
                condition_id=condition["id"],
                candidates=candidates,
                chunks_by_id=chunks_by_id,
                context_budget_tokens=context_budget_tokens,
                base_config_id=base_config_id,
                base_top_k=base_top_k,
                stop_at_tokens=stop_at_tokens,
                minimum_included_before_stop=base_top_k,
                volume_match_tolerance=(
                    volume_match_tolerance if stop_at_tokens is not None else None
                ),
                encoding_name=encoding_name,
            )
            record["assembly_seconds"] = time.perf_counter() - assembly_started
            records.append(record)
    return records, uncapped_combined_tokens


def base_retrieval_candidates(
    base_record: dict[str, Any],
    base_top_k: int,
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": retrieved["chunk_id"],
            "role": "retrieved",
            "retrieval_rank": retrieved["chunk_rank"],
            "retrieval_score": retrieved["score"],
        }
        for retrieved in base_record["retrieved_chunks"][:base_top_k]
    ]


def condition_candidates(
    *,
    condition: dict[str, Any],
    base_candidates: list[dict[str, Any]],
    extended_ranking: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates = list(base_candidates)
    if condition.get("parent"):
        candidates.extend(parent_candidates(base_candidates, chunks_by_id))
    if condition.get("xref"):
        candidates.extend(xref_candidates(base_candidates, chunks_by_id, edges_by_source))
    if condition.get("volume_matched"):
        candidates.extend(volume_match_candidates(extended_ranking))
    return dedupe_candidates(candidates)


def parent_candidates(
    base_candidates: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for base_candidate in base_candidates:
        chunk = chunks_by_id[base_candidate["chunk_id"]]
        parent_chunk_id = chunk["metadata"].get("parent_chunk_id")
        if parent_chunk_id and parent_chunk_id in chunks_by_id:
            candidates.append(
                {
                    "chunk_id": parent_chunk_id,
                    "role": "parent",
                    "source_chunk_id": chunk["chunk_id"],
                    "source_retrieval_rank": base_candidate["retrieval_rank"],
                }
            )
    return candidates


def xref_candidates(
    base_candidates: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for base_candidate in base_candidates:
        chunk = chunks_by_id[base_candidate["chunk_id"]]
        for source_unit_id in source_unit_candidates(chunk):
            for edge in edges_by_source.get(source_unit_id, []):
                target_chunk_id = chunk_id_for_unit(
                    ARTICLE_CONFIG_ID,
                    edge["target_unit_id_normalized"],
                )
                if target_chunk_id in chunks_by_id:
                    candidates.append(
                        {
                            "chunk_id": target_chunk_id,
                            "role": "xref",
                            "source_chunk_id": chunk["chunk_id"],
                            "source_retrieval_rank": base_candidate["retrieval_rank"],
                            "xref_source_unit_id": edge["source_unit_id"],
                            "xref_target_unit_id": edge["target_unit_id"],
                            "xref_target_unit_id_normalized": edge[
                                "target_unit_id_normalized"
                            ],
                            "xref_reference_kind": edge["reference_kind"],
                            "xref_raw_match": edge["raw_match"],
                        }
                    )
    return candidates


def volume_match_candidates(
    extended_ranking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": retrieved["chunk_id"],
            "role": "volume_retrieval",
            "retrieval_rank": retrieved["chunk_rank"],
            "retrieval_score": retrieved["score"],
        }
        for retrieved in extended_ranking
    ]


def source_unit_candidates(chunk: dict[str, Any]) -> list[str]:
    # Follow xrefs only from units whose text is present in the chunk.
    return list(chunk["metadata"]["source_unit_ids"])


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for candidate in candidates:
        chunk_id = candidate["chunk_id"]
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        deduped.append(candidate)
    return deduped


def assemble_context(
    *,
    query: dict[str, Any],
    condition_id: str,
    candidates: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    context_budget_tokens: int | None,
    base_config_id: str,
    base_top_k: int,
    stop_at_tokens: int | None = None,
    minimum_included_before_stop: int = 0,
    volume_match_tolerance: float | None = None,
    encoding_name: str = DEFAULT_ENCODING,
) -> dict[str, Any]:
    # None disables the cap for budget derivation.
    if stop_at_tokens is not None and volume_match_tolerance is None:
        raise ValueError("volume_match_tolerance is required with stop_at_tokens")
    included: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    passages: list[str] = []
    context_token_count = 0

    for candidate in candidates:
        chunk = chunks_by_id[candidate["chunk_id"]]
        passage = render_passage(len(included) + 1, chunk)
        tentative_context = "\n\n".join(passages + [passage])
        tentative_tokens = count_tokens(tentative_context, encoding_name)
        if context_budget_tokens is None or tentative_tokens <= context_budget_tokens:
            if (
                stop_at_tokens is not None
                and len(included) >= minimum_included_before_stop
                and tentative_tokens >= stop_at_tokens
                and abs(context_token_count - stop_at_tokens)
                < abs(tentative_tokens - stop_at_tokens)
            ):
                excluded.append(
                    {
                        **candidate_summary(candidate, chunk),
                        "excluded_reason": "volume_target_closest_prefix",
                    }
                )
                break
            passages.append(passage)
            context_token_count = tentative_tokens
            included.append(included_chunk_record(candidate, chunk))
            if (
                stop_at_tokens is not None
                and len(included) >= minimum_included_before_stop
                and context_token_count >= stop_at_tokens
            ):
                break
        else:
            excluded.append(
                {
                    **candidate_summary(candidate, chunk),
                    "excluded_reason": "context_budget",
                }
            )

    context_text = "\n\n".join(passages)
    answer_units = ranked_answer_units_from_chunks(included)
    coverage = compute_context_coverage(answer_units, query["gold_unit_ids"])
    volume_match: dict[str, Any] = {}
    if stop_at_tokens is not None:
        # Preserve signed error because the target may under- or overshoot.
        delta = context_token_count - stop_at_tokens
        relative_error = abs(delta) / stop_at_tokens if stop_at_tokens else 0.0
        volume_match = {
            "volume_target_tokens": stop_at_tokens,
            "volume_token_delta": delta,
            "volume_relative_error": relative_error,
            "volume_target_met": relative_error <= volume_match_tolerance,
        }
    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "specificity": query["specificity"],
        "target_document_codes": query["target_document_codes"],
        "gold_unit_ids": query["gold_unit_ids"],
        "base_config_id": base_config_id,
        "search_scope": PRIMARY_SEARCH_SCOPE,
        "base_top_k": base_top_k,
        "condition_id": condition_id,
        "context_budget_tokens": context_budget_tokens,
        "candidate_chunk_count": len(candidates),
        "included_chunk_count": len(included),
        "excluded_chunk_count": len(excluded),
        "context_token_count": context_token_count,
        "context_text": context_text,
        "included_chunks": included,
        "excluded_chunks": excluded,
        "context_answer_unit_ids": answer_units,
        **coverage,
        **volume_match,
    }


def render_passage(passage_index: int, chunk: dict[str, Any]) -> str:
    metadata = chunk["metadata"]
    source_refs = ", ".join(metadata.get("source_unit_ids", []))
    answer_units = ", ".join(metadata.get("answer_unit_ids", []))
    return (
        f"[{passage_index}] Reference: {source_refs}\n"
        f"Document: {metadata['document_code']}\n"
        f"Answer unit: {answer_units}\n"
        f"{chunk['text']}"
    )


def included_chunk_record(
    candidate: dict[str, Any],
    chunk: dict[str, Any],
) -> dict[str, Any]:
    return {
        **candidate_summary(candidate, chunk),
        "text_token_count": chunk["token_count"],
    }


def candidate_summary(candidate: dict[str, Any], chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk["metadata"]
    summary = {
        "chunk_id": chunk["chunk_id"],
        "role": candidate["role"],
        "document_code": metadata["document_code"],
        "source_unit_ids": metadata.get("source_unit_ids", []),
        "answer_unit_ids": metadata.get("answer_unit_ids", []),
    }
    optional_keys = [
        "retrieval_rank",
        "retrieval_score",
        "source_chunk_id",
        "source_retrieval_rank",
        "xref_source_unit_id",
        "xref_target_unit_id",
        "xref_target_unit_id_normalized",
        "xref_reference_kind",
        "xref_raw_match",
    ]
    for key in optional_keys:
        if key in candidate:
            summary[key] = candidate[key]
    return summary


def compute_context_coverage(
    context_answer_unit_ids: list[str],
    gold_unit_ids: list[str],
) -> dict[str, Any]:
    gold = set(gold_unit_ids)
    retrieved = set(context_answer_unit_ids)
    relevant = sorted(gold & retrieved)
    recall = len(relevant) / len(gold) if gold else 0.0
    precision = len(relevant) / len(retrieved) if retrieved else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "context_gold_recall": recall,
        "context_gold_precision": precision,
        "context_gold_f1": f1,
        "context_relevant_gold_unit_ids": relevant,
        "context_missing_gold_unit_ids": sorted(gold - retrieved),
    }


def budget_derivation_record(
    *,
    uncapped_combined_tokens: dict[str, int],
    configured_budget: int,
    rounding_step: int = 500,
) -> dict[str, Any]:
    """Apply the registered rounded-p95 budget rule."""
    sizes = list(uncapped_combined_tokens.values())
    p95_raw = percentile(sizes, 95)
    derived_budget = int(round(p95_raw / rounding_step) * rounding_step)
    return {
        "basis": "uncapped combined context token sizes over all queries",
        "query_count": len(sizes),
        "p95_raw": p95_raw,
        "rounding": f"nearest_{rounding_step}",
        "derived_budget": derived_budget,
        "configured_budget": configured_budget,
        "matches_configured_budget": derived_budget == configured_budget,
    }


def assert_context_budget_matches_derivation(derivation: dict[str, Any]) -> None:
    if derivation["matches_configured_budget"]:
        return

    raise RuntimeError(
        "Context budget does not match the registered derivation: configured "
        f"{derivation['configured_budget']} tokens, but the uncapped combined "
        f"context p95 is {derivation['p95_raw']:.1f}, yielding "
        f"{derivation['derived_budget']} tokens after {derivation['rounding']} "
        "rounding. No downstream-ready Stage 5 contexts were written. Before "
        "confirmatory "
        "generation, update config.yaml, the experiment specification, and "
        "Chapter 3 to the derived value; record and commit the pre-generation "
        "amendment; then rerun Stage 5."
    )


def summarize_context_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["condition_id"]].append(record)

    conditions = {
        condition_id: summarize_context_group(group_records)
        for condition_id, group_records in sorted(grouped.items())
    }
    return {
        "base_config_id": config["enrichment"]["base_config_id"],
        "search_scope": PRIMARY_SEARCH_SCOPE,
        "base_top_k": config["enrichment"]["top_k"],
        "context_budget_tokens": config["enrichment"]["context_budget_tokens"],
        "record_count": len(records),
        "conditions": conditions,
        "volume_match": summarize_volume_match(
            records,
            tolerance=config["enrichment"]["volume_match_tolerance"],
        ),
    }


def summarize_volume_match(
    records: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, Any] | None:
    volume_records = sorted(
        (record for record in records if "volume_target_tokens" in record),
        key=lambda record: record["query_id"],
    )
    if not volume_records:
        return None
    relative_errors = [record["volume_relative_error"] for record in volume_records]
    outside = [record for record in volume_records if not record["volume_target_met"]]
    return {
        "tolerance": tolerance,
        "query_count": len(volume_records),
        "outside_tolerance_count": len(outside),
        "outside_tolerance_proportion": len(outside) / len(volume_records),
        "max_relative_error": max(relative_errors),
        "mean_relative_error": sum(relative_errors) / len(relative_errors),
        "median_relative_error": percentile(relative_errors, 50),
        "p95_relative_error": percentile(relative_errors, 95),
        "outside_tolerance_queries": [
            {
                "query_id": record["query_id"],
                "volume_target_tokens": record["volume_target_tokens"],
                "context_token_count": record["context_token_count"],
                "volume_token_delta": record["volume_token_delta"],
                "volume_relative_error": record["volume_relative_error"],
            }
            for record in outside
        ],
    }


def assert_volume_match_within_tolerance(
    volume_summary: dict[str, Any] | None,
    *,
    tolerance: float,
) -> None:
    # Block invalid volume controls before generation.
    if volume_summary is None or not volume_summary["outside_tolerance_count"]:
        return
    affected = [
        entry["query_id"] for entry in volume_summary["outside_tolerance_queries"]
    ]
    raise RuntimeError(
        f"Volume matching failed for {volume_summary['outside_tolerance_count']} "
        f"of {volume_summary['query_count']} queries; maximum relative error "
        f"{volume_summary['max_relative_error']:.1%} exceeds the registered "
        f"tolerance {tolerance:.0%}. Increase enrichment.volume_match_extended_k, "
        "inspect candidate chunk sizes, or revise the registered tolerance "
        f"before generation. Affected queries: {affected[:10]}"
    )


def summarize_context_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    assembly_timings = [
        record["assembly_seconds"]
        for record in records
        if record.get("assembly_seconds") is not None
    ]
    return {
        "query_count": len(records),
        "macro": mean_context_metrics(records),
        "context_token_count": numeric_summary(
            [record["context_token_count"] for record in records]
        ),
        "included_chunk_count": numeric_summary(
            [record["included_chunk_count"] for record in records]
        ),
        "assembly_seconds": (
            numeric_summary(assembly_timings) if assembly_timings else None
        ),
        "by_specificity": summarize_context_by_field(records, "specificity"),
        "by_target_document": summarize_context_by_target_document(records),
    }


def summarize_context_by_field(
    records: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[field]].append(record)
    return {
        value: {
            "query_count": len(value_records),
            "macro": mean_context_metrics(value_records),
        }
        for value, value_records in sorted(grouped.items())
    }


def summarize_context_by_target_document(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for document_code in record["target_document_codes"]:
            grouped[document_code].append(record)
    return {
        document_code: {
            "query_count": len(document_records),
            "macro": mean_context_metrics(document_records),
        }
        for document_code, document_records in sorted(grouped.items())
    }


def mean_context_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        "context_gold_recall",
        "context_gold_precision",
        "context_gold_f1",
        "context_token_count",
        "included_chunk_count",
        "excluded_chunk_count",
        "candidate_chunk_count",
    ]
    return {
        key: sum(float(record[key]) for record in records) / len(records)
        for key in keys
    }


def numeric_summary(values: list[int | float]) -> dict[str, float]:
    sorted_values = sorted(float(value) for value in values)
    return {
        "min": sorted_values[0],
        "median": percentile(sorted_values, 50),
        "p95": percentile(sorted_values, 95),
        "max": sorted_values[-1],
    }


def apply_retrieval_conditioned_xref_eligibility(
    *,
    queries: list[dict[str, Any]],
    xref_graph: dict[str, Any],
    base_records: dict[str, dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    base_config_id: str,
    base_top_k: int,
    existing_report: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    edges_by_source = build_edges_by_source(xref_graph)
    updated_queries: list[dict[str, Any]] = []
    eligible_queries: list[dict[str, Any]] = []

    for query in queries:
        updated = dict(query)
        updated["xref_eligible"] = query_is_retrieval_conditioned_xref_eligible(
            query=query,
            base_record=base_records[query["query_id"]],
            chunks_by_id=chunks_by_id,
            edges_by_source=edges_by_source,
            base_top_k=base_top_k,
        )
        if updated["xref_eligible"]:
            eligible_queries.append(updated)
        updated_queries.append(updated)

    report = dict(existing_report or {})
    by_specificity = Counter(query["specificity"] for query in eligible_queries)
    by_act = Counter(
        document_code
        for query in eligible_queries
        for document_code in query["target_document_codes"]
    )
    eligible_count = len(eligible_queries)
    report.update(
        {
            "stage": "1c-ii",
            "retrieval_conditioned_count": eligible_count,
            "retrieval_conditioned_by_specificity": dict(sorted(by_specificity.items())),
            "retrieval_conditioned_by_act": dict(sorted(by_act.items())),
            "retrieval_conditioned_base_config_id": base_config_id,
            "retrieval_conditioned_top_k": base_top_k,
            "retrieval_conditioned_query_ids": [
                query["query_id"] for query in eligible_queries
            ],
            "xref_power_decision": xref_power_decision(eligible_count),
        }
    )
    return updated_queries, report


def query_is_retrieval_conditioned_xref_eligible(
    *,
    query: dict[str, Any],
    base_record: dict[str, Any],
    chunks_by_id: dict[str, dict[str, Any]],
    edges_by_source: dict[str, list[dict[str, Any]]],
    base_top_k: int,
) -> bool:
    gold_units = set(query["gold_unit_ids"])
    if len(gold_units) < 2:
        return False
    for candidate in base_retrieval_candidates(base_record, base_top_k):
        chunk = chunks_by_id[candidate["chunk_id"]]
        for source_unit_id in source_unit_candidates(chunk):
            if any(
                edge["target_unit_id_normalized"] in gold_units
                for edge in edges_by_source.get(source_unit_id, [])
            ):
                return True
    return False


def xref_power_decision(eligible_count: int) -> str:
    if eligible_count < 20:
        return "xref_contrasts_exploratory_underpowered"
    return "xref_contrasts_confirmatory_allowed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 5 context assembly.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/dataset/q4eu_queries.json"),
    )
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--indices-dir", type=Path, default=Path("data/indices"))
    parser.add_argument(
        "--xref-graph-path",
        type=Path,
        default=Path("data/parsed/xref_graph.json"),
    )
    parser.add_argument(
        "--retrieval-results-path",
        type=Path,
        default=Path("data/retrieval/retrieval_results.json"),
    )
    parser.add_argument(
        "--context-records-path",
        type=Path,
        default=Path("data/evaluation/context_assembly_records.json"),
    )
    parser.add_argument(
        "--coverage-metrics-path",
        type=Path,
        default=Path("data/evaluation/context_coverage_metrics.json"),
    )
    parser.add_argument(
        "--xref-eligibility-report-path",
        type=Path,
        default=Path("data/dataset/xref_eligibility_report.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records, _, xref_report = write_context_assembly_outputs(
        config_path=args.config_path,
        queries_path=args.queries_path,
        legal_units_path=args.legal_units_path,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        indices_dir=args.indices_dir,
        xref_graph_path=args.xref_graph_path,
        retrieval_results_path=args.retrieval_results_path,
        context_records_path=args.context_records_path,
        coverage_metrics_path=args.coverage_metrics_path,
        xref_eligibility_report_path=args.xref_eligibility_report_path,
    )
    print(
        "Stage 5 context assembly complete: "
        f"{len(records)} records, "
        f"{xref_report['retrieval_conditioned_count']} xref-eligible queries."
    )


if __name__ == "__main__":
    main()
