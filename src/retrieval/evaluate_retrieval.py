from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_import.id_normalization import dedupe_preserving_order
from evaluation.ir_metrics import compute_ir_metrics, mean_metric_dict
from retrieval.index_corpus import (
    build_embedding_metadata,
    config_chunk_files,
    file_sha256,
    metadata_matches,
    l2_normalize,
)
from retrieval.openrouter_embeddings import OpenRouterEmbedder
from utils.summary_stats import percentile


def write_retrieval_outputs(
    *,
    config_path: Path,
    queries_path: Path,
    chunks_dir: Path,
    embeddings_dir: Path,
    indices_dir: Path,
    retrieval_results_path: Path,
    retrieval_metrics_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    chunk_sets = load_chunk_sets(config, chunks_dir, embeddings_dir, indices_dir)
    query_embeddings, query_embedding_seconds = embed_queries_timed(queries, config)
    top_k_values = config["retrieval"]["top_k_values"]
    max_k = max(top_k_values)

    records: list[dict[str, Any]] = []
    for chunk_set in chunk_sets:
        for scope in config["retrieval"]["search_scopes"]:
            for query, query_embedding, embedding_seconds in zip(
                queries, query_embeddings, query_embedding_seconds
            ):
                if not scope_includes_query(scope, query):
                    continue
                search_started = time.perf_counter()
                ranked_chunks = search_chunks(
                    chunk_set=chunk_set,
                    query_embedding=query_embedding,
                    allowed_document_codes=scope_allowed_document_codes(
                        scope, query
                    ),
                    max_k=max_k,
                )
                search_seconds = time.perf_counter() - search_started
                records.append(
                    build_retrieval_record(
                        query=query,
                        chunk_set=chunk_set,
                        search_scope=scope["id"],
                        ranked_chunks=ranked_chunks,
                        top_k_values=top_k_values,
                        query_embedding_seconds=embedding_seconds,
                        search_seconds=search_seconds,
                    )
                )

    metrics = summarize_retrieval_records(records, config)
    retrieval_results_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    retrieval_results_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    retrieval_metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, metrics


def recompute_metrics_from_results(
    *,
    config_path: Path,
    retrieval_results_path: Path,
    retrieval_metrics_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-score stored rankings without embedding calls."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    records = json.loads(retrieval_results_path.read_text(encoding="utf-8"))
    top_k_values = config["retrieval"]["top_k_values"]
    for record in records:
        metrics_by_k: dict[str, Any] = {}
        ranked_answer_units_by_k: dict[str, list[str]] = {}
        for k in top_k_values:
            ranked_answer_units = ranked_answer_units_from_chunks(
                record["retrieved_chunks"][:k]
            )
            ranked_answer_units_by_k[str(k)] = ranked_answer_units
            metrics_by_k[str(k)] = compute_ir_metrics(
                ranked_answer_units=ranked_answer_units,
                gold_unit_ids=record["gold_unit_ids"],
            )
        record["ranked_answer_units_by_k"] = ranked_answer_units_by_k
        record["metrics_by_k"] = metrics_by_k

    metrics = summarize_retrieval_records(records, config)
    retrieval_results_path.write_text(
        json.dumps(records, indent=2) + "\n",
        encoding="utf-8",
    )
    retrieval_metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return records, metrics


def load_chunk_sets(
    config: dict[str, Any],
    chunks_dir: Path,
    embeddings_dir: Path,
    indices_dir: Path,
) -> list[dict[str, Any]]:
    chunk_sets: list[dict[str, Any]] = []
    for chunk_file in config_chunk_files(config, chunks_dir):
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        if not chunks:
            continue
        config_id = chunks[0]["config_id"]
        # Reject indices that do not match their embeddings.
        expected_metadata = build_embedding_metadata(chunks, chunk_file, config)
        metadata_path = embeddings_dir / config_id / "embedding_metadata.json"
        stored_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        )
        if stored_metadata is None or not metadata_matches(
            stored_metadata, expected_metadata
        ):
            raise RuntimeError(
                f"Embeddings/index for {config_id} were built from a "
                "different chunk file or embedding configuration; rerun the "
                "index stage."
            )
        embeddings_path = embeddings_dir / config_id / "embeddings.npy"
        chunk_id_order_path = embeddings_dir / config_id / "chunk_id_order.json"
        if (
            not embeddings_path.exists()
            or not chunk_id_order_path.exists()
            or stored_metadata.get("embeddings_sha256")
            != file_sha256(embeddings_path)
            or stored_metadata.get("chunk_id_order_sha256")
            != file_sha256(chunk_id_order_path)
        ):
            raise RuntimeError(
                f"embeddings.npy or chunk_id_order.json for {config_id} do "
                "not match the hashes recorded at embedding time; rerun the "
                "index stage."
            )
        chunk_id_order = json.loads(chunk_id_order_path.read_text(encoding="utf-8"))
        chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
        ordered_chunks = [chunks_by_id[chunk_id] for chunk_id in chunk_id_order]
        index = faiss.read_index(str(indices_dir / config_id / "index.faiss"))
        if index.ntotal != len(ordered_chunks):
            raise ValueError(f"Index row count mismatch for {config_id}")
        assert_index_matches_embeddings(
            index,
            np.load(embeddings_path),
            config_id=config_id,
        )
        chunk_sets.append(
            {
                "config_id": config_id,
                "strategy": chunks[0]["strategy"],
                "chunk_file": str(chunk_file),
                "chunk_file_sha256": expected_metadata["chunk_file_sha256"],
                "chunks": ordered_chunks,
                "index": index,
            }
        )
    return sorted(chunk_sets, key=lambda item: item["config_id"])


def assert_index_matches_embeddings(
    index: Any,
    embeddings: np.ndarray,
    *,
    config_id: str,
) -> None:
    # IndexFlatIP permits exact comparison with normalized embeddings.
    normalized = l2_normalize(embeddings)
    if index.ntotal != normalized.shape[0] or index.d != normalized.shape[1]:
        raise RuntimeError(
            f"FAISS index shape for {config_id} ({index.ntotal}x{index.d}) "
            f"does not match the embeddings {normalized.shape}; rerun the "
            "index stage."
        )
    if not np.allclose(index.reconstruct_n(0, index.ntotal), normalized, atol=1e-6):
        raise RuntimeError(
            f"FAISS index vectors for {config_id} differ from the current "
            "embeddings; rerun the index stage."
        )


def embed_queries(
    queries: list[dict[str, Any]],
    config: dict[str, Any],
) -> np.ndarray:
    embedder = OpenRouterEmbedder(
        config,
        cache_dir=Path("data/embeddings/text_cache"),
    )
    embeddings = np.asarray(
        embedder.embed_texts([query["question"] for query in queries]),
        dtype=np.float32,
    )
    return l2_normalize(embeddings)


def embed_queries_timed(
    queries: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[np.ndarray, list[float]]:
    """Embed queries individually and record wall-clock time."""
    embedder = OpenRouterEmbedder(
        config,
        cache_dir=Path("data/embeddings/text_cache"),
    )
    vectors: list[list[float]] = []
    timings: list[float] = []
    for query in queries:
        vector, elapsed_seconds = embedder.embed_single_timed(query["question"])
        vectors.append(vector)
        timings.append(elapsed_seconds)
    embeddings = np.asarray(vectors, dtype=np.float32)
    return l2_normalize(embeddings), timings


def scope_includes_query(scope: dict[str, Any], query: dict[str, Any]) -> bool:
    """Return whether a query belongs to a search scope."""
    fixed_codes = scope.get("restrict_to_document_codes")
    if fixed_codes and scope.get("only_queries_targeting_restriction", False):
        return set(query["target_document_codes"]) <= set(fixed_codes)
    return True


def scope_allowed_document_codes(
    scope: dict[str, Any],
    query: dict[str, Any],
) -> set[str] | None:
    """Return allowed document codes, or None when unrestricted."""
    fixed_codes = scope.get("restrict_to_document_codes")
    if fixed_codes:
        return set(fixed_codes)
    if scope.get("restrict_to_target_documents", False):
        return set(query["target_document_codes"])
    return None


def search_chunks(
    *,
    chunk_set: dict[str, Any],
    query_embedding: np.ndarray,
    allowed_document_codes: set[str] | None,
    max_k: int,
) -> list[dict[str, Any]]:
    index = chunk_set["index"]
    search_k = (
        index.ntotal
        if allowed_document_codes is not None
        else min(max_k, index.ntotal)
    )
    scores, indices = index.search(query_embedding.reshape(1, -1), search_k)
    ranked_chunks: list[dict[str, Any]] = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        chunk = chunk_set["chunks"][int(index_position)]
        document_code = chunk["metadata"]["document_code"]
        if (
            allowed_document_codes is not None
            and document_code not in allowed_document_codes
        ):
            continue
        ranked_chunks.append(
            {
                "chunk_rank": len(ranked_chunks) + 1,
                "chunk_id": chunk["chunk_id"],
                "score": float(score),
                "document_code": document_code,
                "answer_unit_ids": chunk["metadata"]["answer_unit_ids"],
            }
        )
        if len(ranked_chunks) == max_k:
            break
    return ranked_chunks


def build_retrieval_record(
    *,
    query: dict[str, Any],
    chunk_set: dict[str, Any],
    search_scope: str,
    ranked_chunks: list[dict[str, Any]],
    top_k_values: list[int],
    query_embedding_seconds: float | None = None,
    search_seconds: float | None = None,
) -> dict[str, Any]:
    metrics_by_k: dict[str, Any] = {}
    ranked_answer_units_by_k: dict[str, list[str]] = {}
    for k in top_k_values:
        ranked_answer_units = ranked_answer_units_from_chunks(ranked_chunks[:k])
        ranked_answer_units_by_k[str(k)] = ranked_answer_units
        metrics_by_k[str(k)] = compute_ir_metrics(
            ranked_answer_units=ranked_answer_units,
            gold_unit_ids=query["gold_unit_ids"],
        )

    return {
        "query_id": query["query_id"],
        "question": query["question"],
        "specificity": query["specificity"],
        "target_document_codes": query["target_document_codes"],
        "gold_unit_ids": query["gold_unit_ids"],
        "config_id": chunk_set["config_id"],
        "strategy": chunk_set["strategy"],
        "chunk_file_sha256": chunk_set["chunk_file_sha256"],
        "search_scope": search_scope,
        "query_embedding_seconds": query_embedding_seconds,
        "search_seconds": search_seconds,
        "retrieved_chunks": ranked_chunks,
        "ranked_answer_units_by_k": ranked_answer_units_by_k,
        "metrics_by_k": metrics_by_k,
    }


def ranked_answer_units_from_chunks(ranked_chunks: list[dict[str, Any]]) -> list[str]:
    return dedupe_preserving_order(
        unit_id for chunk in ranked_chunks for unit_id in chunk["answer_unit_ids"]
    )


def summarize_retrieval_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    top_k_values = [str(k) for k in config["retrieval"]["top_k_values"]]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["config_id"], record["search_scope"])].append(record)

    configs: dict[str, Any] = {}
    for (config_id, search_scope), group_records in sorted(grouped.items()):
        configs.setdefault(config_id, {})[search_scope] = summarize_record_group(
            group_records,
            top_k_values,
        )

    return {
        "top_k_values": config["retrieval"]["top_k_values"],
        "primary_top_k": config["retrieval"]["primary_top_k"],
        "record_count": len(records),
        "configs": configs,
    }


def summarize_record_group(
    records: list[dict[str, Any]],
    top_k_values: list[str],
) -> dict[str, Any]:
    return {
        "query_count": len(records),
        "macro": {
            k: mean_metric_dict([record["metrics_by_k"][k] for record in records])
            for k in top_k_values
        },
        "efficiency": summarize_retrieval_timings(records),
        "by_specificity": summarize_by_field(records, top_k_values, "specificity"),
        "by_target_document": summarize_by_target_document(records, top_k_values),
    }


def summarize_retrieval_timings(
    records: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Summarize query embedding and FAISS search latency."""
    timed = [
        record
        for record in records
        if record.get("query_embedding_seconds") is not None
        and record.get("search_seconds") is not None
    ]
    if not timed:
        return None
    embedding = [record["query_embedding_seconds"] for record in timed]
    search = [record["search_seconds"] for record in timed]
    retrieval_side = [e + s for e, s in zip(embedding, search)]
    return {
        "query_count": len(timed),
        "query_embedding_seconds": timing_summary(embedding),
        "search_seconds": timing_summary(search),
        "retrieval_side_seconds": timing_summary(retrieval_side),
    }


def timing_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": sum(values) / len(values),
        "median": percentile(values, 50),
        "p95": percentile(values, 95),
        "total": sum(values),
    }


def summarize_by_field(
    records: list[dict[str, Any]],
    top_k_values: list[str],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record[field]].append(record)
    return {
        value: {
            "query_count": len(value_records),
            "metrics": {
                k: mean_metric_dict(
                    [record["metrics_by_k"][k] for record in value_records]
                )
                for k in top_k_values
            },
        }
        for value, value_records in sorted(grouped.items())
    }


def summarize_by_target_document(
    records: list[dict[str, Any]],
    top_k_values: list[str],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for document_code in record["target_document_codes"]:
            grouped[document_code].append(record)
    return {
        document_code: {
            "query_count": len(document_records),
            "metrics": {
                k: mean_metric_dict(
                    [record["metrics_by_k"][k] for record in document_records]
                )
                for k in top_k_values
            },
        }
        for document_code, document_records in sorted(grouped.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 4 retrieval evaluation.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=Path("data/dataset/q4eu_queries.json"),
    )
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--indices-dir", type=Path, default=Path("data/indices"))
    parser.add_argument(
        "--retrieval-results-path",
        type=Path,
        default=Path("data/retrieval/retrieval_results.json"),
    )
    parser.add_argument(
        "--retrieval-metrics-path",
        type=Path,
        default=Path("data/evaluation/retrieval_metrics.json"),
    )
    parser.add_argument(
        "--recompute-from-results",
        action="store_true",
        help=(
            "Re-score the stored retrieval results with the current metric "
            "definitions instead of re-running retrieval (no API calls)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.recompute_from_results:
        records, _ = recompute_metrics_from_results(
            config_path=args.config_path,
            retrieval_results_path=args.retrieval_results_path,
            retrieval_metrics_path=args.retrieval_metrics_path,
        )
        print(f"Recomputed metrics for {len(records)} stored records.")
        return
    records, _ = write_retrieval_outputs(
        config_path=args.config_path,
        queries_path=args.queries_path,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        indices_dir=args.indices_dir,
        retrieval_results_path=args.retrieval_results_path,
        retrieval_metrics_path=args.retrieval_metrics_path,
    )
    print(f"Stage 4 retrieval evaluation complete: {len(records)} records.")


if __name__ == "__main__":
    main()
