import sys
import unittest
from math import log2
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.ir_metrics import compute_ir_metrics
from retrieval.evaluate_retrieval import (
    ranked_answer_units_from_chunks,
    scope_allowed_document_codes,
    scope_includes_query,
    search_chunks,
    summarize_retrieval_timings,
)
from retrieval.index_corpus import l2_normalize


class RetrievalEvaluationTests(unittest.TestCase):
    def test_ir_metrics_follow_standard_definitions(self) -> None:
        metrics = compute_ir_metrics(
            ranked_answer_units=["G Art. 1", "G Art. 2", "G Art. 3"],
            gold_unit_ids=["G Art. 2", "G Art. 4"],
        )

        self.assertAlmostEqual(metrics["precision"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 1 / 2)
        self.assertAlmostEqual(metrics["mrr"], 1 / 2)
        self.assertGreater(metrics["ndcg"], 0)
        self.assertAlmostEqual(metrics["r_precision"], 1 / 2)

    def test_r_precision_uses_query_adaptive_cutoff_not_fixed_depth(self) -> None:
        units = [f"G Art. {number}" for number in range(1, 7)]
        metrics = compute_ir_metrics(
            ranked_answer_units=units,
            gold_unit_ids=["G Art. 5"],
        )
        self.assertAlmostEqual(metrics["precision"], 1 / 6)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["r_precision"], 0.0)

    def test_ndcg_ideal_uses_min_of_gold_and_retrieved(self) -> None:
        metrics = compute_ir_metrics(
            ranked_answer_units=["G Art. 1", "G Art. 2"],
            gold_unit_ids=["G Art. 1", "G Art. 2", "G Art. 3"],
        )

        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 2 / 3)
        self.assertAlmostEqual(metrics["ndcg"], 1.0)

    def test_all_metrics_score_the_full_derived_ranking(self) -> None:
        units = [f"G Art. {number}" for number in range(1, 7)]
        metrics = compute_ir_metrics(
            ranked_answer_units=units,
            gold_unit_ids=["G Art. 5"],
        )

        self.assertAlmostEqual(metrics["precision"], 1 / 6)
        self.assertAlmostEqual(metrics["recall"], 1.0)
        self.assertAlmostEqual(metrics["mrr"], 1 / 5)
        self.assertAlmostEqual(metrics["ndcg"], (1 / log2(6)) / 1.0)

    def test_ranked_answer_units_dedupes_first_seen(self) -> None:
        ranked = ranked_answer_units_from_chunks(
            [
                {"answer_unit_ids": ["G Art. 1", "G Art. 2"]},
                {"answer_unit_ids": ["G Art. 2", "G Art. 3"]},
            ]
        )

        self.assertEqual(ranked, ["G Art. 1", "G Art. 2", "G Art. 3"])

    def test_target_scope_filters_to_target_documents(self) -> None:
        vectors = l2_normalize(
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            )
        )
        index = faiss.IndexFlatIP(2)
        index.add(vectors)
        chunk_set = {
            "index": index,
            "chunks": [
                chunk("B_chunk", "B"),
                chunk("G_chunk_1", "G"),
                chunk("G_chunk_2", "G"),
            ],
        }

        ranked = search_chunks(
            chunk_set=chunk_set,
            query_embedding=l2_normalize(np.asarray([[1.0, 0.0]], dtype=np.float32))[0],
            allowed_document_codes={"G"},
            max_k=2,
        )

        self.assertEqual([item["document_code"] for item in ranked], ["G", "G"])

    def test_scope_restriction_resolution_per_scope_kind(self) -> None:
        query = {"target_document_codes": ["G"]}
        all_acts = {"id": "all_acts", "restrict_to_target_documents": False}
        target_acts = {"id": "target_acts", "restrict_to_target_documents": True}
        pil_acts = {
            "id": "pil_acts",
            "restrict_to_document_codes": ["B", "RI", "RII"],
            "only_queries_targeting_restriction": True,
        }

        self.assertIsNone(scope_allowed_document_codes(all_acts, query))
        self.assertEqual(scope_allowed_document_codes(target_acts, query), {"G"})
        self.assertEqual(
            scope_allowed_document_codes(pil_acts, query), {"B", "RI", "RII"}
        )

    def test_fixed_restriction_scope_runs_only_for_matching_queries(self) -> None:
        pil_acts = {
            "id": "pil_acts",
            "restrict_to_document_codes": ["B", "RI", "RII"],
            "only_queries_targeting_restriction": True,
        }
        all_acts = {"id": "all_acts", "restrict_to_target_documents": False}
        pil_query = {"target_document_codes": ["RI"]}
        multi_pil_query = {"target_document_codes": ["B", "RII"]}
        gdpr_query = {"target_document_codes": ["G"]}
        mixed_query = {"target_document_codes": ["B", "G"]}

        self.assertTrue(scope_includes_query(pil_acts, pil_query))
        self.assertTrue(scope_includes_query(pil_acts, multi_pil_query))
        self.assertFalse(scope_includes_query(pil_acts, gdpr_query))
        self.assertFalse(scope_includes_query(pil_acts, mixed_query))
        self.assertTrue(scope_includes_query(all_acts, gdpr_query))

    def test_retrieval_timing_summary_covers_embedding_search_and_sum(self) -> None:
        records = [
            {"query_embedding_seconds": 0.2, "search_seconds": 0.01},
            {"query_embedding_seconds": 0.4, "search_seconds": 0.03},
        ]

        summary = summarize_retrieval_timings(records)

        self.assertEqual(summary["query_count"], 2)
        self.assertAlmostEqual(summary["query_embedding_seconds"]["mean"], 0.3)
        self.assertAlmostEqual(summary["search_seconds"]["mean"], 0.02)
        self.assertAlmostEqual(summary["retrieval_side_seconds"]["mean"], 0.32)
        self.assertAlmostEqual(summary["retrieval_side_seconds"]["total"], 0.64)

    def test_retrieval_timing_summary_is_none_for_untimed_records(self) -> None:
        self.assertIsNone(summarize_retrieval_timings([{"query_id": "q1"}]))


def chunk(chunk_id: str, document_code: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "metadata": {
            "document_code": document_code,
            "answer_unit_ids": [f"{document_code} Art. 1"],
        },
    }


if __name__ == "__main__":
    unittest.main()
