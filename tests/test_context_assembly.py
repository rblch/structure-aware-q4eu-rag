from __future__ import annotations

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from enrichment.context_assembly import (
    apply_retrieval_conditioned_xref_eligibility,
    assemble_context,
    assert_context_budget_matches_derivation,
    assert_extended_rankings_match_base,
    assert_volume_match_within_tolerance,
    build_context_records,
    build_edges_by_source,
    condition_candidates,
    summarize_volume_match,
)


class ContextAssemblyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chunks_by_id = {
            "hier_paragraph_X_Art_1_1": chunk(
                "hier_paragraph_X_Art_1_1",
                ["X Art. 1.1"],
                ["X Art. 1"],
                parent_chunk_id="hier_article_X_Art_1",
                text="Paragraph one.",
            ),
            "hier_paragraph_X_Art_2_1": chunk(
                "hier_paragraph_X_Art_2_1",
                ["X Art. 2.1"],
                ["X Art. 2"],
                parent_chunk_id="hier_article_X_Art_2",
                text="Paragraph two.",
            ),
            "hier_paragraph_X_Art_4_1": chunk(
                "hier_paragraph_X_Art_4_1",
                ["X Art. 4.1"],
                ["X Art. 4"],
                parent_chunk_id="hier_article_X_Art_4",
                text="Extra ranked paragraph.",
            ),
            "hier_article_X_Art_1": chunk(
                "hier_article_X_Art_1",
                ["X Art. 1"],
                ["X Art. 1"],
                text="Article one parent.",
            ),
            "hier_article_X_Art_2": chunk(
                "hier_article_X_Art_2",
                ["X Art. 2"],
                ["X Art. 2"],
                text="Article two parent.",
            ),
            "hier_article_X_Art_3": chunk(
                "hier_article_X_Art_3",
                ["X Art. 3"],
                ["X Art. 3"],
                text="Referenced article three.",
            ),
            "hier_article_X_Art_4": chunk(
                "hier_article_X_Art_4",
                ["X Art. 4"],
                ["X Art. 4"],
                text="Article four parent.",
            ),
        }
        self.base_record = {
            "query_id": "q1",
            "retrieved_chunks": [
                {
                    "chunk_rank": 1,
                    "chunk_id": "hier_paragraph_X_Art_1_1",
                    "score": 0.9,
                },
                {
                    "chunk_rank": 2,
                    "chunk_id": "hier_paragraph_X_Art_2_1",
                    "score": 0.8,
                },
            ],
        }
        self.extended_ranking = [
            *self.base_record["retrieved_chunks"],
            {
                "chunk_rank": 3,
                "chunk_id": "hier_paragraph_X_Art_4_1",
                "score": 0.7,
            },
        ]
        self.xref_graph = {
            "edges": [
                {
                    "source_unit_id": "X Art. 1.1",
                    "target_unit_id": "X Art. 3",
                    "target_unit_id_normalized": "X Art. 3",
                    "document_code": "X",
                    "reference_kind": "single",
                    "raw_match": "Article 3",
                }
            ]
        }
        self.query = {
            "query_id": "q1",
            "question": "Question?",
            "specificity": "H",
            "target_document_codes": ["X"],
            "gold_unit_ids": ["X Art. 1", "X Art. 3"],
        }

    def test_parent_and_xref_candidates_preserve_base_first(self) -> None:
        candidates = condition_candidates(
            condition={
                "id": "combined",
                "parent": True,
                "xref": True,
                "volume_matched": False,
            },
            base_candidates=[
                {
                    "chunk_id": "hier_paragraph_X_Art_1_1",
                    "role": "retrieved",
                    "retrieval_rank": 1,
                    "retrieval_score": 0.9,
                }
            ],
            extended_ranking=[],
            chunks_by_id=self.chunks_by_id,
            edges_by_source=build_edges_by_source(self.xref_graph),
        )

        self.assertEqual(
            [candidate["chunk_id"] for candidate in candidates],
            [
                "hier_paragraph_X_Art_1_1",
                "hier_article_X_Art_1",
                "hier_article_X_Art_3",
            ],
        )
        self.assertEqual(
            [candidate["role"] for candidate in candidates],
            ["retrieved", "parent", "xref"],
        )

    def test_context_records_include_volume_matched_control(self) -> None:
        records, uncapped_combined_tokens = build_context_records(
            queries=[self.query],
            conditions=[
                {
                    "id": "combined",
                    "parent": True,
                    "xref": True,
                    "volume_matched": False,
                },
                {
                    "id": "volume_matched",
                    "parent": False,
                    "xref": False,
                    "volume_matched": True,
                },
            ],
            chunks_by_id=self.chunks_by_id,
            base_records={"q1": self.base_record},
            extended_rankings={"q1": self.extended_ranking},
            edges_by_source=build_edges_by_source(self.xref_graph),
            base_config_id="hier_paragraph",
            base_top_k=2,
            context_budget_tokens=1000,
            volume_match_tolerance=0.05,
        )

        by_condition = {record["condition_id"]: record for record in records}
        combined = by_condition["combined"]
        volume = by_condition["volume_matched"]

        self.assertIn("X Art. 3", combined["context_answer_unit_ids"])
        self.assertIn("X Art. 4", volume["context_answer_unit_ids"])
        self.assertIn(
            "volume_retrieval",
            {chunk["role"] for chunk in volume["included_chunks"]},
        )
        self.assertNotIn(
            "parent",
            {chunk["role"] for chunk in volume["included_chunks"]},
        )
        self.assertNotIn("xref", {chunk["role"] for chunk in volume["included_chunks"]})
        self.assertNotIn("combined", combined["context_text"])
        self.assertIn("q1", uncapped_combined_tokens)
        self.assertGreaterEqual(
            uncapped_combined_tokens["q1"],
            combined["context_token_count"],
        )

    def _volume_pool(self, texts: list[str]) -> tuple[list[dict], dict[str, dict]]:
        chunks_by_id = {
            f"vol_{index}": chunk(
                f"vol_{index}", [f"X Art. {index + 10}.1"], [f"X Art. {index + 10}"],
                text=text,
            )
            for index, text in enumerate(texts)
        }
        candidates = [
            {"chunk_id": chunk_id, "role": "volume_retrieval"}
            for chunk_id in chunks_by_id
        ]
        return candidates, chunks_by_id

    def _assemble_volume(
        self,
        candidates: list[dict],
        chunks_by_id: dict[str, dict],
        *,
        target: int | None,
        tolerance: float | None = 0.05,
        budget: int = 4500,
    ) -> dict:
        return assemble_context(
            query=self.query,
            condition_id="volume_matched",
            candidates=candidates,
            chunks_by_id=chunks_by_id,
            context_budget_tokens=budget,
            base_config_id="hier_paragraph",
            base_top_k=2,
            stop_at_tokens=target,
            minimum_included_before_stop=0,
            volume_match_tolerance=tolerance if target is not None else None,
        )

    def test_volume_exact_match_records_zero_delta(self) -> None:
        candidates, chunks_by_id = self._volume_pool(["one two", "three four", "five"])
        full = self._assemble_volume(candidates, chunks_by_id, target=None)
        target = full["context_token_count"]

        record = self._assemble_volume(candidates, chunks_by_id, target=target)

        self.assertEqual(record["volume_target_tokens"], target)
        self.assertEqual(record["volume_token_delta"], 0)
        self.assertEqual(record["volume_relative_error"], 0.0)
        self.assertTrue(record["volume_target_met"])

    def test_volume_matching_chooses_closer_prefix_below_target(self) -> None:
        candidates, chunks_by_id = self._volume_pool(
            ["one two", "three four", "five six seven eight nine ten"]
        )
        two_chunks = self._assemble_volume(candidates[:2], chunks_by_id, target=None)
        target = two_chunks["context_token_count"] + 1

        record = self._assemble_volume(candidates, chunks_by_id, target=target)

        self.assertEqual(record["volume_token_delta"], -1)
        self.assertTrue(record["volume_target_met"])
        self.assertEqual(
            record["excluded_chunks"][0]["excluded_reason"],
            "volume_target_closest_prefix",
        )

    def test_volume_matching_keeps_closer_prefix_above_target(self) -> None:
        candidates, chunks_by_id = self._volume_pool(
            ["one two", "three four", "five six seven eight nine ten"]
        )
        full = self._assemble_volume(candidates, chunks_by_id, target=None)
        target = full["context_token_count"] - 1

        record = self._assemble_volume(candidates, chunks_by_id, target=target)

        self.assertEqual(record["volume_token_delta"], 1)
        self.assertTrue(record["volume_target_met"])

    def test_volume_matching_tie_keeps_crossing_chunk(self) -> None:
        # Registered tie rule (spec Section 5 revision note / Stage 5): when
        # the prefixes below and above the target are equidistant, keep the
        # crossing chunk. "five six seven eight nine" makes the third chunk's
        # marginal cost even (32 tokens), so an exactly equidistant target
        # exists.
        candidates, chunks_by_id = self._volume_pool(
            ["one two", "three four", "five six seven eight nine"]
        )
        two_chunks = self._assemble_volume(candidates[:2], chunks_by_id, target=None)
        full = self._assemble_volume(candidates, chunks_by_id, target=None)
        marginal = full["context_token_count"] - two_chunks["context_token_count"]
        self.assertEqual(marginal % 2, 0)
        target = two_chunks["context_token_count"] + marginal // 2

        record = self._assemble_volume(
            candidates, chunks_by_id, target=target, tolerance=0.5
        )

        self.assertEqual(record["included_chunk_count"], 3)
        self.assertEqual(record["volume_token_delta"], marginal // 2)

    def test_volume_pool_exhaustion_records_shortfall(self) -> None:
        candidates, chunks_by_id = self._volume_pool(["one two", "three four"])
        full = self._assemble_volume(candidates, chunks_by_id, target=None)
        target = full["context_token_count"] + 500

        record = self._assemble_volume(candidates, chunks_by_id, target=target)

        self.assertLess(record["volume_token_delta"], 0)
        self.assertFalse(record["volume_target_met"])

    def test_volume_oversized_candidate_is_skipped_not_terminal(self) -> None:
        big_text = " ".join(f"word{i}" for i in range(200))
        candidates, chunks_by_id = self._volume_pool(["one two", big_text, "three"])
        without_big = self._assemble_volume(
            [candidates[0], candidates[2]], chunks_by_id, target=None
        )
        budget = without_big["context_token_count"]

        record = self._assemble_volume(
            candidates, chunks_by_id, target=budget, budget=budget
        )

        included_ids = {chunk["chunk_id"] for chunk in record["included_chunks"]}
        excluded = {
            chunk["chunk_id"]: chunk["excluded_reason"]
            for chunk in record["excluded_chunks"]
        }
        self.assertEqual(excluded.get("vol_1"), "context_budget")
        self.assertIn("vol_2", included_ids)
        self.assertTrue(record["volume_target_met"])

    def test_volume_tolerance_boundary_is_inclusive(self) -> None:
        candidates, chunks_by_id = self._volume_pool(["one two", "three four"])
        full = self._assemble_volume(candidates, chunks_by_id, target=None)
        actual = full["context_token_count"]
        target = actual + 4
        boundary = abs(actual - target) / target

        at_boundary = self._assemble_volume(
            candidates, chunks_by_id, target=target, tolerance=boundary
        )
        below_boundary = self._assemble_volume(
            candidates, chunks_by_id, target=target, tolerance=boundary * 0.99
        )

        self.assertTrue(at_boundary["volume_target_met"])
        self.assertFalse(below_boundary["volume_target_met"])

    def test_volume_summary_and_enforcement(self) -> None:
        met = {
            "query_id": "q1", "volume_target_tokens": 100,
            "context_token_count": 100, "volume_token_delta": 0,
            "volume_relative_error": 0.0, "volume_target_met": True,
            "condition_id": "volume_matched",
        }
        failed = {
            "query_id": "q2", "volume_target_tokens": 100,
            "context_token_count": 35, "volume_token_delta": -65,
            "volume_relative_error": 0.65, "volume_target_met": False,
            "condition_id": "volume_matched",
        }

        summary = summarize_volume_match([met, failed], tolerance=0.05)
        self.assertEqual(summary["outside_tolerance_count"], 1)
        self.assertEqual(summary["max_relative_error"], 0.65)
        self.assertEqual(
            summary["outside_tolerance_queries"][0]["query_id"], "q2"
        )
        self.assertEqual(
            summary["outside_tolerance_queries"][0]["volume_token_delta"], -65
        )

        with self.assertRaisesRegex(RuntimeError, "q2"):
            assert_volume_match_within_tolerance(summary, tolerance=0.05)
        assert_volume_match_within_tolerance(
            summarize_volume_match([met], tolerance=0.05), tolerance=0.05
        )
        assert_volume_match_within_tolerance(None, tolerance=0.05)

    def test_context_budget_must_match_registered_derivation(self) -> None:
        matching = {
            "p95_raw": 4321.0,
            "rounding": "nearest_500",
            "derived_budget": 4500,
            "configured_budget": 4500,
            "matches_configured_budget": True,
        }
        assert_context_budget_matches_derivation(matching)

        mismatching = {
            **matching,
            "derived_budget": 5000,
            "matches_configured_budget": False,
        }
        with self.assertRaises(RuntimeError) as raised:
            assert_context_budget_matches_derivation(mismatching)

        message = str(raised.exception)
        self.assertIn("configured 4500", message)
        self.assertIn("p95 is 4321.0", message)
        self.assertIn("yielding 5000", message)
        self.assertIn("No downstream-ready Stage 5 contexts were written", message)
        self.assertIn("pre-generation amendment", message)

    def test_extended_ranking_must_share_the_base_top_k_prefix(self) -> None:
        base_records = {"q1": self.base_record}

        assert_extended_rankings_match_base(
            base_records,
            {"q1": self.extended_ranking},
            base_top_k=5,
        )

        reordered = list(reversed(self.extended_ranking))
        with self.assertRaises(RuntimeError) as raised:
            assert_extended_rankings_match_base(
                base_records,
                {"q1": reordered},
                base_top_k=5,
            )
        self.assertIn("q1", str(raised.exception))

        with self.assertRaises(RuntimeError):
            assert_extended_rankings_match_base(
                base_records,
                {},
                base_top_k=5,
            )

    def test_retrieval_conditioned_xref_eligibility_uses_top_k_edges(self) -> None:
        updated_queries, report = apply_retrieval_conditioned_xref_eligibility(
            queries=[self.query],
            xref_graph=self.xref_graph,
            base_records={"q1": self.base_record},
            chunks_by_id=self.chunks_by_id,
            base_config_id="hier_paragraph",
            base_top_k=2,
            existing_report={"structural_upper_bound_count": 1},
        )

        self.assertTrue(updated_queries[0]["xref_eligible"])
        self.assertEqual(report["stage"], "1c-ii")
        self.assertEqual(report["retrieval_conditioned_count"], 1)
        self.assertEqual(
            report["xref_power_decision"],
            "xref_contrasts_exploratory_underpowered",
        )


def chunk(
    chunk_id: str,
    source_unit_ids: list[str],
    answer_unit_ids: list[str],
    *,
    text: str,
    parent_chunk_id: str | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "token_count": 4,
        "metadata": {
            "document_code": "X",
            "source_unit_ids": source_unit_ids,
            "answer_unit_ids": answer_unit_ids,
            "parent_chunk_id": parent_chunk_id,
        },
    }


if __name__ == "__main__":
    unittest.main()
