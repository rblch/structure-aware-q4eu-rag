import math
import sys
import unittest
from pathlib import Path

from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.reporting import (
    answer_quality_table,
    anti_gaming_gate_table,
    bootstrap_ci,
    bootstrap_settings_from_config,
    external_baseline_modified_ir_table,
    external_baseline_q4pil_table,
    external_baseline_table,
    friedman_test,
    holm_adjusted_p_values,
    paired_retrieval_differences,
    pipeline_latency_by_condition,
    released_script_modified_ir_metrics,
    rq1_confirmatory_config_ids_from_config,
    rq1_inferential_table,
    rq2_inferential_table,
    wilcoxon_signed_rank,
)


class ReportingTests(unittest.TestCase):
    def test_external_baseline_table_uses_standard_metrics_for_our_rows(self) -> None:
        macro = {
            "10": {
                "precision": 0.2,
                "recall": 0.6,
                "f1": 0.3,
                "mrr": 0.7,
                "ndcg": 0.5,
            }
        }
        rows = external_baseline_table(
            {
                "configs": {
                    config_id: {
                        "all_acts": {"macro": macro},
                        "target_acts": {"macro": macro},
                    }
                    for config_id in (
                        "fs_64_12",
                        "fs_256_50",
                        "sem_50_64",
                        "sem_50_256",
                        "hier_paragraph",
                    )
                }
            },
            config_ids=["fs_256_50", "sem_50_256", "hier_paragraph"],
        )

        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["metric_basis"], "standard_ir_on_pinned_dataset")
        self.assertAlmostEqual(rows[0]["recall"], 0.6)

        all_acts_rows = [row for row in rows if row["search_scope"] == "all_acts"]
        self.assertEqual(rows[0]["search_scope"], "all_acts")
        all_acts_anchor = all_acts_rows[-1]
        self.assertEqual(
            all_acts_anchor["metric_basis"],
            "paper_reported_released_script_convention",
        )
        self.assertAlmostEqual(all_acts_anchor["precision"], 0.688)
        self.assertAlmostEqual(all_acts_anchor["f1"], 0.380)

        target_acts_anchor = [
            row
            for row in rows
            if row["search_scope"] == "target_acts" and row["source"] == "paper"
        ][0]
        self.assertAlmostEqual(target_acts_anchor["precision"], 0.726)
        self.assertAlmostEqual(target_acts_anchor["f1"], 0.413)
        self.assertAlmostEqual(target_acts_anchor["ndcg"], 0.506)
        self.assertAlmostEqual(target_acts_anchor["mrr"], 0.755)

    def test_modified_ir_table_is_scope_matched_primary_comparison(self) -> None:
        results = [
            {
                "config_id": config_id,
                "search_scope": scope,
                "gold_unit_ids": ["G Art. 1", "G Art. 2"],
                "retrieved_chunks": [
                    {"answer_unit_ids": ["G Art. 1"]},
                    {"answer_unit_ids": ["G Art. 2"]},
                ],
            }
            for config_id in ("fs_256_50", "sem_50_256", "hier_paragraph")
            for scope in ("all_acts", "target_acts")
        ]
        rows = external_baseline_modified_ir_table(
            results,
            config_ids=["fs_256_50", "sem_50_256", "hier_paragraph"],
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["search_scope"], "all_acts")
        all_anchor = [
            r for r in rows if r["search_scope"] == "all_acts" and r["source"] == "paper"
        ][0]
        tgt_anchor = [
            r
            for r in rows
            if r["search_scope"] == "target_acts" and r["source"] == "paper"
        ][0]
        self.assertAlmostEqual(all_anchor["modified_precision"], 0.688)
        self.assertAlmostEqual(tgt_anchor["modified_precision"], 0.726)
        self.assertEqual(rows[0]["metric_basis"], "released_evaluate_py_modified_ir")

    def test_q4pil_table_scores_pil_scope_at_top5_beside_published_anchor(
        self,
    ) -> None:
        results = [
            {
                "config_id": config_id,
                "search_scope": scope,
                "gold_unit_ids": ["RI Art. 3", "B Art. 7"],
                "retrieved_chunks": [
                    {"answer_unit_ids": ["RI Art. 3"]},
                    {"answer_unit_ids": ["B Art. 7"]},
                ],
            }
            for config_id in ("fs_256_50", "sem_50_256", "hier_paragraph")
            for scope in ("all_acts", "pil_acts")
        ]
        rows = external_baseline_q4pil_table(
            results,
            config_ids=["fs_256_50", "sem_50_256", "hier_paragraph"],
        )
        self.assertEqual(len(rows), 4)
        self.assertTrue(
            all(row["search_scope"] == "pil_acts" for row in rows)
        )
        self.assertTrue(all(row["top_k"] == 5 for row in rows))
        anchor = [row for row in rows if row["source"] == "paper"][0]
        self.assertAlmostEqual(anchor["modified_precision"], 0.4517)
        self.assertAlmostEqual(anchor["modified_recall"], 0.3758)
        self.assertAlmostEqual(anchor["modified_f1"], 0.3805)
        study = [row for row in rows if row["source"] == "this_experiment"]
        self.assertEqual(
            {row["metric_basis"] for row in study},
            {"released_evaluate_py_modified_ir"},
        )
        self.assertNotIn("modified_ndcg", rows[0])
        self.assertNotIn("modified_mrr", rows[0])

    def test_q4pil_table_emits_anchor_even_without_pil_scope_records(self) -> None:
        rows = external_baseline_q4pil_table(
            [
                {
                    "config_id": "fs_256_50",
                    "search_scope": "all_acts",
                    "gold_unit_ids": ["G Art. 1"],
                    "retrieved_chunks": [{"answer_unit_ids": ["G Art. 1"]}],
                }
            ],
            config_ids=["fs_256_50"],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["source"], "paper")

    def test_released_script_modified_ir_uses_capped_gold_denominator(self) -> None:
        metrics = released_script_modified_ir_metrics(
            record={
                "gold_unit_ids": ["G Art. 1", "G Art. 2", "G Art. 3"],
                "retrieved_chunks": [
                    {"answer_unit_ids": ["G Art. 1", "G Art. 99"]},
                    {"answer_unit_ids": ["G Art. 1", "G Art. 2"]},
                    {"answer_unit_ids": ["G Art. 3"]},
                ],
            },
            top_k=4,
        )

        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["recall"], 3 / 4)
        self.assertAlmostEqual(metrics["f1"], 12 / 17)
        self.assertAlmostEqual(metrics["mrr"], 1.0)
        self.assertGreater(metrics["ndcg"], 0.8)

    def test_paired_retrieval_differences_aligns_records_by_query(self) -> None:
        rows = paired_retrieval_differences(
            [
                retrieval_record("hier_paragraph", "q1", 0.75),
                retrieval_record("fs_256_50", "q1", 0.5),
                retrieval_record("sem_50_256", "q1", 0.625),
                retrieval_record("hier_paragraph", "q2", 0.25),
            ],
            search_scope="target_acts",
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            [(row["baseline_config_id"], row["delta_recall"]) for row in rows],
            [("fs_256_50", 0.25), ("sem_50_256", 0.125)],
        )

    def test_rq1_confirmatory_table_uses_f1_as_primary_outcome(self) -> None:
        rows = [
            retrieval_query_row("fs_64_12", "q1", f1=0.6, recall=0.7),
            retrieval_query_row("fs_256_50", "q1", f1=0.2, recall=1.0),
            retrieval_query_row("sem_50_64", "q1", f1=0.7, recall=0.7),
            retrieval_query_row("sem_50_256", "q1", f1=0.4, recall=0.8),
            retrieval_query_row("hier_paragraph", "q1", f1=0.8, recall=0.6),
            retrieval_query_row(
                "hier_paragraph_contextualized",
                "q1",
                f1=1.0,
                recall=1.0,
            ),
            retrieval_query_row("fs_64_12", "q2", f1=0.6, recall=0.7),
            retrieval_query_row("fs_256_50", "q2", f1=0.3, recall=1.0),
            retrieval_query_row("sem_50_64", "q2", f1=0.7, recall=0.7),
            retrieval_query_row("sem_50_256", "q2", f1=0.5, recall=0.8),
            retrieval_query_row("hier_paragraph", "q2", f1=0.9, recall=0.6),
            retrieval_query_row(
                "hier_paragraph_contextualized",
                "q2",
                f1=1.0,
                recall=1.0,
            ),
        ]

        stats = rq1_inferential_table(rows)

        self.assertEqual({row["outcome"] for row in stats}, {"gold_f1_at_10"})
        self.assertEqual(
            stats[0]["comparison"],
            "fs_256_50 vs sem_50_256 vs hier_paragraph",
        )
        hierarchy_vs_fixed = next(
            row
            for row in stats
            if row["comparison"] == "hier_paragraph - fs_256_50"
        )
        self.assertAlmostEqual(hierarchy_vs_fixed["mean_difference"], 0.6)

    def test_rq1_confirmatory_configs_read_config_flags(self) -> None:
        config = {
            "chunking": {
                "fixed_size": {
                    "configs": [
                        {"id": "fs_64_12", "confirmatory": False},
                        {"id": "fs_256_50", "confirmatory": True},
                    ]
                },
                "semantic": {
                    "configs": [
                        {"id": "sem_70_64", "confirmatory": False},
                        {"id": "sem_50_256", "confirmatory": True},
                    ]
                },
                "hierarchical": {
                    "configs": [
                        {"id": "hier_subparagraph", "confirmatory": False},
                        {"id": "hier_paragraph", "confirmatory": True},
                    ]
                },
            }
        }

        self.assertEqual(
            rq1_confirmatory_config_ids_from_config(config),
            ["fs_256_50", "sem_50_256", "hier_paragraph"],
        )

    def test_anti_gaming_gate_blocks_gamed_faithfulness(self) -> None:
        def record(condition, query_id, faithfulness, correctness, unjustified):
            return {
                "condition_id": condition,
                "query_id": query_id,
                "faithfulness_score": faithfulness,
                "correctness_score": correctness,
                "either_judge_unjustified_abstention": unjustified,
            }

        records = []
        for index in range(4):
            records.append(record("no_enrichment", f"q{index}", 0.5, 0.5, False))
            records.append(record("parent_only", f"q{index}", 0.7, 0.6, False))
            records.append(record("combined", f"q{index}", 0.8, 0.4, index < 2))

        rows = anti_gaming_gate_table(
            records,
            contrasts=[
                ("parent_only", "no_enrichment"),
                ("combined", "no_enrichment"),
                ("missing_condition", "no_enrichment"),
            ],
            max_unjustified_abstention_increase=0.10,
        )
        by_contrast = {row["contrast"]: row for row in rows}

        honest = by_contrast["parent_only - no_enrichment"]
        self.assertTrue(honest["faithfulness_improves"])
        self.assertTrue(honest["correctness_improves"])
        self.assertTrue(honest["either_judge_unjustified_abstention_within_tolerance"])
        self.assertTrue(honest["interpretation_gate_passed"])

        gamed = by_contrast["combined - no_enrichment"]
        self.assertTrue(gamed["faithfulness_improves"])
        self.assertFalse(gamed["correctness_improves"])
        self.assertFalse(gamed["either_judge_unjustified_abstention_within_tolerance"])
        self.assertFalse(gamed["interpretation_gate_passed"])

        self.assertNotIn("missing_condition - no_enrichment", by_contrast)

    def test_rq2_confirmatory_holm_family_excludes_xref_exploratory_contrasts(
        self,
    ) -> None:
        records = []
        condition_scores = {
            "no_enrichment": [0.2, 0.2, 0.3, 0.3, 0.4],
            "parent_only": [0.3, 0.4, 0.5, 0.4, 0.5],
            "xref_only": [0.2, 0.3, 0.3, 0.4, 0.4],
            "combined": [0.4, 0.5, 0.6, 0.5, 0.6],
            "volume_matched": [0.3, 0.5, 0.5, 0.5, 0.5],
        }
        for condition_id, scores in condition_scores.items():
            for index, score in enumerate(scores, start=1):
                records.append(answer_record(condition_id, f"q{index}", score))

        rows = [
            row
            for row in rq2_inferential_table(records)
            if row["outcome"] == "faithfulness_score"
        ]
        confirmatory_rows = [
            row for row in rows if row["family"] == "enrichment_confirmatory"
        ]
        exploratory_rows = [row for row in rows if row["family"] == "xref_exploratory"]

        self.assertEqual(
            [row["comparison"] for row in confirmatory_rows],
            [
                "combined - no_enrichment",
                "parent_only - no_enrichment",
            ],
        )
        self.assertEqual(
            [row["comparison"] for row in exploratory_rows],
            [
                "combined - parent_only",
                "xref_only - no_enrichment",
                "combined - xref_only",
            ],
        )

        ungated_rows = [
            row
            for row in rq2_inferential_table(records, xref_gate_triggered=False)
            if row["outcome"] == "faithfulness_score"
        ]
        self.assertEqual(
            [
                row["comparison"]
                for row in ungated_rows
                if row["family"] == "enrichment_confirmatory"
            ],
            [
                "combined - no_enrichment",
                "parent_only - no_enrichment",
                "combined - parent_only",
            ],
        )
        self.assertEqual(
            [
                row["comparison"]
                for row in ungated_rows
                if row["family"] == "xref_exploratory"
            ],
            ["xref_only - no_enrichment", "combined - xref_only"],
        )
        self.assertEqual(
            [
                row["p_value_adjusted_holm"]
                for row in confirmatory_rows
            ],
            holm_adjusted_p_values(
                [row["p_value"] for row in confirmatory_rows]
            ),
        )
        for row in exploratory_rows:
            self.assertEqual(row["p_value_adjusted_holm"], "")
            self.assertIn("outside confirmatory Holm family", row["notes"])

    def test_holm_adjustment_is_monotonic_in_rank_order(self) -> None:
        adjusted = holm_adjusted_p_values([0.04, 0.01, 0.03])

        self.assertEqual(adjusted, [0.06, 0.03, 0.06])

    def test_bootstrap_settings_are_read_from_config(self) -> None:
        settings = bootstrap_settings_from_config(
            {
                "random_seed": 7,
                "analysis": {
                    "bootstrap_iterations": 123,
                    "bootstrap_ci": 0.9,
                },
            }
        )

        self.assertEqual(
            settings,
            {
                "iterations": 123,
                "confidence": 0.9,
                "random_seed": 7,
            },
        )

    def test_bootstrap_ci_uses_explicit_seed(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4]

        first = bootstrap_ci(values, iterations=100, random_seed=7)
        second = bootstrap_ci(values, iterations=100, random_seed=7)
        different_seed = bootstrap_ci(values, iterations=100, random_seed=8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_seed)

    def test_wilcoxon_signed_rank_reports_effect_direction(self) -> None:
        result = wilcoxon_signed_rank([0.3, 0.2, -0.1, 0.0])

        self.assertEqual(result["effective_n"], 3)
        self.assertEqual(result["method"], "scipy.stats.wilcoxon_exact")
        self.assertGreater(result["rank_biserial"], 0)

    def test_wilcoxon_signed_rank_matches_scipy_exact(self) -> None:
        differences = [0.3, 0.2, -0.1]
        expected = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="exact",
        )

        result = wilcoxon_signed_rank(differences)

        self.assertEqual(result["method"], "scipy.stats.wilcoxon_exact")
        self.assertAlmostEqual(result["statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))

    def test_wilcoxon_signed_rank_uses_exact_permutation_for_tied_small_n(self) -> None:
        differences = [0.1] * 12 + [-0.1] * 4
        method = stats.PermutationMethod(n_resamples=math.inf, random_state=42)
        expected = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method=method,
        )

        result = wilcoxon_signed_rank(differences)

        self.assertEqual(
            result["method"],
            "scipy.stats.wilcoxon_permutation_exact_for_ties",
        )
        self.assertAlmostEqual(result["statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))

    def test_friedman_test_matches_scipy(self) -> None:
        aligned = [
            [0.2, 0.3, 0.5],
            [0.1, 0.4, 0.7],
            [0.3, 0.2, 0.6],
        ]
        expected = stats.friedmanchisquare(*zip(*aligned))

        result = friedman_test(aligned)

        self.assertEqual(result["method"], "scipy.stats.friedmanchisquare")
        self.assertAlmostEqual(result["statistic"], float(expected.statistic))
        self.assertAlmostEqual(result["p_value"], float(expected.pvalue))


class AnswerQualityTableTests(unittest.TestCase):
    def test_faithfulness_never_appears_without_anti_gaming_columns(self) -> None:
        generation_metrics = {
            "conditions": {
                "no_enrichment": {
                    "query_count": 1,
                    "faithfulness": {
                        "mean": 0.9,
                        "min": 0.9,
                        "max": 0.9,
                        "unfaithful_response_rate": 0.0,
                        "answer_abstention_rate": 0.25,
                        "unjustified_abstention_rate": 0.125,
                    },
                    "correctness": {"mean": 0.8, "min": 0.8, "max": 0.8},
                    "citation_coverage": {
                        "mean_precision": 0.5,
                        "mean_recall": 0.5,
                        "mean_f1": 0.5,
                    },
                }
            }
        }

        rows = answer_quality_table(generation_metrics, [])

        row = rows[0]
        self.assertIn("faithfulness_mean", row)
        self.assertEqual(row["correctness_mean"], 0.8)
        self.assertEqual(row["unfaithful_response_rate"], 0.0)
        self.assertEqual(row["faithfulness_judge_abstention_rate"], 0.25)
        self.assertEqual(
            row["faithfulness_judge_unjustified_abstention_rate"], 0.125
        )


class PipelineLatencyTests(unittest.TestCase):
    def test_end_to_end_composes_all_deployment_path_stages(self) -> None:
        generation_metrics = {
            "conditions": {"no_enrichment": {}},
            "generation_per_query": {
                "no_enrichment": {"q1": {"elapsed_seconds": 10.0}}
            },
        }
        retrieval_results = [
            {
                "query_id": "q1",
                "config_id": "hier_paragraph",
                "search_scope": "all_acts",
                "query_embedding_seconds": 0.5,
                "search_seconds": 0.01,
            }
        ]
        context_records = [
            {
                "query_id": "q1",
                "condition_id": "no_enrichment",
                "base_config_id": "hier_paragraph",
                "search_scope": "all_acts",
                "assembly_seconds": 0.2,
            }
        ]

        columns = pipeline_latency_by_condition(
            generation_metrics=generation_metrics,
            retrieval_results=retrieval_results,
            context_records=context_records,
        )

        latency = columns["no_enrichment"]
        self.assertAlmostEqual(latency["end_to_end_mean_seconds"], 10.71)
        self.assertAlmostEqual(latency["query_embedding_mean_seconds"], 0.5)
        self.assertAlmostEqual(latency["search_mean_seconds"], 0.01)
        self.assertAlmostEqual(latency["assembly_mean_seconds"], 0.2)
        self.assertEqual(latency["end_to_end_query_count"], 1)

    def test_returns_empty_when_timings_are_missing(self) -> None:
        columns = pipeline_latency_by_condition(
            generation_metrics={"conditions": {}},
            retrieval_results=[{"query_id": "q1", "config_id": "hier_paragraph"}],
            context_records=[],
        )

        self.assertEqual(columns, {})


def retrieval_record(config_id: str, query_id: str, recall: float) -> dict:
    return {
        "config_id": config_id,
        "search_scope": "target_acts",
        "query_id": query_id,
        "specificity": "article",
        "target_document_codes": ["G"],
        "metrics_by_k": {
            "10": {
                "precision": 0.5,
                "recall": recall,
                "f1": 0.4,
                "r_precision": 0.5,
                "mrr": 1.0,
                "ndcg": 0.8,
            }
        },
    }


def retrieval_query_row(
    config_id: str,
    query_id: str,
    *,
    f1: float,
    recall: float,
) -> dict:
    return {
        "config_id": config_id,
        "search_scope": "all_acts",
        "query_id": query_id,
        "top_k": 10,
        "f1": f1,
        "recall": recall,
    }


def answer_record(condition_id: str, query_id: str, score: float) -> dict:
    return {
        "condition_id": condition_id,
        "query_id": query_id,
        "faithfulness_score": score,
    }


if __name__ == "__main__":
    unittest.main()
