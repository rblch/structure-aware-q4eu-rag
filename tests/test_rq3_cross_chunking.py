from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.rq3_cross_chunking import paired_contrasts, summarize_strategies
from enrichment.rq3_context_assembly import (
    assert_rq3_context_records,
    validate_rq3_config,
)
from generation.answer_generation import assert_rq3_generation_checkpoint


class Rq3CrossChunkingTests(unittest.TestCase):
    def test_context_checkpoint_accepts_exact_144_record_design(self) -> None:
        config = sample_config()
        rq3_config = validate_rq3_config(config)
        records = sample_context_records(rq3_config)
        assert_rq3_context_records(records, rq3_config)

        serialized = json.dumps(records, indent=2) + "\n"
        metrics = {
            "checkpoint_type": "rq3_cross_chunking",
            "ready_for_generation": True,
            "rq3_config": rq3_config,
            "record_count": 144,
            "context_records_sha256": hashlib.sha256(
                serialized.encode("utf-8")
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            records_path = Path(temp_dir) / "records.json"
            metrics_path = Path(temp_dir) / "metrics.json"
            records_path.write_text(serialized, encoding="utf-8")
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            assert_rq3_generation_checkpoint(
                config=config,
                context_records=records,
                context_records_path=records_path,
                context_metrics_path=metrics_path,
            )

            metrics["record_count"] = 143
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "generation blocked"):
                assert_rq3_generation_checkpoint(
                    config=config,
                    context_records=records,
                    context_records_path=records_path,
                    context_metrics_path=metrics_path,
                )

    def test_context_validation_rejects_budget_exclusion(self) -> None:
        rq3_config = validate_rq3_config(sample_config())
        records = sample_context_records(rq3_config)
        records[0]["included_chunk_count"] = 4
        records[0]["excluded_chunk_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "excluded a top-k chunk"):
            assert_rq3_context_records(records, rq3_config)

    def test_analysis_summarizes_paired_differences(self) -> None:
        rows = []
        for query_id in ("q1", "q2"):
            for strategy, value in (
                ("fixed_size", 3.0),
                ("semantic", 2.0),
                ("hierarchical", 1.0),
            ):
                row = {"query_id": query_id, "strategy": strategy}
                row.update({measurement: value for measurement in analysis_metrics()})
                rows.append(row)

        summary = summarize_strategies(rows)
        contrasts = paired_contrasts(
            rows, iterations=100, confidence=0.95, random_seed=42
        )

        self.assertEqual(summary["fixed_size"]["context_tokens"]["mean"], 3.0)
        fixed_context = next(
            row
            for row in contrasts
            if row["contrast"] == "fixed_size - hierarchical"
            and row["metric"] == "context_tokens"
        )
        self.assertEqual(fixed_context["mean_paired_difference"], 2.0)
        self.assertEqual(fixed_context["relative_mean_difference"], 2.0)


def sample_config() -> dict:
    return {
        "enrichment": {"context_budget_tokens": 3500},
        "rq3_cross_chunking": {
            "search_scope": "all_acts",
            "top_k": 5,
            "context_budget_tokens": 3500,
            "expected_query_count": 72,
            "new_conditions": [
                {
                    "id": "rq3_fixed_size",
                    "config_id": "fs_256_50",
                    "strategy": "fixed_size",
                },
                {
                    "id": "rq3_semantic",
                    "config_id": "sem_50_256",
                    "strategy": "semantic",
                },
            ],
            "hierarchical_reference": {
                "condition_id": "no_enrichment",
                "config_id": "hier_paragraph",
                "strategy": "hierarchical",
            },
        },
    }


def sample_context_records(rq3_config: dict) -> list[dict]:
    records = []
    for index in range(72):
        for condition in rq3_config["new_conditions"]:
            records.append(
                {
                    "query_id": f"q{index:03d}",
                    "condition_id": condition["id"],
                    "base_config_id": condition["config_id"],
                    "strategy": condition["strategy"],
                    "search_scope": "all_acts",
                    "base_top_k": 5,
                    "context_budget_tokens": 3500,
                    "candidate_chunk_count": 5,
                    "included_chunk_count": 5,
                    "excluded_chunk_count": 0,
                }
            )
    return records


def analysis_metrics() -> list[str]:
    return [
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


if __name__ == "__main__":
    unittest.main()
