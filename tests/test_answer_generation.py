import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generation.answer_generation import (
    PromptTemplate,
    assert_context_budget_checkpoint,
    assert_context_window,
    build_answer_record,
    fallback_completion_kwargs,
    load_prompt_template,
    render_messages,
    select_context_records,
)


class AnswerGenerationTests(unittest.TestCase):
    def test_generation_requires_matching_context_budget_checkpoint(self) -> None:
        config = {"enrichment": {"context_budget_tokens": 3500}}
        records = [sample_full_context_record("q1", "combined")]
        matching_metrics = {
            "context_budget_tokens": 3500,
            "volume_match": {"outside_tolerance_count": 0},
            "budget_derivation": {
                "configured_budget": 3500,
                "derived_budget": 3500,
                "matches_configured_budget": True,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "context_coverage_metrics.json"
            with self.assertRaisesRegex(RuntimeError, "requires the Stage 5"):
                assert_context_budget_checkpoint(
                    config=config,
                    context_records=records,
                    coverage_metrics_path=metrics_path,
                )

            metrics_path.write_text(
                json.dumps(matching_metrics), encoding="utf-8"
            )
            assert_context_budget_checkpoint(
                config=config,
                context_records=records,
                coverage_metrics_path=metrics_path,
            )

            matching_metrics["budget_derivation"]["derived_budget"] = 4000
            metrics_path.write_text(
                json.dumps(matching_metrics), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "Stage 6 blocked"):
                assert_context_budget_checkpoint(
                    config=config,
                    context_records=records,
                    coverage_metrics_path=metrics_path,
                )

    def test_prompt_template_parses_system_and_user_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            prompt_path = Path(temp_dir) / "prompt.txt"
            prompt_path.write_text(
                "System:\nFollow the rules.\n\nUser:\nContext: {context}\nQ: {question}",
                encoding="utf-8",
            )

            prompt = load_prompt_template(prompt_path)

            self.assertEqual(prompt.system, "Follow the rules.")
            self.assertEqual(prompt.user, "Context: {context}\nQ: {question}")

    def test_render_messages_does_not_include_condition_label(self) -> None:
        record = sample_context_record("q2", "combined")
        prompt = PromptTemplate(
            system="Use only context.",
            user="Context:\n{context}\n\nQuestion: {question}",
        )

        messages = render_messages(prompt, record)

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("A context passage.", messages[1]["content"])
        self.assertIn("What is tested?", messages[1]["content"])
        self.assertNotIn("combined", messages[1]["content"])

    def test_pilot_selection_keeps_all_conditions_for_first_queries(self) -> None:
        records = [
            sample_context_record("q2", "combined"),
            sample_context_record("q1", "combined"),
            sample_context_record("q2", "no_enrichment"),
            sample_context_record("q1", "no_enrichment"),
            sample_context_record("q3", "combined"),
        ]

        selected = select_context_records(
            records,
            condition_ids=None,
            query_ids=None,
            pilot_query_count=2,
        )

        self.assertEqual(
            [(record["query_id"], record["condition_id"]) for record in selected],
            [
                ("q1", "combined"),
                ("q1", "no_enrichment"),
                ("q2", "combined"),
                ("q2", "no_enrichment"),
            ],
        )

    def test_context_window_assertion_fails_when_budget_is_too_small(self) -> None:
        with self.assertRaises(ValueError):
            assert_context_window(
                [sample_context_record("q1", "combined")],
                prompt=PromptTemplate(system="Rules", user="{context}\n{question}"),
                generation_config={"context_window_tokens": 8, "max_tokens": 4},
                encoding_name="cl100k_base",
            )

    def test_fallback_completion_kwargs_removes_rejected_temperature(self) -> None:
        kwargs = {
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": "Question?"}],
            "max_tokens": 2048,
            "temperature": 0.0,
        }

        retry_kwargs = fallback_completion_kwargs(
            kwargs,
            "Unsupported parameter: temperature",
        )

        self.assertNotIn("temperature", retry_kwargs)
        self.assertEqual(retry_kwargs["max_tokens"], 2048)
        self.assertEqual(kwargs["temperature"], 0.0)

    def test_build_answer_record_logs_actual_completion_kwargs(self) -> None:
        result = {
            "answer_text": "The answer.",
            "finish_reason": "stop",
            "response_model": "openai/gpt-5-mini-2025-08-07",
            "completion_kwargs": {
                "model": "openai/gpt-5-mini",
                "max_tokens": 2048,
            },
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "elapsed_seconds": 1.25,
        }

        answer = build_answer_record(
            context_record=sample_full_context_record("q1", "combined"),
            result=result,
            prompt_path=Path("config/prompts/answer_generation.txt"),
            prompt_sha256="abc123",
            generation_config={
                "model": "openai/gpt-5-mini",
                "temperature": 0.0,
                "max_tokens": 2048,
            },
            messages=[{"role": "user", "content": "Context\nQuestion?"}],
            encoding_name="cl100k_base",
        )

        self.assertIsNone(answer["temperature"])
        self.assertEqual(answer["configured_temperature"], 0.0)
        self.assertEqual(answer["completion_kwargs"], result["completion_kwargs"])
        self.assertEqual(
            answer["configured_completion_kwargs"],
            {
                "model": "openai/gpt-5-mini",
                "max_tokens": 2048,
                "temperature": 0.0,
            },
        )
        self.assertTrue(answer["completion_kwargs_fallback_applied"])


def sample_context_record(query_id: str, condition_id: str) -> dict:
    return {
        "query_id": query_id,
        "condition_id": condition_id,
        "question": "What is tested?",
        "context_text": "A context passage.",
    }


def sample_full_context_record(query_id: str, condition_id: str) -> dict:
    record = sample_context_record(query_id, condition_id)
    record.update(
        {
            "specificity": "L",
            "target_document_codes": ["32016R0679"],
            "gold_unit_ids": ["32016R0679:art_1"],
            "base_config_id": "hier_paragraph",
            "search_scope": "target_documents",
            "base_top_k": 10,
            "context_budget_tokens": 3500,
            "context_token_count": 5,
            "included_chunk_count": 1,
            "excluded_chunk_count": 0,
            "context_gold_recall": 1.0,
            "context_gold_precision": 1.0,
            "context_gold_f1": 1.0,
            "context_answer_unit_ids": ["32016R0679:art_1"],
            "context_relevant_gold_unit_ids": ["32016R0679:art_1"],
            "context_missing_gold_unit_ids": [],
            "included_chunks": [],
        }
    )
    return record


if __name__ == "__main__":
    unittest.main()
