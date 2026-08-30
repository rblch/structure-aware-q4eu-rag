from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.generated_answer_evaluation import (
    CORRECTNESS_OUTPUT_SCHEMA_PATH,
    FAITHFULNESS_OUTPUT_SCHEMA_PATH,
    OpenRouterJudge,
    build_answer_evaluation_records,
    citation_coverage,
    evaluation_config_for,
    load_output_schema,
    parse_json_object,
    render_correctness_messages,
    render_faithfulness_messages,
    schema_violations,
    summarize_generation_metrics,
    sum_usage_records,
)
from generation.answer_generation import PromptTemplate


class GeneratedAnswerEvaluationTests(unittest.TestCase):
    def test_evaluation_roles_use_independent_model_configs(self) -> None:
        config = {
            "models": {
                "evaluation": {
                    "faithfulness": {"model": "anthropic/claude-haiku-4.5"},
                    "correctness": {"model": "google/gemini-3.5-flash"},
                }
            }
        }

        faithfulness = evaluation_config_for(config, "faithfulness")
        correctness = evaluation_config_for(config, "correctness")

        self.assertEqual(faithfulness["model"], "anthropic/claude-haiku-4.5")
        self.assertEqual(correctness["model"], "google/gemini-3.5-flash")

    def test_evaluation_role_must_be_configured(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "models.evaluation.correctness",
        ):
            evaluation_config_for(
                {"models": {"evaluation": {"faithfulness": {}}}},
                "correctness",
            )

    def test_parse_json_object_accepts_fenced_json(self) -> None:
        parsed = parse_json_object(
            '```json\n{"faithfulness_score": 0.75, "answer_abstains": false}\n```'
        )

        self.assertEqual(parsed["faithfulness_score"], 0.75)
        self.assertFalse(parsed["answer_abstains"])

    def test_judge_retries_score_outside_registered_increment(self) -> None:
        responses = [
            judge_response(faithfulness_output(0.75)),
            judge_response(faithfulness_output(0.8)),
        ]
        requested_messages = []
        judge = OpenRouterJudge.__new__(OpenRouterJudge)

        def create_completion(messages):
            requested_messages.append(messages)
            return responses.pop(0)

        judge._create_completion = create_completion

        result = judge.judge(
            [{"role": "user", "content": "Judge this answer."}],
            load_output_schema(FAITHFULNESS_OUTPUT_SCHEMA_PATH),
        )

        self.assertEqual(result["parsed_judge_output"]["faithfulness_score"], 0.8)
        self.assertEqual(result["judge_attempt_count"], 2)
        self.assertIn("multiple of 0.1", result["judge_retry_errors"][0])
        self.assertEqual(result["judge_retry_outputs"], [faithfulness_output(0.75)])
        self.assertIn("permitted increments", requested_messages[1][-1]["content"])

    def test_both_judge_schemas_require_tenth_point_scores(self) -> None:
        for schema_path, score_field in (
            (FAITHFULNESS_OUTPUT_SCHEMA_PATH, "faithfulness_score"),
            (CORRECTNESS_OUTPUT_SCHEMA_PATH, "correctness_score"),
        ):
            schema = load_output_schema(schema_path)
            with self.subTest(score_field=score_field):
                invalid = schema_violations({score_field: 0.75}, schema)
                valid = schema_violations({score_field: 0.7}, schema)
                self.assertTrue(any("multiple of 0.1" in item for item in invalid))
                self.assertFalse(any("multiple of 0.1" in item for item in valid))

    def test_sum_usage_records_adds_numeric_fields_and_keeps_metadata(self) -> None:
        usage = sum_usage_records(
            [
                {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                {
                    "prompt_tokens": 7,
                    "completion_tokens": 3,
                    "total_tokens": 10,
                    "cost": 0.01,
                    "is_byok": False,
                },
            ]
        )

        self.assertEqual(usage["prompt_tokens"], 17)
        self.assertEqual(usage["completion_tokens"], 5)
        self.assertEqual(usage["total_tokens"], 22)
        self.assertEqual(usage["cost"], 0.01)
        self.assertFalse(usage["is_byok"])

    def test_faithfulness_prompt_hides_condition_label(self) -> None:
        answer = sample_answer("combined")
        prompt = PromptTemplate(
            system="Judge faithfulness.",
            user="Q: {question}\nC: {context}\nA: {answer}",
        )

        messages = render_faithfulness_messages(prompt, answer)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Context text", messages[1]["content"])
        self.assertIn("Answer text", messages[1]["content"])
        self.assertNotIn("combined", messages[1]["content"])

    def test_correctness_prompt_uses_gold_texts_and_hides_condition_label(self) -> None:
        answer = sample_answer("combined")
        prompt = PromptTemplate(
            system="Judge correctness.",
            user="Q: {question}\nG: {gold_texts}\nA: {answer}",
        )
        legal_units = {"E Art. 32": {"unit_id": "E Art. 32", "text": "Gold text"}}

        messages = render_correctness_messages(prompt, answer, legal_units)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("[G1] E Art. 32", messages[1]["content"])
        self.assertIn("Gold text", messages[1]["content"])
        self.assertIn("Answer text", messages[1]["content"])
        self.assertNotIn("combined", messages[1]["content"])

    def test_citation_coverage_normalizes_article_references(self) -> None:
        answer = {
            **sample_answer("xref_only"),
            "answer_text": "Use Art. 32(1), E Art. 33.1, and E Art. 26(a).",
        }

        coverage = citation_coverage(answer)

        self.assertEqual(
            coverage["cited_unit_ids"],
            ["E Art. 32", "E Art. 33", "E Art. 26"],
        )
        self.assertEqual(coverage["cited_gold_unit_ids"], ["E Art. 32"])
        self.assertEqual(coverage["missing_gold_citation_unit_ids"], [])
        self.assertEqual(coverage["extra_citation_unit_ids"], ["E Art. 26", "E Art. 33"])
        self.assertAlmostEqual(coverage["citation_recall"], 1.0)
        self.assertAlmostEqual(coverage["citation_precision"], 1 / 3)

    def test_citation_coverage_expands_plural_conjunction_lists(self) -> None:
        answer = {
            **sample_answer("parent_only"),
            "gold_unit_ids": ["E Art. 13", "E Art. 14", "E Rec. 71"],
            "answer_text": (
                "Articles 13 and 14 apply, as noted in Recitals 71, 72 and 73. "
                "Arts. 15, 16 or 18 do not. See also Art. 32(1)."
            ),
        }

        coverage = citation_coverage(answer)

        self.assertEqual(
            coverage["cited_unit_ids"],
            [
                "E Art. 13",
                "E Art. 14",
                "E Rec. 71",
                "E Rec. 72",
                "E Rec. 73",
                "E Art. 15",
                "E Art. 16",
                "E Art. 18",
                "E Art. 32",
            ],
        )
        self.assertEqual(coverage["citation_count"], 9)
        self.assertEqual(
            coverage["cited_gold_unit_ids"], ["E Art. 13", "E Art. 14", "E Rec. 71"]
        )
        self.assertAlmostEqual(coverage["citation_recall"], 1.0)

    def test_generation_metrics_include_efficiency_fields(self) -> None:
        answer = sample_answer("no_enrichment")
        faithfulness_score = {
            "query_id": answer["query_id"],
            "condition_id": answer["condition_id"],
            "faithfulness_score": 0.9,
            "answer_abstains": False,
            "abstention_justified": False,
            "unjustified_abstention": False,
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
            "estimated_cost_usd": 0.03,
            "elapsed_seconds": 2.0,
        }
        correctness_score = {
            "query_id": answer["query_id"],
            "condition_id": answer["condition_id"],
            "correctness_score": 0.7,
            "answer_abstains": False,
            "abstention_justified": False,
            "unjustified_abstention": False,
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            "estimated_cost_usd": 0.04,
            "elapsed_seconds": 3.0,
        }
        evaluation_records = build_answer_evaluation_records(
            answers=[answer],
            faithfulness_scores=[faithfulness_score],
            correctness_scores=[correctness_score],
        )

        metrics = summarize_generation_metrics(
            answers=[answer],
            faithfulness_scores=[faithfulness_score],
            correctness_scores=[correctness_score],
            answer_evaluation_records=evaluation_records,
            unfaithful_threshold=0.5,
        )

        condition = metrics["conditions"]["no_enrichment"]
        self.assertEqual(condition["faithfulness"]["mean"], 0.9)
        self.assertEqual(condition["correctness"]["mean"], 0.7)
        self.assertEqual(
            condition["generation_efficiency"]["total_cost_usd"],
            0.02,
        )
        self.assertEqual(condition["faithfulness_judge_efficiency"]["total_tokens"], 14)
        self.assertEqual(condition["correctness_judge_efficiency"]["total_tokens"], 16)
        self.assertEqual(condition["judge_efficiency"]["total_tokens"], 30)


def sample_answer(condition_id: str) -> dict:
    return {
        "query_id": "q1",
        "condition_id": condition_id,
        "question": "Question text",
        "target_document_codes": ["E"],
        "gold_unit_ids": ["E Art. 32"],
        "context_answer_unit_ids": ["E Art. 32"],
        "context_gold_recall": 1.0,
        "context_text": "Context text",
        "answer_text": "Answer text",
        "usage": {"prompt_tokens": 8, "completion_tokens": 5, "total_tokens": 13},
        "estimated_cost_usd": 0.02,
        "elapsed_seconds": 1.0,
    }


def faithfulness_output(score: float) -> str:
    return (
        '{"faithfulness_score": '
        f"{score}, "
        '"answer_abstains": false, '
        '"abstention_justified": false, '
        '"abstention_reason": "", '
        '"unfaithful_claims": [], '
        '"supporting_evidence": [], '
        '"rationale": "Test."}'
    )


def judge_response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="judge-test",
        model="test-judge",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(content=content),
            )
        ],
        usage=None,
    )


if __name__ == "__main__":
    unittest.main()
