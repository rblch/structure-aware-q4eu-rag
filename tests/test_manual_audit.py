from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.manual_audit import (  # noqa: E402
    MANUAL_COLUMNS,
    build_audit_row,
    manual_audit_settings_from_config,
    stratified_sample,
)


class ManualAuditTests(unittest.TestCase):
    def test_manual_audit_settings_read_fraction_and_seed(self) -> None:
        settings = manual_audit_settings_from_config(
            {
                "random_seed": 9,
                "analysis": {"manual_audit_fraction": 0.2},
            }
        )

        self.assertEqual(settings, {"fraction": 0.2, "random_seed": 9})

    def test_stratified_sample_keeps_rare_faithfulness_bands(self) -> None:
        records = []
        for condition_id in ("combined", "no_enrichment"):
            for index in range(10):
                records.append(
                    sample_record(condition_id, f"{condition_id}_h{index}", 1.0)
                )
            records.append(sample_record(condition_id, f"{condition_id}_m", 0.7))

        sampled, metadata = stratified_sample(
            records=records,
            sample_size=6,
            random_seed=42,
        )

        self.assertEqual(len(sampled), 6)
        self.assertEqual(metadata["stratum_sample_counts"]["combined|mid"], 1)
        self.assertEqual(metadata["stratum_sample_counts"]["no_enrichment|mid"], 1)

    def test_build_audit_row_leaves_manual_columns_blank(self) -> None:
        record = sample_record("combined", "q1", 1.0)
        record.update(
            {
                "faithfulness_band": "high",
                "correctness_score": 0.9,
                "answer_abstains": False,
                "faithfulness_abstention_justified": False,
                "correctness_answer_abstains": False,
                "correctness_abstention_justified": None,
                "either_judge_unjustified_abstention": False,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "citation_f1": 1.0,
                "cited_unit_ids": ["G Art. 1"],
                "extra_citation_unit_ids": [],
                "missing_gold_citation_unit_ids": [],
                "answer_text": "Answer.",
            }
        )

        row = build_audit_row(
            index=1,
            record=record,
            answer={
                "target_document_codes": ["G"],
                "context_text": "Context.",
            },
            faithfulness={"rationale": "Supported."},
            correctness={"rationale": "Correct."},
        )

        for column in MANUAL_COLUMNS:
            self.assertEqual(row[column], "")
        self.assertEqual(row["context_text"], "Context.")


def sample_record(condition_id: str, query_id: str, faithfulness_score: float) -> dict:
    return {
        "condition_id": condition_id,
        "query_id": query_id,
        "question": "Question?",
        "gold_unit_ids": ["G Art. 1"],
        "context_answer_unit_ids": ["G Art. 1"],
        "context_gold_recall": 1.0,
        "faithfulness_score": faithfulness_score,
    }


if __name__ == "__main__":
    unittest.main()
