import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluation.manual_audit_review import (  # noqa: E402
    gold_reference_text,
    load_reviews,
    merge_reviews,
    phase_one_complete,
    review_complete,
    review_order,
    reviewer_output_path,
    save_review,
)


class ManualAuditReviewTests(unittest.TestCase):
    def test_reviewer_output_path_rejects_unsafe_identifier(self) -> None:
        with self.assertRaises(ValueError):
            reviewer_output_path(Path("reviews"), "../reviewer")

    def test_gold_reference_text_resolves_all_gold_units(self) -> None:
        text = gold_reference_text(
            {"gold_unit_ids": '["A Art. 1", "A Art. 2"]'},
            {"A Art. 1": "First.", "A Art. 2": "Second."},
        )

        self.assertEqual(text, "[A Art. 1]\nFirst.\n\n[A Art. 2]\nSecond.")

    def test_save_review_supports_partial_then_completed_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            partial = sample_review()
            partial.update(
                {
                    "manual_faithfulness_score": "0.8",
                    "manual_correctness_score": "0.7",
                    "manual_abstention_handled_correctly": "Not applicable",
                }
            )
            save_review(path, partial)

            saved = load_reviews(path)["audit_001"]
            self.assertTrue(phase_one_complete(saved))
            self.assertFalse(review_complete(saved))

            saved.update(
                {
                    "manual_judge_score_reasonable": "Yes",
                    "manual_unsupported_claims_missed": "No",
                    "reviewer_id": "reviewer-1",
                    "reviewed_at_utc": "2026-01-01T00:00:00+00:00",
                }
            )
            save_review(path, saved)

            self.assertTrue(review_complete(load_reviews(path)["audit_001"]))

    def test_review_order_is_deterministic_and_merge_preserves_audit_order(
        self,
    ) -> None:
        audit_rows = [
            {"audit_id": "audit_001", "manual_notes": ""},
            {"audit_id": "audit_002", "manual_notes": ""},
            {"audit_id": "audit_003", "manual_notes": ""},
        ]
        self.assertEqual(review_order(audit_rows), review_order(audit_rows))
        reviews = {"audit_002": {"manual_notes": "Checked."}}

        merged = merge_reviews(audit_rows, reviews)

        self.assertEqual(
            [row["audit_id"] for row in merged], ["audit_001", "audit_002", "audit_003"]
        )
        self.assertEqual(merged[1]["manual_notes"], "Checked.")

    def test_saved_csv_has_one_row_per_audit_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            review = sample_review()
            save_review(path, review)
            review["manual_notes"] = "Revised."
            save_review(path, review)

            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["manual_notes"], "Revised.")


def sample_review() -> dict[str, str]:
    return {"audit_id": "audit_001"}


if __name__ == "__main__":
    unittest.main()
