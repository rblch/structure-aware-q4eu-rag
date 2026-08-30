from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_import.q4eu_parser import build_import_report, build_queries, parse_question_dict


ROOT = Path(__file__).resolve().parents[1]


class Q4EuParserTests(unittest.TestCase):
    def test_parse_real_question_dict_counts(self) -> None:
        question_dict = parse_question_dict(ROOT / "data/raw/discolqa/evaluate.py")

        self.assertEqual(len(question_dict), 72)
        self.assertEqual(sum(len(info["expected_answers"]) for info in question_dict.values()), 246)

    def test_build_queries_normalizes_gold_units(self) -> None:
        question_dict = parse_question_dict(ROOT / "data/raw/discolqa/evaluate.py")
        queries = build_queries(question_dict, "Francesco-Sovrano/DiscoLQA", "f" * 40)
        all_questions = {query["question"]: query for query in queries}
        query = all_questions["Does the GDPR provide a right to explanation?"]

        self.assertIn("G Art. 12", query["gold_unit_ids"])
        self.assertIn("G Art. 12.3", query["gold_unit_ids_paragraph_level"])

    def test_import_report_matches_verified_counts(self) -> None:
        question_dict = parse_question_dict(ROOT / "data/raw/discolqa/evaluate.py")
        queries = build_queries(question_dict, "Francesco-Sovrano/DiscoLQA", "f" * 40)
        report = build_import_report(
            queries,
            {
                "source_repo": "Francesco-Sovrano/DiscoLQA",
                "source_branch": "main",
                "source_commit": "f" * 40,
            },
        )

        self.assertEqual(report["specificity_counts"], {"H": 22, "L": 22, "N": 28})
        self.assertEqual(report["target_document_tag_counts"]["W"], 21)
        self.assertEqual(report["multi_act_question_count"], 2)
        self.assertEqual(report["stage_1b_sanity_check"]["status"], "passed")
        self.assertEqual(
            report["stage_1b_sanity_check"]["checked_examples"]["G Art. 35.3"],
            "G Art. 35",
        )


if __name__ == "__main__":
    unittest.main()
