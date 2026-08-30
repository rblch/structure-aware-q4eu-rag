from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_import.discolqa_fetch import raw_url, validate_commit


class DiscolqaFetchTests(unittest.TestCase):
    def test_raw_url_escapes_bracketed_path(self) -> None:
        url = raw_url(
            "Francesco-Sovrano/DiscoLQA",
            "f8614b7d75d4054fb6b02119cde92492172fd6b6",
            "[Law_EU]qa_overview/oke/evaluate.py",
        )

        self.assertIn("%5BLaw_EU%5Dqa_overview/oke/evaluate.py", url)

    def test_validate_commit_rejects_placeholder(self) -> None:
        with self.assertRaises(SystemExit):
            validate_commit("REQUIRED")

    def test_validate_commit_accepts_pinned_sha(self) -> None:
        validate_commit("f8614b7d75d4054fb6b02119cde92492172fd6b6")

    def test_fetcher_output_path_keeps_document_name(self) -> None:
        self.assertEqual(
            Path("[Law_EU]qa_overview/oke/documents/gdpr.akn").name,
            "gdpr.akn",
        )


if __name__ == "__main__":
    unittest.main()
