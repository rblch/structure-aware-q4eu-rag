from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_import.id_normalization import (
    normalize_to_article_recital,
    normalize_to_paragraph,
    parse_reference_label,
)


class IdNormalizationTests(unittest.TestCase):
    def test_parse_article_reference(self) -> None:
        reference = parse_reference_label("G Art. 35.1")

        self.assertEqual(reference.document_code, "G")
        self.assertEqual(reference.unit_type, "Art")
        self.assertEqual(reference.number, "35.1")

    def test_article_recital_normalization_drops_article_suffix(self) -> None:
        self.assertEqual(normalize_to_article_recital("G Art. 35.3"), "G Art. 35")

    def test_article_recital_normalization_keeps_recital_number(self) -> None:
        self.assertEqual(normalize_to_article_recital("E Rec. 57"), "E Rec. 57")

    def test_paragraph_normalization_keeps_suffix(self) -> None:
        self.assertEqual(normalize_to_paragraph("B Article 8.3"), "B Art. 8.3")

    def test_lettered_article_is_supported(self) -> None:
        self.assertEqual(normalize_to_article_recital("W Art. 4a"), "W Art. 4a")

    def test_lettered_subparagraph_is_supported(self) -> None:
        self.assertEqual(normalize_to_paragraph("G Art. 5.1.b"), "G Art. 5.1.b")

    def test_roman_subparagraph_is_supported(self) -> None:
        self.assertEqual(normalize_to_article_recital("B Art. 2.2.ii"), "B Art. 2")


if __name__ == "__main__":
    unittest.main()
