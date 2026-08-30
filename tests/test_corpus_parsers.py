import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parsing.akn_parser import parse_akn_document
from parsing.corpus_parser import parse_corpus
from parsing.warrant_html_parser import collect_paragraphs, parse_warrant_document


ROOT = Path(__file__).resolve().parents[1]
RAW_DOCS = ROOT / "data/raw/discolqa/documents"


class CorpusParserTests(unittest.TestCase):
    def test_gdpr_parser_spot_checks(self) -> None:
        units, canonical = parse_akn_document("G", RAW_DOCS / "gdpr.akn")
        units_by_id = {unit.unit_id: unit for unit in units}

        self.assertIn("G Rec. 71", units_by_id)
        self.assertIn("G Art. 35", units_by_id)
        self.assertIn("G Art. 35.1", units_by_id)
        self.assertIn("Data protection impact assessment", units_by_id["G Art. 35"].text)
        self.assertGreater(canonical["char_length"], 100000)

    def test_warrant_html_collector_finds_article_titles(self) -> None:
        paragraphs = collect_paragraphs(RAW_DOCS / "warrant.html")
        titles = [p["text"] for p in paragraphs if "title-article-norm" in p["classes"]]

        self.assertIn("Article 4a", titles)
        self.assertIn("Article 30", titles)

    def test_warrant_parser_spot_checks(self) -> None:
        units, canonical = parse_warrant_document(RAW_DOCS / "warrant.html")
        units_by_id = {unit.unit_id: unit for unit in units}

        self.assertIn("W Rec. 12", units_by_id)
        self.assertIn("W Art. 4a", units_by_id)
        self.assertIn("W Art. 30", units_by_id)
        self.assertGreater(canonical["char_length"], 30000)

    def test_primary_gold_units_resolve(self) -> None:
        queries_path = ROOT / "data/dataset/q4eu_queries.json"
        if not queries_path.exists():
            raise unittest.SkipTest(
                "Pipeline artifacts missing; run `python3 -B src/pipeline.py parse` "
                f"first: {queries_path}"
            )
        legal_units, _, _ = parse_corpus(RAW_DOCS)
        parsed_ids = {unit["unit_id"] for unit in legal_units}
        queries = json.loads(queries_path.read_text())
        missing = sorted(
            unit_id
            for query in queries
            for unit_id in query["gold_unit_ids"]
            if unit_id not in parsed_ids
        )

        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
