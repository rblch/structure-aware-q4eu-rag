from pathlib import Path
import json
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_import.xref_eligibility import apply_structural_xref_eligibility
from parsing.xref_extractor import extract_xrefs


ROOT = Path(__file__).resolve().parents[1]


def unit(unit_id: str, document_code: str, unit_type: str, text: str) -> dict:
    return {
        "unit_id": unit_id,
        "document_code": document_code,
        "unit_type": unit_type,
        "text": text,
    }


class XrefExtractorTests(unittest.TestCase):
    """Integration tests over generated pipeline artifacts (parse stage)."""

    @classmethod
    def setUpClass(cls) -> None:
        legal_units_path = ROOT / "data/parsed/legal_units.json"
        if not legal_units_path.exists():
            raise unittest.SkipTest(
                "Pipeline artifacts missing; run `python3 -B src/pipeline.py parse` "
                f"first: {legal_units_path}"
            )
        cls.legal_units = json.loads(legal_units_path.read_text())

    def test_extracts_known_gdpr_reference(self) -> None:
        graph = extract_xrefs(self.legal_units)
        edge_pairs = {
            (edge["source_unit_id"], edge["target_unit_id_normalized"])
            for edge in graph["edges"]
        }

        self.assertIn(("G Art. 83.2.a", "G Art. 58"), edge_pairs)

    def test_extracts_known_warrant_reference(self) -> None:
        graph = extract_xrefs(self.legal_units)
        edge_pairs = {
            (edge["source_unit_id"], edge["target_unit_id_normalized"])
            for edge in graph["edges"]
        }

        self.assertIn(("W Art. 15.2", "W Art. 5"), edge_pairs)

    def test_extracts_article_conjunction_list(self) -> None:
        units = [
            unit("G Art. 13", "G", "article", "Information to be provided."),
            unit("G Art. 14", "G", "article", "Information where data not obtained."),
            unit("G Art. 16", "G", "article", "Right to rectification."),
            unit("G Art. 17", "G", "article", "Right to erasure."),
            unit("G Art. 18", "G", "article", "Right to restriction."),
            unit(
                "G Art. 99",
                "G",
                "article",
                "The controller shall provide the information referred to in "
                "Articles 13 and 14 and honour Articles 16, 17 or 18 of this "
                "Regulation.",
            ),
        ]
        graph = extract_xrefs(units)
        triples = {
            (
                edge["source_unit_id"],
                edge["target_unit_id_normalized"],
                edge["reference_kind"],
            )
            for edge in graph["edges"]
        }
        for target in ["G Art. 13", "G Art. 14", "G Art. 16", "G Art. 17", "G Art. 18"]:
            self.assertIn(("G Art. 99", target, "list_expanded"), triples)

    def test_extracts_recital_conjunction_list(self) -> None:
        units = [
            unit("G Rec. 71", "G", "recital", "Profiling recital."),
            unit("G Rec. 72", "G", "recital", "Profiling guidance recital."),
            unit(
                "G Art. 22",
                "G",
                "article",
                "Automated decision-making as noted in recitals 71 and 72.",
            ),
        ]
        graph = extract_xrefs(units)
        pairs = {
            (edge["source_unit_id"], edge["target_unit_id_normalized"])
            for edge in graph["edges"]
        }
        self.assertIn(("G Art. 22", "G Rec. 71"), pairs)
        self.assertIn(("G Art. 22", "G Rec. 72"), pairs)

    def test_article_list_with_external_marker_is_excluded(self) -> None:
        units = [
            unit("G Art. 15", "G", "article", "Right of access."),
            unit("G Art. 16", "G", "article", "Right to rectification."),
            unit(
                "G Art. 94",
                "G",
                "article",
                "References to Articles 15 and 16 of Directive 95/46/EC shall be "
                "construed as references to the corresponding provisions.",
            ),
        ]
        graph = extract_xrefs(units)
        list_edges = [
            edge for edge in graph["edges"] if edge["reference_kind"] == "list_expanded"
        ]
        self.assertEqual(list_edges, [])
        self.assertTrue(
            any(
                "Articles 15 and 16" in reference["raw_match"]
                for reference in graph["external_references_excluded"]
            )
        )

    def test_range_reference_is_not_double_counted_as_list(self) -> None:
        units = [
            unit("G Art. 12", "G", "article", "Transparency."),
            unit("G Art. 13", "G", "article", "Information."),
            unit("G Art. 14", "G", "article", "Information, indirect."),
            unit(
                "G Art. 5",
                "G",
                "article",
                "Compliance with Articles 12 to 14 of this Regulation.",
            ),
        ]
        graph = extract_xrefs(units)
        kinds = {
            (edge["target_unit_id_normalized"], edge["reference_kind"])
            for edge in graph["edges"]
            if edge["source_unit_id"] == "G Art. 5"
        }
        self.assertIn(("G Art. 12", "range_expanded"), kinds)
        self.assertNotIn(("G Art. 12", "list_expanded"), kinds)

    def test_structural_eligibility_report_has_pending_retrieval_gate(self) -> None:
        queries = json.loads((ROOT / "data/dataset/q4eu_queries.json").read_text())
        graph = extract_xrefs(self.legal_units)
        _, report = apply_structural_xref_eligibility(queries, graph)

        self.assertEqual(report["stage"], "1c-i")
        self.assertIsNone(report["retrieval_conditioned_count"])
        self.assertGreaterEqual(report["structural_upper_bound_count"], 1)


if __name__ == "__main__":
    unittest.main()
