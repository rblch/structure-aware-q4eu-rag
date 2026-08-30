import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chunking.answer_units import build_unit_spans, chunk_to_answer_units
from chunking.chunk import chunk_id_for_unit
from chunking.hierarchical import build_hierarchical_chunks


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_ARTIFACTS = [
    ROOT / "data/parsed/legal_units.json",
    ROOT / "data/chunks/fixed_size/fs_256_50.json",
    ROOT / "data/chunks/hierarchical/hier_article.json",
    ROOT / "data/chunks/hierarchical/hier_paragraph.json",
    ROOT / "data/chunks/chunk_characteristics.json",
]


class ChunkingTests(unittest.TestCase):
    """Integration tests over generated pipeline artifacts (parse + chunk stages)."""

    @classmethod
    def setUpClass(cls) -> None:
        missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
        if missing:
            raise unittest.SkipTest(
                "Pipeline artifacts missing; run `python3 -B src/pipeline.py parse` "
                f"and `chunk` first: {missing}"
            )
        cls.legal_units = json.loads((ROOT / "data/parsed/legal_units.json").read_text())
        cls.fixed_chunks = json.loads(
            (ROOT / "data/chunks/fixed_size/fs_256_50.json").read_text()
        )
        cls.hier_article = json.loads(
            (ROOT / "data/chunks/hierarchical/hier_article.json").read_text()
        )
        cls.hier_paragraph = json.loads(
            (ROOT / "data/chunks/hierarchical/hier_paragraph.json").read_text()
        )

    def test_chunk_to_answer_units_maps_paragraph_to_article(self) -> None:
        units_by_id = {unit["unit_id"]: unit for unit in self.legal_units}
        spans = build_unit_spans(self.legal_units)
        article_ids = chunk_to_answer_units(
            document_code="G",
            canonical_intervals=units_by_id["G Art. 35.1"]["canonical_intervals"],
            spans_by_document=spans,
        )

        self.assertEqual(article_ids, ["G Art. 35"])

    def test_fixed_size_chunks_use_exact_token_stride(self) -> None:
        by_document: dict[str, list[dict]] = {}
        for chunk in self.fixed_chunks:
            by_document.setdefault(chunk["metadata"]["document_code"], []).append(chunk)

        for chunks in by_document.values():
            for previous, current in zip(chunks, chunks[1:]):
                stride = (
                    current["metadata"]["token_start"]
                    - previous["metadata"]["token_start"]
                )
                self.assertEqual(
                    stride,
                    206,
                )
                self.assertLessEqual(current["token_count"], 256)

    def test_chunks_do_not_cross_document_boundaries(self) -> None:
        for chunk in self.fixed_chunks + self.hier_paragraph:
            document_code = chunk["metadata"]["document_code"]
            for unit_id in chunk["metadata"]["answer_unit_ids"]:
                self.assertTrue(unit_id.startswith(f"{document_code} "))

    def test_hierarchical_parent_and_child_links_are_stable(self) -> None:
        paragraph_chunk = next(
            chunk
            for chunk in self.hier_paragraph
            if chunk["metadata"]["source_unit_ids"] == ["G Art. 35.1"]
        )
        article_chunk = next(
            chunk
            for chunk in self.hier_article
            if chunk["metadata"]["source_unit_ids"] == ["G Art. 35"]
        )

        self.assertEqual(paragraph_chunk["metadata"]["answer_unit_ids"], ["G Art. 35"])
        self.assertEqual(
            paragraph_chunk["metadata"]["parent_chunk_id"],
            chunk_id_for_unit("hier_article", "G Art. 35"),
        )
        self.assertIn(
            chunk_id_for_unit("hier_paragraph", "G Art. 35.1"),
            article_chunk["metadata"]["child_chunk_ids"],
        )

    def test_contextualized_hierarchical_chunks_add_embedding_text_only(self) -> None:
        contextualized = build_hierarchical_chunks(
            config_id="hier_paragraph_contextualized",
            leaf_level="paragraph_or_recital",
            legal_units=self.legal_units,
            embedding_context="structural_breadcrumb",
        )
        chunk = next(
            item
            for item in contextualized
            if item["metadata"]["source_unit_ids"] == ["G Art. 35.1"]
        )

        self.assertEqual(chunk["metadata"]["answer_unit_ids"], ["G Art. 35"])
        self.assertEqual(chunk["metadata"]["embedding_context"], "structural_breadcrumb")
        self.assertTrue(chunk["text"].startswith("1. Where a type of processing"))
        self.assertIn("G Chap. IV: Controller and processor", chunk["embedding_text"])
        self.assertIn(
            "G Art. 35: Data protection impact assessment",
            chunk["embedding_text"],
        )
        self.assertTrue(chunk["embedding_text"].endswith(chunk["text"]))

    def test_characteristics_report_has_no_unmapped_chunks(self) -> None:
        characteristics = json.loads(
            (ROOT / "data/chunks/chunk_characteristics.json").read_text()
        )
        for config_summary in characteristics["configs"].values():
            self.assertEqual(config_summary["chunks_without_answer_units"], [])


if __name__ == "__main__":
    unittest.main()
