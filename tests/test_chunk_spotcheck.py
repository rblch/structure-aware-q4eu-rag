from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.chunk_spotcheck import (
    build_edges_by_source,
    build_track,
    render_excerpt,
    select_anchor_chunk,
)


class ChunkSpotcheckTests(unittest.TestCase):
    def test_anchor_selection_prefers_midpoint_chunk_with_xrefs(self) -> None:
        chunks = [
            chunk("hier_paragraph_X_Art_1_1", 10, 20, ["X Art. 1.1"]),
            chunk("hier_paragraph_X_Art_5_1", 490, 520, ["X Art. 5.1"]),
            chunk("hier_paragraph_X_Art_9_1", 900, 930, ["X Art. 9.1"]),
        ]
        edges = build_edges_by_source(
            {
                "edges": [
                    {
                        "source_unit_id": "X Art. 5.1",
                        "target_unit_id_normalized": "X Art. 6",
                    },
                    {
                        "source_unit_id": "X Art. 9.1",
                        "target_unit_id_normalized": "X Art. 8",
                    },
                ]
            }
        )

        selected = select_anchor_chunk(
            document_code="X",
            canonical_text_length=1000,
            base_chunks=chunks,
            edges_by_source=edges,
        )

        self.assertEqual(selected["chunk_id"], "hier_paragraph_X_Art_5_1")

    def test_track_marks_smallest_chunk_containing_anchor(self) -> None:
        chunks = [
            chunk("wide", 0, 100, ["X Art. 1"]),
            chunk("narrow", 40, 60, ["X Art. 1.1"]),
        ]

        track = build_track(
            config_id="fs_test",
            label="Fixed test",
            chunks=chunks,
            document_code="X",
            window_start=0,
            window_end=100,
            anchor_center=50,
        )

        self.assertEqual(track["anchor_chunk_id"], "narrow")

    def test_excerpt_escapes_html_and_marks_anchor(self) -> None:
        rendered = render_excerpt(
            "before <tag> anchor & after",
            window_start=100,
            anchor_start=113,
            anchor_end=121,
        )

        self.assertIn("&lt;tag&gt;", rendered)
        self.assertIn("<mark>anchor &amp;</mark>", rendered)


def chunk(
    chunk_id: str,
    start: int,
    end: int,
    source_unit_ids: list[str],
) -> dict:
    return {
        "chunk_id": chunk_id,
        "config_id": "hier_paragraph",
        "strategy": "hierarchical",
        "text": chunk_id,
        "token_count": 5,
        "canonical_intervals": [[start, end]],
        "metadata": {
            "document_code": "X",
            "source_unit_ids": source_unit_ids,
            "answer_unit_ids": [source_unit_ids[0].rsplit(".", 1)[0]],
            "legal_order": start,
        },
    }


if __name__ == "__main__":
    unittest.main()
