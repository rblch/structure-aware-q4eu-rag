from pathlib import Path
import sys
import unittest
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

from chunking.semantic import build_semantic_chunks
from chunking.sentences import segment_document_sentences


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if any(term in text for term in ["Apple", "fruit", "Article 1"]):
                embeddings.append([1.0, 0.0])
            else:
                embeddings.append([0.0, 1.0])
        return embeddings


class SemanticChunkingTests(unittest.TestCase):
    def test_sentence_segments_keep_canonical_offsets(self) -> None:
        canonical_text, legal_units = sample_corpus()
        segments = segment_document_sentences(
            document_code="G",
            canonical_text=canonical_text,
            legal_units=legal_units,
        )

        self.assertGreaterEqual(len(segments), 4)
        for segment in segments:
            start, end = segment.canonical_intervals[0]
            self.assertEqual(segment.text, canonical_text[start:end])
            self.assertTrue(segment.answer_unit_ids)

    def test_semantic_chunks_have_strategy_and_answer_units(self) -> None:
        canonical_text, legal_units = sample_corpus()
        chunks = build_semantic_chunks(
            config_id="sem_test",
            breakpoint_percentile=60,
            max_chunk_size=512,
            min_chunk_size=1,
            window_size=1,
            canonical_texts={
                "G": {
                    "document_code": "G",
                    "text": canonical_text,
                    "source_text_sha256": "sha",
                }
            },
            legal_units=legal_units,
            embedder=FakeEmbedder(),
        )

        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(chunk["strategy"], "semantic")
            self.assertTrue(chunk["metadata"]["answer_unit_ids"])
            self.assertTrue(
                all(
                    unit_id.startswith("G ")
                    for unit_id in chunk["metadata"]["answer_unit_ids"]
                )
            )

    def test_legal_numbering_fragments_merge_with_following_segment(self) -> None:
        canonical_text = "Article 1. Scope\n\n1. This Regulation applies. 2. It binds courts."
        legal_units = [
            legal_unit("G Art. 1", "1", 0, len(canonical_text), 1, canonical_text)
        ]

        segments = segment_document_sentences(
            document_code="G",
            canonical_text=canonical_text,
            legal_units=legal_units,
        )

        self.assertNotIn("1.", [segment.text for segment in segments])
        self.assertNotIn("2.", [segment.text for segment in segments])
        self.assertFalse(
            any(segment.text.strip().endswith("\n\n1.") for segment in segments)
        )
        self.assertTrue(
            any(
                segment.text.startswith("Article 1. Scope\n\n1. This Regulation applies.")
                for segment in segments
            )
        )

    def test_semantic_chunks_enforce_hard_max_for_long_single_sentence(self) -> None:
        long_sentence = "Article 1. " + " ".join(f"term{i}" for i in range(90)) + "."
        legal_units = [
            legal_unit("G Art. 1", "1", 0, len(long_sentence), 1, long_sentence)
        ]
        chunks = build_semantic_chunks(
            config_id="sem_test",
            breakpoint_percentile=50,
            max_chunk_size=32,
            min_chunk_size=16,
            window_size=1,
            canonical_texts={
                "G": {
                    "document_code": "G",
                    "text": long_sentence,
                    "source_text_sha256": "sha",
                }
            },
            legal_units=legal_units,
            embedder=FakeEmbedder(),
        )

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(chunk["token_count"] for chunk in chunks), 32)
        self.assertTrue(all(chunk["text"].strip() for chunk in chunks))


def sample_corpus() -> tuple[str, list[dict]]:
    article_1 = "Article 1. Apples are red. Apples are fruit."
    article_2 = "Article 2. Courts decide cases. Judges write orders."
    canonical_text = f"{article_1}\n\n{article_2}"
    article_2_start = len(article_1) + 2
    legal_units = [
        legal_unit("G Art. 1", "1", 0, len(article_1), 1, article_1),
        legal_unit(
            "G Art. 2",
            "2",
            article_2_start,
            len(canonical_text),
            2,
            article_2,
        ),
    ]
    return canonical_text, legal_units


def legal_unit(
    unit_id: str,
    number: str,
    start: int,
    end: int,
    legal_order: int,
    text: str,
) -> dict:
    return {
        "unit_id": unit_id,
        "document_code": "G",
        "unit_type": "article",
        "number": number,
        "title": "",
        "text": text,
        "parent_unit_id": "G Root",
        "child_unit_ids": [],
        "canonical_char_start": start,
        "canonical_char_end": end,
        "canonical_intervals": [[start, end]],
        "source_text_sha256": "sha",
        "metadata": {"legal_order": legal_order, "source_path": "sample"},
    }


if __name__ == "__main__":
    unittest.main()
