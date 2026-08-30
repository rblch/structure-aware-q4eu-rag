import json
from pathlib import Path
import sys
import tempfile
import unittest

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from retrieval.index_corpus import (
    l2_normalize,
    load_or_embed_chunks,
    metadata_matches,
    write_faiss_index,
)


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls = 0
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts = texts
        return [[float(index + 1), 1.0, 0.5] for index, _ in enumerate(texts)]


class IndexingTests(unittest.TestCase):
    def test_l2_normalize_outputs_unit_vectors(self) -> None:
        vectors = np.asarray([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
        normalized = l2_normalize(vectors)

        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])

    def test_embedding_cache_reuses_matching_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_file = temp_path / "chunks.json"
            chunks = sample_chunks()
            chunk_file.write_text(json.dumps(chunks), encoding="utf-8")
            embedder = FakeEmbedder()

            first, first_metadata = load_or_embed_chunks(
                chunks=chunks,
                chunk_file=chunk_file,
                config=sample_config(),
                embedder=embedder,
                output_dir=temp_path / "embeddings",
            )
            second, second_metadata = load_or_embed_chunks(
                chunks=chunks,
                chunk_file=chunk_file,
                config=sample_config(),
                embedder=embedder,
                output_dir=temp_path / "embeddings",
            )

            self.assertEqual(embedder.calls, 1)
            np.testing.assert_allclose(first, second)
            self.assertEqual(first_metadata, second_metadata)

    def test_embedding_text_overrides_display_text_for_indexing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunk_file = temp_path / "chunks.json"
            chunks = sample_chunks()
            chunks[0]["embedding_text"] = "breadcrumb first chunk"
            chunk_file.write_text(json.dumps(chunks), encoding="utf-8")
            embedder = FakeEmbedder()

            _, metadata = load_or_embed_chunks(
                chunks=chunks,
                chunk_file=chunk_file,
                config=sample_config(),
                embedder=embedder,
                output_dir=temp_path / "embeddings",
            )

            self.assertEqual(embedder.texts[0], "breadcrumb first chunk")
            self.assertEqual(embedder.texts[1], "second chunk")
            self.assertEqual(metadata["embedding_text_field"], "embedding_text")

    def test_plain_text_embedding_metadata_remains_backward_compatible(self) -> None:
        self.assertTrue(
            metadata_matches(
                {"config_id": "fs_64_12"},
                {"config_id": "fs_64_12", "embedding_text_field": "text"},
            )
        )

    def test_write_faiss_index_uses_all_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = Path(temp_dir) / "index.faiss"
            embeddings = l2_normalize(
                np.asarray([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=np.float32)
            )

            write_faiss_index(embeddings, index_path, "IndexFlatIP")
            index = faiss.read_index(str(index_path))

            self.assertEqual(index.ntotal, 2)
            self.assertEqual(index.d, 3)


def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "test_1",
            "strategy": "fixed_size",
            "config_id": "test_config",
            "text": "first chunk",
        },
        {
            "chunk_id": "test_2",
            "strategy": "fixed_size",
            "config_id": "test_config",
            "text": "second chunk",
        },
    ]


def sample_config() -> dict:
    return {
        "models": {
            "embedding": {
                "provider": "openrouter",
                "model": "openai/text-embedding-3-small",
                "dimensions": 3,
            }
        }
    }


if __name__ == "__main__":
    unittest.main()
