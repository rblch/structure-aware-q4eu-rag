from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.openrouter_embeddings import OpenRouterEmbedder


def write_index_outputs(
    *,
    config_path: Path,
    chunks_dir: Path,
    embeddings_dir: Path,
    indices_dir: Path,
    characteristics_path: Path,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    embedder = OpenRouterEmbedder(config)
    chunk_files = config_chunk_files(config, chunks_dir)
    config_summaries: dict[str, Any] = {}

    for chunk_file in chunk_files:
        chunks = json.loads(chunk_file.read_text(encoding="utf-8"))
        if not chunks:
            continue
        config_id = chunks[0]["config_id"]
        embeddings, metadata = load_or_embed_chunks(
            chunks=chunks,
            chunk_file=chunk_file,
            config=config,
            embedder=embedder,
            output_dir=embeddings_dir / config_id,
        )
        normalized = l2_normalize(embeddings)
        index_path = indices_dir / config_id / "index.faiss"
        write_faiss_index(normalized, index_path, config["retrieval"]["index_type"])

        config_summaries[config_id] = build_index_summary(
            config_id=config_id,
            chunk_file=chunk_file,
            metadata=metadata,
            normalized_embeddings=normalized,
            index_path=index_path,
        )

    characteristics = {
        "index_type": config["retrieval"]["index_type"],
        "configs": dict(sorted(config_summaries.items())),
    }
    characteristics_path.parent.mkdir(parents=True, exist_ok=True)
    characteristics_path.write_text(
        json.dumps(characteristics, indent=2) + "\n",
        encoding="utf-8",
    )
    return characteristics


def config_chunk_files(config: dict[str, Any], chunks_dir: Path) -> list[Path]:
    # Ignore stale chunk files not declared in the config.
    chunk_files = [
        chunks_dir / family / f"{chunk_config['id']}.json"
        for family in ("fixed_size", "semantic", "hierarchical")
        for chunk_config in config["chunking"][family]["configs"]
    ]
    missing = [str(path) for path in chunk_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Config-declared chunk files are missing; run the chunk stage "
            f"(semantic configs require --include-semantic): {missing}"
        )
    return sorted(chunk_files)


def load_or_embed_chunks(
    *,
    chunks: list[dict[str, Any]],
    chunk_file: Path,
    config: dict[str, Any],
    embedder: OpenRouterEmbedder,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = output_dir / "embeddings.npy"
    chunk_id_order_path = output_dir / "chunk_id_order.json"
    metadata_path = output_dir / "embedding_metadata.json"
    expected_metadata = build_embedding_metadata(chunks, chunk_file, config)

    if (
        embeddings_path.exists()
        and chunk_id_order_path.exists()
        and metadata_path.exists()
    ):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_matches(metadata, expected_metadata):
            return np.load(embeddings_path), metadata

    texts = [embedding_text_for_chunk(chunk) for chunk in chunks]
    embeddings = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Embedding matrix for {chunk_file} is not two-dimensional")
    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            f"Embedding row count mismatch for {chunk_file}: "
            f"{embeddings.shape[0]} rows for {len(chunks)} chunks"
        )

    metadata = dict(expected_metadata)
    metadata["actual_dimensions"] = int(embeddings.shape[1])
    np.save(embeddings_path, embeddings)
    chunk_id_order_path.write_text(
        json.dumps([chunk["chunk_id"] for chunk in chunks], indent=2) + "\n",
        encoding="utf-8",
    )
    # Bind embeddings and chunk order to their metadata.
    metadata["embeddings_sha256"] = file_sha256(embeddings_path)
    metadata["chunk_id_order_sha256"] = file_sha256(chunk_id_order_path)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return embeddings, metadata


def build_embedding_metadata(
    chunks: list[dict[str, Any]],
    chunk_file: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    embedding_config = config["models"]["embedding"]
    return {
        "config_id": chunks[0]["config_id"],
        "strategy": chunks[0]["strategy"],
        "chunk_count": len(chunks),
        "chunk_file": str(chunk_file),
        "chunk_file_sha256": file_sha256(chunk_file),
        "embedding_text_field": embedding_text_field(chunks),
        "embedding_provider": embedding_config["provider"],
        "embedding_model": embedding_config["model"],
        "configured_dimensions": embedding_config.get("dimensions"),
    }


def metadata_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if (
            key == "embedding_text_field"
            and value == "text"
            and key not in actual
        ):
            continue
        if actual.get(key) != value:
            return False
    return True


def embedding_text_for_chunk(chunk: dict[str, Any]) -> str:
    return chunk.get("embedding_text") or chunk["text"]


def embedding_text_field(chunks: list[dict[str, Any]]) -> str:
    return "embedding_text" if any("embedding_text" in chunk for chunk in chunks) else "text"


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    vectors = embeddings.astype(np.float32, copy=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot index zero-norm embeddings")
    vectors /= norms
    return vectors


def write_faiss_index(
    normalized_embeddings: np.ndarray,
    index_path: Path,
    index_type: str,
) -> None:
    if index_type != "IndexFlatIP":
        raise ValueError(f"Unsupported FAISS index type: {index_type}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    index.add(normalized_embeddings)
    faiss.write_index(index, str(index_path))


def build_index_summary(
    *,
    config_id: str,
    chunk_file: Path,
    metadata: dict[str, Any],
    normalized_embeddings: np.ndarray,
    index_path: Path,
) -> dict[str, Any]:
    norms = np.linalg.norm(normalized_embeddings, axis=1)
    return {
        "config_id": config_id,
        "chunk_file": str(chunk_file),
        "chunk_count": int(normalized_embeddings.shape[0]),
        "dimensions": int(normalized_embeddings.shape[1]),
        "embedding_model": metadata["embedding_model"],
        "chunk_file_sha256": metadata["chunk_file_sha256"],
        "index_path": str(index_path),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed chunks and build FAISS indices."
    )
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument("--chunks-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--embeddings-dir", type=Path, default=Path("data/embeddings"))
    parser.add_argument("--indices-dir", type=Path, default=Path("data/indices"))
    parser.add_argument(
        "--characteristics-path",
        type=Path,
        default=Path("data/indices/index_characteristics.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    characteristics = write_index_outputs(
        config_path=args.config_path,
        chunks_dir=args.chunks_dir,
        embeddings_dir=args.embeddings_dir,
        indices_dir=args.indices_dir,
        characteristics_path=args.characteristics_path,
    )
    print(
        "Stage 3 indexing complete: "
        f"{len(characteristics['configs'])} configurations."
    )


if __name__ == "__main__":
    main()
