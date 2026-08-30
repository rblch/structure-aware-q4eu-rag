from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chunking.characteristics import build_chunk_characteristics
from chunking.fixed_size import build_fixed_size_chunks
from chunking.hierarchical import build_hierarchical_chunks


def write_chunk_outputs(
    *,
    config_path: Path,
    legal_units_path: Path,
    canonical_texts_path: Path,
    output_dir: Path,
    characteristics_path: Path,
    include_semantic: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    legal_units = json.loads(legal_units_path.read_text(encoding="utf-8"))
    canonical_texts = json.loads(canonical_texts_path.read_text(encoding="utf-8"))

    chunks_by_config: dict[str, list[dict[str, Any]]] = {}
    write_fixed_size_chunks(
        config,
        canonical_texts,
        legal_units,
        output_dir,
        chunks_by_config,
    )
    write_hierarchical_chunks(config, legal_units, output_dir, chunks_by_config)
    if include_semantic:
        write_semantic_chunks(
            config,
            canonical_texts,
            legal_units,
            output_dir,
            chunks_by_config,
        )

    characteristics = build_chunk_characteristics(chunks_by_config)
    characteristics_path.parent.mkdir(parents=True, exist_ok=True)
    characteristics_path.write_text(
        json.dumps(characteristics, indent=2) + "\n",
        encoding="utf-8",
    )
    return characteristics


def write_fixed_size_chunks(
    config: dict[str, Any],
    canonical_texts: dict[str, dict[str, Any]],
    legal_units: list[dict[str, Any]],
    output_dir: Path,
    chunks_by_config: dict[str, list[dict[str, Any]]],
) -> None:
    fixed_config = config["chunking"]["fixed_size"]
    for chunk_config in fixed_config["configs"]:
        config_id = chunk_config["id"]
        chunks = build_fixed_size_chunks(
            config_id=config_id,
            chunk_size=chunk_config["chunk_size"],
            chunk_overlap=chunk_config["chunk_overlap"],
            canonical_texts=canonical_texts,
            legal_units=legal_units,
        )
        write_chunks(output_dir / "fixed_size" / f"{config_id}.json", chunks)
        chunks_by_config[config_id] = chunks


def write_hierarchical_chunks(
    config: dict[str, Any],
    legal_units: list[dict[str, Any]],
    output_dir: Path,
    chunks_by_config: dict[str, list[dict[str, Any]]],
) -> None:
    for chunk_config in config["chunking"]["hierarchical"]["configs"]:
        config_id = chunk_config["id"]
        chunks = build_hierarchical_chunks(
            config_id=config_id,
            leaf_level=chunk_config["leaf_level"],
            legal_units=legal_units,
            embedding_context=chunk_config.get("embedding_context"),
        )
        write_chunks(output_dir / "hierarchical" / f"{config_id}.json", chunks)
        chunks_by_config[config_id] = chunks


def write_semantic_chunks(
    config: dict[str, Any],
    canonical_texts: dict[str, dict[str, Any]],
    legal_units: list[dict[str, Any]],
    output_dir: Path,
    chunks_by_config: dict[str, list[dict[str, Any]]],
) -> None:
    from chunking.semantic import build_semantic_chunks
    from retrieval.openrouter_embeddings import OpenRouterEmbedder

    semantic_config = config["chunking"]["semantic"]
    embedder = OpenRouterEmbedder(
        config,
        cache_dir=Path("data/embeddings/text_cache"),
    )
    for chunk_config in semantic_config["configs"]:
        config_id = chunk_config["id"]
        chunks = build_semantic_chunks(
            config_id=config_id,
            breakpoint_percentile=chunk_config["breakpoint_percentile"],
            max_chunk_size=chunk_config["max_chunk_size"],
            min_chunk_size=semantic_config["min_chunk_size"],
            window_size=semantic_config["window_size"],
            canonical_texts=canonical_texts,
            legal_units=legal_units,
            embedder=embedder,
        )
        write_chunks(output_dir / "semantic" / f"{config_id}.json", chunks)
        chunks_by_config[config_id] = chunks


def write_chunks(path: Path, chunks: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chunks, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage 2 chunk files.")
    parser.add_argument("--config-path", type=Path, default=Path("config/config.yaml"))
    parser.add_argument(
        "--legal-units-path",
        type=Path,
        default=Path("data/parsed/legal_units.json"),
    )
    parser.add_argument(
        "--canonical-texts-path",
        type=Path,
        default=Path("data/parsed/canonical_texts.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument(
        "--include-semantic",
        action="store_true",
        help="Generate semantic chunks using OpenRouter embeddings.",
    )
    parser.add_argument(
        "--characteristics-path",
        type=Path,
        default=Path("data/chunks/chunk_characteristics.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    characteristics = write_chunk_outputs(
        config_path=args.config_path,
        legal_units_path=args.legal_units_path,
        canonical_texts_path=args.canonical_texts_path,
        output_dir=args.output_dir,
        characteristics_path=args.characteristics_path,
        include_semantic=args.include_semantic,
    )
    print(
        "Generated chunks for "
        f"{len(characteristics['configs'])} configurations."
    )


if __name__ == "__main__":
    main()
