from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from chunking.semantic import compute_boundary_scores, split_oversize_segments
from chunking.sentences import segment_document_sentences
from chunking.tokenization import DEFAULT_ENCODING
from figures.style import AQUA, AXIS, BLUE, INK, ORANGE, apply_style, save

CANONICAL_TEXTS_PATH = Path("data/parsed/canonical_texts.json")
LEGAL_UNITS_PATH = Path("data/parsed/legal_units.json")
EMBEDDING_CACHE_PATH = Path(
    "data/embeddings/text_cache/openai_text-embedding-3-small_1536.jsonl"
)
DOCUMENT_CODE = "G"
MAX_CHUNK_SIZE = 256
WINDOW_SIZE = 3
PERCENTILES = (50, 70)
UNIT_TYPES = {"article", "recital"}


def load_legal_units(legal_units_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(legal_units_path.read_text(encoding="utf-8"))
    return payload["legal_units"] if isinstance(payload, dict) else payload


def load_canonical_text(canonical_texts_path: Path, document_code: str) -> str:
    payload = json.loads(canonical_texts_path.read_text(encoding="utf-8"))
    entry = payload.get(document_code) or payload["documents"][document_code]
    return entry["text"]


def load_embedding_cache(cache_path: Path) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            cache[entry["h"]] = entry["v"]
    return cache


def boundary_scores(
    *,
    canonical_texts_path: Path,
    legal_units_path: Path,
    cache_path: Path,
    document_code: str,
) -> tuple[np.ndarray, np.ndarray]:
    legal_units = load_legal_units(legal_units_path)
    text = load_canonical_text(canonical_texts_path, document_code)
    segments = split_oversize_segments(
        segment_document_sentences(
            document_code=document_code,
            canonical_text=text,
            legal_units=legal_units,
        ),
        max_chunk_size=MAX_CHUNK_SIZE,
        canonical_text=text,
        encoding_name=DEFAULT_ENCODING,
    )
    cache = load_embedding_cache(cache_path)
    missing = [
        segment.text
        for segment in segments
        if hashlib.sha256(segment.text.encode("utf-8")).hexdigest() not in cache
    ]
    if missing:
        raise RuntimeError(f"{len(missing)} segments are absent from the cache")
    embeddings = np.asarray(
        [
            cache[hashlib.sha256(segment.text.encode("utf-8")).hexdigest()]
            for segment in segments
        ],
        dtype=float,
    )
    scores = compute_boundary_scores(embeddings, WINDOW_SIZE)
    order = sorted(scores)
    unit_starts = {
        unit["canonical_intervals"][0][0]
        for unit in legal_units
        if unit["document_code"] == document_code
        and unit["unit_type"] in UNIT_TYPES
        and unit.get("canonical_intervals")
    }
    values = np.array([scores[index] for index in order])
    at_unit_start = np.array([segments[index].start in unit_starts for index in order])
    return values, at_unit_start


def write_gdpr_similarity_profile(
    *,
    output_path: Path,
    canonical_texts_path: Path = CANONICAL_TEXTS_PATH,
    legal_units_path: Path = LEGAL_UNITS_PATH,
    cache_path: Path = EMBEDDING_CACHE_PATH,
    document_code: str = DOCUMENT_CODE,
) -> dict[str, Any]:
    apply_style()
    values, at_unit_start = boundary_scores(
        canonical_texts_path=canonical_texts_path,
        legal_units_path=legal_units_path,
        cache_path=cache_path,
        document_code=document_code,
    )
    thresholds = {p: float(np.percentile(values, p)) for p in PERCENTILES}

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), width_ratios=(1.9, 1.0))
    axes[0].plot(values, color=BLUE, linewidth=0.4, alpha=0.85)
    for percentile, colour in zip(PERCENTILES, (ORANGE, AQUA)):
        axes[0].axhline(
            thresholds[percentile], color=colour, linewidth=1.0, linestyle="--"
        )
        axes[0].text(
            len(values),
            thresholds[percentile],
            f" p{percentile}",
            color=colour,
            fontsize=7.5,
            va="center",
        )
    axes[0].set_title("(a) Similarity across the act", loc="left", pad=6)
    axes[0].set_xlabel(f"{len(values)} candidate boundaries, in reading order")
    axes[0].set_ylabel("Adjacent-window similarity")
    axes[0].margins(x=0.01)

    sns.kdeplot(x=values[at_unit_start], ax=axes[1], color=BLUE, fill=True, alpha=0.25)
    sns.kdeplot(x=values[~at_unit_start], ax=axes[1], color=AXIS, fill=True, alpha=0.15)
    axes[1].axvline(
        thresholds[50], color=ORANGE, linewidth=1.0, linestyle="--", ymax=0.78
    )
    axes[1].set_title("(b) Distribution by position", loc="left", pad=6)
    axes[1].set_xlabel("Adjacent-window similarity")
    axes[1].set_ylabel("Density")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.32)
    axes[1].text(
        0.03,
        0.97,
        f"article or recital start (n={int(at_unit_start.sum())})",
        transform=axes[1].transAxes,
        color=BLUE,
        fontsize=7.5,
        va="top",
    )
    axes[1].text(
        0.03,
        0.88,
        f"other positions (n={int((~at_unit_start).sum())})",
        transform=axes[1].transAxes,
        color=INK,
        fontsize=7.5,
        va="top",
    )

    figure.tight_layout()
    return {
        "boundaries": int(values.size),
        "mean": float(values.mean()),
        "thresholds": thresholds,
        "written": [str(path) for path in save(figure, output_path)],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the GDPR similarity profile.")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("reports/figures/gdpr_similarity_profile"),
    )
    parser.add_argument(
        "--canonical-texts-path", type=Path, default=CANONICAL_TEXTS_PATH
    )
    parser.add_argument("--legal-units-path", type=Path, default=LEGAL_UNITS_PATH)
    parser.add_argument("--cache-path", type=Path, default=EMBEDDING_CACHE_PATH)
    parser.add_argument("--document-code", default=DOCUMENT_CODE)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = write_gdpr_similarity_profile(
        output_path=args.output_path,
        canonical_texts_path=args.canonical_texts_path,
        legal_units_path=args.legal_units_path,
        cache_path=args.cache_path,
        document_code=args.document_code,
    )
    print("GDPR similarity profile complete: " + ", ".join(summary["written"]))


if __name__ == "__main__":
    main()
