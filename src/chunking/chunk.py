from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    strategy: str
    config_id: str
    text: str
    token_count: int
    canonical_intervals: list[list[int]]
    source_text_sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "chunk_id": self.chunk_id,
            "strategy": self.strategy,
            "config_id": self.config_id,
            "text": self.text,
            "token_count": self.token_count,
            "canonical_intervals": self.canonical_intervals,
            "source_text_sha256": self.source_text_sha256,
            "metadata": self.metadata,
        }
        if self.embedding_text is not None:
            value["embedding_text"] = self.embedding_text
        return value


def chunk_id_for_unit(config_id: str, unit_id: str) -> str:
    safe_unit_id = re.sub(r"_+", "_", re.sub(r"[^0-9A-Za-z]+", "_", unit_id)).strip("_")
    return f"{config_id}_{safe_unit_id}"
