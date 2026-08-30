from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from utils.env import load_dotenv


class OpenRouterEmbedder:
    def __init__(
        self,
        config: dict[str, Any],
        cache_dir: Path | None = None,
    ) -> None:
        load_dotenv()
        gateway = config["models"]["gateway"]
        embedding_config = config["models"]["embedding"]
        api_key = os.environ.get(gateway["api_key_env"])
        if not api_key:
            raise RuntimeError(f"{gateway['api_key_env']} is not set")

        headers = {}
        if referer := os.environ.get(gateway.get("http_referer_env", "")):
            headers["HTTP-Referer"] = referer
        if title := os.environ.get(gateway.get("title_env", "")):
            headers["X-Title"] = title

        self.client = OpenAI(
            base_url=gateway["base_url"],
            api_key=api_key,
            default_headers=headers or None,
        )
        self.model = embedding_config["model"]
        self.dimensions = embedding_config.get("dimensions")
        self.batch_size = embedding_config.get("batch_size", 100)
        self._cache: dict[str, list[float]] = {}
        # Persistent text cache for sentence and query embeddings.
        self._cache_path = self._resolve_cache_path(cache_dir)
        self._load_disk_cache()

    def _resolve_cache_path(self, cache_dir: Path | None) -> Path | None:
        if cache_dir is None:
            return None
        model_slug = re.sub(r"[^A-Za-z0-9._-]", "_", self.model)
        dimension_slug = "default" if self.dimensions is None else str(self.dimensions)
        return cache_dir / f"{model_slug}_{dimension_slug}.jsonl"

    def _load_disk_cache(self) -> None:
        if self._cache_path is None or not self._cache_path.exists():
            return
        with self._cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entry = json.loads(line)
                self._cache[entry["h"]] = entry["v"]

    def _append_disk_cache(self, entries: list[tuple[str, list[float]]]) -> None:
        if self._cache_path is None or not entries:
            return
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as handle:
            for text_hash, vector in entries:
                handle.write(json.dumps({"h": text_hash, "v": vector}) + "\n")

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        hashes = {text: self._text_hash(text) for text in dict.fromkeys(texts)}
        missing_texts = [
            text for text, text_hash in hashes.items() if text_hash not in self._cache
        ]
        for start in range(0, len(missing_texts), self.batch_size):
            batch = missing_texts[start : start + self.batch_size]
            kwargs: dict[str, Any] = {"model": self.model, "input": batch}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            response = self.client.embeddings.create(**kwargs)
            ordered_data = sorted(response.data, key=lambda item: item.index)
            fetched: list[tuple[str, list[float]]] = []
            for text, item in zip(batch, ordered_data):
                self._cache[hashes[text]] = item.embedding
                fetched.append((hashes[text], item.embedding))
            self._append_disk_cache(fetched)
        return [self._cache[hashes[text]] for text in texts]

    def embed_single_timed(self, text: str) -> tuple[list[float], float]:
        """Embed one uncached input, time it, and cache the result."""
        kwargs: dict[str, Any] = {"model": self.model, "input": [text]}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        started = time.perf_counter()
        response = self.client.embeddings.create(**kwargs)
        elapsed_seconds = time.perf_counter() - started
        vector = response.data[0].embedding
        text_hash = self._text_hash(text)
        self._cache[text_hash] = vector
        self._append_disk_cache([(text_hash, vector)])
        return vector, elapsed_seconds
