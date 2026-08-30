from __future__ import annotations

from functools import lru_cache

import tiktoken


DEFAULT_ENCODING = "cl100k_base"


@lru_cache(maxsize=8)
def get_tokenizer(encoding_name: str = DEFAULT_ENCODING) -> tiktoken.Encoding:
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = DEFAULT_ENCODING) -> int:
    return len(get_tokenizer(encoding_name).encode(text))


def token_offsets(
    text: str,
    encoding_name: str = DEFAULT_ENCODING,
) -> tuple[list[int], str, int]:
    encoding = get_tokenizer(encoding_name)
    tokens = encoding.encode(text)
    decoded_text, offsets = encoding.decode_with_offsets(tokens)
    if decoded_text != text:
        raise ValueError("Tokenizer decoded text does not match canonical input text")
    return offsets, decoded_text, len(tokens)
