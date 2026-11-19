"""Token counting for chunk sizes, with the embedding model's own tokenizer."""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from typing import Any, Protocol


class TokenCounter(Protocol):
    name: str

    def count(self, text: str) -> int: ...


class ApproxTokenCounter:
    """About 4 characters per token, which is close for English with bge-m3. No model needed."""

    name = "approx"

    def count(self, text: str) -> int:
        return math.ceil(len(text) / 4)


class HFTokenCounter:
    """Counts with a Hugging Face tokenizer (only the tokenizer files are loaded, not the model)."""

    def __init__(self, model: str) -> None:
        self.name = model
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    def _get(self) -> Any:
        with self._lock:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(self.name)
                tokenizer.model_max_length = 10**9  # counting only; silences length warnings
                self._tokenizer = tokenizer
            return self._tokenizer

    def count(self, text: str) -> int:
        return len(self._get()(text, add_special_tokens=False)["input_ids"])


@lru_cache(maxsize=4)
def get_token_counter(name: str) -> TokenCounter:
    return ApproxTokenCounter() if name == "approx" else HFTokenCounter(name)
