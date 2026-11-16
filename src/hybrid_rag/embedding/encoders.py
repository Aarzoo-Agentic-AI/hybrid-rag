"""Encoders that produce a dense and a sparse vector for each text in one pass.

``bge-m3`` computes both from a single forward pass, so one encoder serves both LangChain
wrappers (see ``adapters.py``). ``CachingEncoder`` remembers recent results, because
``QdrantVectorStore`` asks the dense and the sparse wrapper for the same texts one after
the other.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol

from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Encoding:
    dense: list[float]
    sparse: dict[int, float]  # token id -> weight; only non-zero weights


class DenseSparseEncoder(Protocol):
    @property
    def dense_dim(self) -> int: ...

    def encode(self, texts: list[str]) -> list[Encoding]: ...


class BgeM3Encoder:
    """``BAAI/bge-m3`` through FlagEmbedding. The model is loaded on first use."""

    DENSE_DIM = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def dense_dim(self) -> int:
        return self.DENSE_DIM

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                from FlagEmbedding import BGEM3FlagModel

                logger.info(
                    "Loading %s on %s (first run downloads ~2.3 GB)", self.model_name, self.device
                )
                self._model = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.device.startswith("cuda"),  # fp16 is only faster on GPU
                    devices=self.device,
                    normalize_embeddings=True,
                )
            return self._model

    def encode(self, texts: list[str]) -> list[Encoding]:
        if not texts:
            return []
        output = self._load().encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        if dense.ndim == 1:  # a single text comes back as a 1-D array
            dense = dense.reshape(1, -1)
        return [
            Encoding(
                dense=[float(x) for x in vector],
                sparse={
                    int(token): float(weight) for token, weight in weights.items() if weight > 0
                },
            )
            for vector, weights in zip(dense, output["lexical_weights"], strict=True)
        ]


_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


class HashingEncoder:
    """A deterministic stand-in for ``bge-m3`` (feature hashing of lower-cased words).

    It needs no download and is instant, so tests and offline demos can run the whole
    pipeline. It matches words, not meaning: use ``bge-m3`` for real retrieval.
    """

    def __init__(self, dense_dim: int = 384, sparse_buckets: int = 2**20) -> None:
        self._dense_dim = dense_dim
        self.sparse_buckets = sparse_buckets

    @property
    def dense_dim(self) -> int:
        return self._dense_dim

    def encode(self, texts: list[str]) -> list[Encoding]:
        return [self._encode_one(t) for t in texts]

    def _encode_one(self, text: str) -> Encoding:
        dense = [0.0] * self._dense_dim
        sparse: dict[int, float] = {}
        for token in _TOKEN.findall(text.lower()):
            digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
            dense[digest % self._dense_dim] += 1.0 if (digest >> 32) & 1 else -1.0
            bucket = digest % self.sparse_buckets
            sparse[bucket] = sparse.get(bucket, 0.0) + 1.0
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        sparse = {k: 1.0 + math.log(v) for k, v in sparse.items()}  # dampen repeated words
        return Encoding(dense=[x / norm for x in dense], sparse=sparse)


class CachingEncoder:
    """Wraps an encoder with a thread-safe LRU cache keyed by the exact text."""

    def __init__(self, inner: DenseSparseEncoder, max_entries: int = 4096) -> None:
        self.inner = inner
        self.max_entries = max_entries
        self._cache: OrderedDict[str, Encoding] = OrderedDict()
        self._lock = threading.Lock()
        self.encoded_count = 0  # texts actually sent to the inner encoder (for tests/metrics)

    @property
    def dense_dim(self) -> int:
        return self.inner.dense_dim

    def encode(self, texts: list[str]) -> list[Encoding]:
        results: dict[str, Encoding] = {}
        with self._lock:
            for text in texts:
                cached = self._cache.get(text)
                if cached is not None:
                    self._cache.move_to_end(text)
                    results[text] = cached
        missing = [t for t in dict.fromkeys(texts) if t not in results]
        if missing:
            # Encode outside the lock so other threads aren't blocked by the model.
            encoded = self._encode_uncached(
                missing,
                requested=len(texts),
                encoder=getattr(self.inner, "model_name", type(self.inner).__name__),
            )
            with self._lock:
                self.encoded_count += len(missing)
                for text, encoding in zip(missing, encoded, strict=True):
                    results[text] = encoding
                    self._store(text, encoding)
        return [results[t] for t in texts]

    @traceable(
        name="embed",
        run_type="embedding",
        process_inputs=lambda i: {
            "encoder": i["encoder"],
            "texts": len(i["texts"]),
            "served_from_cache": i["requested"] - len(i["texts"]),
        },
        process_outputs=lambda o: {
            "vectors": len(o),
            "dense_dim": len(o[0].dense) if o else 0,
            "avg_sparse_terms": round(sum(len(e.sparse) for e in o) / len(o), 1) if o else 0,
        },
    )
    def _encode_uncached(self, texts: list[str], requested: int, encoder: str) -> list[Encoding]:
        """The only place the model runs; traced (counts only, never text or vectors).

        ``requested`` and ``encoder`` are only there for the trace.
        """
        return self.inner.encode(texts)

    def _store(self, text: str, encoding: Encoding) -> None:
        if self.max_entries == 0:
            return
        self._cache[text] = encoding
        self._cache.move_to_end(text)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)
