import math

import pytest

from hybrid_rag.config import EmbeddingSettings
from hybrid_rag.embedding import (
    CachingEncoder,
    DenseEmbeddings,
    Encoding,
    HashingEncoder,
    SparseEmbeddingsAdapter,
    build_embeddings,
)


class CountingEncoder:
    """Records every batch it is asked to encode."""

    dense_dim = 4

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode(self, texts: list[str]) -> list[Encoding]:
        self.batches.append(list(texts))
        return [Encoding(dense=[float(len(t))] * 4, sparse={len(t): 1.0}) for t in texts]


def test_hashing_encoder_is_deterministic_and_normalised() -> None:
    encoder = HashingEncoder(dense_dim=64)

    a, b = encoder.encode(["Error E-4012 on the LR-200", "Error E-4012 on the LR-200"])

    assert a == b
    assert len(a.dense) == 64
    assert math.isclose(math.sqrt(sum(x * x for x in a.dense)), 1.0)
    assert a.sparse and all(w > 0 for w in a.sparse.values())


def test_hashing_encoder_keeps_hyphenated_codes_as_one_token() -> None:
    encoder = HashingEncoder()
    (code,) = encoder.encode(["E-4012"])
    (parts,) = encoder.encode(["E 4012"])

    assert len(code.sparse) == 1
    assert code.sparse.keys() != parts.sparse.keys()


def test_cache_encodes_each_text_once() -> None:
    inner = CountingEncoder()
    cache = CachingEncoder(inner, max_entries=100)

    cache.encode(["a", "bb"])
    cache.encode(["bb", "a", "ccc"])

    assert inner.batches == [["a", "bb"], ["ccc"]]
    assert cache.encoded_count == 3


def test_cache_deduplicates_within_a_batch() -> None:
    inner = CountingEncoder()
    results = CachingEncoder(inner).encode(["x", "x", "y"])

    assert inner.batches == [["x", "y"]]
    assert results[0] == results[1]


@pytest.mark.parametrize("max_entries", [0, 1])
def test_small_or_disabled_cache_never_encodes_a_text_twice_in_one_call(max_entries: int) -> None:
    inner = CountingEncoder()
    results = CachingEncoder(inner, max_entries=max_entries).encode(["a", "bb", "ccc"])

    assert inner.batches == [["a", "bb", "ccc"]]
    assert [r.dense[0] for r in results] == [1.0, 2.0, 3.0]


def test_cache_evicts_least_recently_used() -> None:
    inner = CountingEncoder()
    cache = CachingEncoder(inner, max_entries=2)

    cache.encode(["a"])
    cache.encode(["bb"])
    cache.encode(["a"])  # "a" is now the most recently used
    cache.encode(["ccc"])  # evicts "bb"
    cache.encode(["a", "bb"])

    assert inner.batches[-1] == ["bb"]


def test_dense_and_sparse_wrappers_share_one_encoding_pass() -> None:
    inner = CountingEncoder()
    cache = CachingEncoder(inner)
    dense, sparse = DenseEmbeddings(cache), SparseEmbeddingsAdapter(cache)
    texts = ["one", "three"]

    vectors = dense.embed_documents(texts)
    sparse_vectors = sparse.embed_documents(texts)  # what QdrantVectorStore does next

    assert inner.batches == [["one", "three"]]
    assert vectors[0] == [3.0] * 4
    assert sparse_vectors[1].indices == [5]
    assert sparse_vectors[1].values == [1.0]


def test_sparse_indices_are_sorted() -> None:
    class Unsorted(CountingEncoder):
        def encode(self, texts):
            return [Encoding(dense=[0.0] * 4, sparse={9: 0.9, 2: 0.2, 5: 0.5}) for _ in texts]

    vector = SparseEmbeddingsAdapter(Unsorted()).embed_query("q")

    assert vector.indices == [2, 5, 9]
    assert vector.values == [0.2, 0.5, 0.9]


def test_build_embeddings_hashing_provider_needs_no_model() -> None:
    models = build_embeddings(EmbeddingSettings(provider="hashing"))

    assert models.dense_dim == 384
    assert len(models.dense.embed_query("hello")) == 384
    assert models.dense.encoder is models.sparse.encoder


def test_build_embeddings_bge_m3_does_not_load_the_model_until_used() -> None:
    models = build_embeddings(EmbeddingSettings(provider="bge-m3"))

    assert models.dense_dim == 1024
    assert models.encoder.inner._model is None  # nothing downloaded or loaded yet
