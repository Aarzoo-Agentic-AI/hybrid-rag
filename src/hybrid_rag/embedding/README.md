# Embeddings

This package produces a **dense** vector (meaning) and a **sparse** vector (exact words) for each text, from a single model, and exposes them through LangChain's interfaces so `QdrantVectorStore` can use them directly.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.embedding import build_embeddings

models = build_embeddings(get_settings().embedding)  # build once per process
models.dense.embed_query("annual leave")  # list[float], 1024 values for bge-m3
models.sparse.embed_query("error E-4012")  # SparseVector(indices=[...], values=[...])
```

## Files

| File | Contents |
|---|---|
| [`encoders.py`](encoders.py) | `BgeM3Encoder`, `HashingEncoder` and `CachingEncoder`; each produces an `Encoding(dense, sparse)` |
| [`adapters.py`](adapters.py) | `DenseEmbeddings` (LangChain `Embeddings`) and `SparseEmbeddingsAdapter` (`langchain_qdrant.SparseEmbeddings`) |
| [`__init__.py`](__init__.py) | `build_embeddings()`, which returns an `EmbeddingModels` (encoder, dense, sparse) |

## One model, one pass, two vectors

`bge-m3` produces both vectors from the same forward pass:
- **Dense:** 1024 numbers, normalised, compared with cosine similarity. It finds passages with the same meaning in different words ("annual leave" and "vacation days").
- **Sparse:** a weight for each important token in the text, for example `E-4012`, `LR-200` or `encoder`. It finds exact terms that dense vectors tend to blur.

LangChain treats the two as separate objects, and `QdrantVectorStore` calls the dense wrapper and then the sparse wrapper on the same batch of texts. To avoid running the model twice, both wrappers share one `CachingEncoder`. The dense call encodes the texts and stores both vectors, and the sparse call reads them back from the cache.

```
QdrantVectorStore ──► DenseEmbeddings ──┐
                  └─► SparseEmbeddings ─┴─► CachingEncoder ──► BgeM3Encoder (one pass)
```

`CachingEncoder` is an LRU cache (default 4096 texts, `embedding.cache_size`) keyed by the exact text. It's safe to use from several threads at once, and the model runs outside the lock. `encoded_count` reports how many texts actually reached the model; the tests use it to check that unchanged files aren't embedded again.

## Providers

| `embedding.provider` | What it is | When to use it |
|---|---|---|
| `bge-m3` (default) | `BAAI/bge-m3` through FlagEmbedding. About 2.3 GB, downloaded to the Hugging Face cache on first use and loaded lazily | Real retrieval |
| `hashing` | `HashingEncoder`: feature hashing of lower-cased words (384-dimensional dense vector plus word-level sparse vector). Instant, with no download | Unit tests, CI and offline demos. **It matches words, not meaning** |

Switching provider changes the vector size (1024 versus 384). The store detects the mismatch and asks you to re-create the collection (`scripts/ingest.py --reset`).

## Configuration

```yaml
embedding:
  provider: bge-m3
  model: BAAI/bge-m3
  batch_size: 16       # texts per model call (also the Qdrant upload batch size)
  max_length: 1024     # tokens per text; longer text is truncated
  device: cpu          # "cuda" uses a GPU, with fp16 automatically
  cache_size: 4096
```

**Measured on the development machine (CPU only):**

| Operation | Time |
|---|---|
| First run: download and load the model | ~6.5 min (one-time; depends on your connection) |
| Load the model from the local cache | ~10 s, once per process |
| Embed the 10 catalogue chunks | ~3 s |
| One hybrid search, including embedding the query | ~0.2 s |

A GPU is much faster for large ingestion jobs; see "Scaling to production" in [FEATURE.md](../../../FEATURE.md).

**Harmless warnings on first use:**
- `You are sending unauthenticated requests to the HF Hub`: downloads work without a token. Setting `HF_TOKEN` in your environment only raises Hugging Face's rate limits.
- `To support symlinks on Windows…`: without Windows Developer Mode, the Hugging Face cache stores real copies of files instead of links. That uses a little more disk space but works the same.

## Tests

- `tests/test_embedding.py` (unit tests): caching, deduplication, eviction, the single pass shared by both wrappers, and sorted sparse indices.
- `tests/test_store_live.py` (`-m integration`): real `bge-m3` vectors (1024 dimensions, normalised, sparse weights) and a check that meaning is captured ("annual leave" is closer to "vacation days" than to "robot arm payload").
