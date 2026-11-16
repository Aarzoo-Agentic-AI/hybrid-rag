# Vector store (Qdrant)

This package stores chunks in Qdrant for hybrid search, and keeps exactly one version of each document. Re-indexing a file skips it if it's unchanged, replaces it if it changed, and repairs it if an earlier write was interrupted.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.store import build_document_index

settings = get_settings()
index = build_document_index(settings)  # connects and creates the collection
outcomes = ingest_paths(["data/samples"], index, settings.chunking)

index.list_documents()  # one DocumentInfo per document
index.delete(doc_id)  # remove a document's chunks
index.vector_store  # the LangChain QdrantVectorStore (hybrid mode), used by retrievers
```

From the command line:

```bash
uv run python scripts/ingest.py               # index data/samples
uv run python scripts/ingest.py my/folder a.pdf
uv run python scripts/ingest.py --list        # show what's indexed
uv run python scripts/ingest.py --force       # re-embed even unchanged files
uv run python scripts/ingest.py --reset       # empty the collection first
```

## Files

| File | Contents |
|---|---|
| [`qdrant.py`](qdrant.py) | `ensure_collection()`, `DocumentIndex`, `DocumentInfo`, `IndexResult` and the errors |
| [`__init__.py`](__init__.py) | `create_client()` and `build_document_index()` |
| [`../indexing.py`](../indexing.py) | `ingest_file()` / `ingest_paths()`, which load files with the ingestion package and index them, reporting bad files instead of stopping |

## Collection layout

One collection (`qdrant.collection`, default `documents`). Each chunk is one point:

| Part | Name | Details |
|---|---|---|
| Point ID | | The chunk's deterministic UUID (`chunk_id`) |
| Dense vector | `dense` | 1024 dimensions for `bge-m3`, cosine distance |
| Sparse vector | `sparse` | `bge-m3` token weights |
| Payload | `page_content` | Chunk text, including the contextual header |
| Payload | `metadata` | The chunk metadata from ingestion, plus `ingested_at` |

This is the payload layout that LangChain's `QdrantVectorStore` expects, and all writes go through `QdrantVectorStore.add_documents`. So the retrievers in step 4 read exactly what was written, with no conversion.

**Payload indexes** on `metadata.doc_id`, `doc_hash`, `source`, `file_type` and `chunk_index` keep filtered searches, deletes and the document list fast.

`ensure_collection()` creates the collection if needed and checks an existing one. If the vector size or vector names don't match the embedding model (for example after switching from `hashing` to `bge-m3`), it raises `CollectionConfigError` with instructions, instead of failing later on the first write.

## How `index()` keeps one version of each document

```
index(document)
  ├─ stored version has the same doc_hash, the same chunk_count,
  │  and all its chunks are present?              ──► "unchanged" (nothing embedded)
  └─ otherwise
       1. write all new chunks (upsert by chunk ID, overwriting old chunks in place)
       2. delete chunks of this doc_id whose doc_hash differs   ──► "added" / "updated"
```

- **Unchanged files cost nothing.** No embedding runs, which matters because embedding is the slow part.
- **No gap during updates.** New chunks are written before stale ones are deleted, so a document never disappears from search while it's being updated. A failure part-way leaves the old and new versions side by side, never neither.
- **Self-repairing.** If a write was interrupted, the chunk count doesn't match on the next run, so the document is indexed again.
- **Shorter documents clean up after themselves.** If a file shrinks from 10 chunks to 6, the 4 leftover chunks are deleted (`IndexResult.removed`).

A document's identity is its `doc_id`, which comes from the source file name (see the ingestion README). Re-uploading `report.pdf` therefore replaces the earlier `report.pdf`.

## Reading

| Method | Returns |
|---|---|
| `list_documents()` | One `DocumentInfo` per document (`doc_id`, `source`, `title`, `file_type`, `chunk_count`, `doc_hash`, `ingested_at`), read from each document's first chunk and sorted by source |
| `get_document(doc_id)` | One `DocumentInfo`, or `None` |
| `count_chunks(doc_id=None)` | Exact chunk count, for one document or for all |
| `health()` | Collection name, status and point count, for `/health` |

## Tests

- `tests/test_store.py` (unit tests): uses `QdrantClient(":memory:")` and the `hashing` encoder, so no Docker or model download is needed. It covers the collection layout and validation, add, skip, update with stale-chunk removal, force, interrupted-write repair, delete, list, reset, ingesting the samples twice (the second run embeds nothing), bad files, and a hybrid search for an exact term (`LR-200`).
- `tests/test_store_live.py` (`-m integration`): real `bge-m3` and the real Qdrant server, using a temporary collection that is deleted afterwards.
