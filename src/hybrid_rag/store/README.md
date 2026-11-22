# Vector store (Qdrant)

This package stores chunks in Qdrant for hybrid search, and keeps exactly one version of each document. Re-indexing a file skips it if it's unchanged, replaces it if it changed, and repairs it if an earlier write was interrupted. Every read is filtered by the caller's permissions inside Qdrant.

```python
from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline
from hybrid_rag.store import build_document_index

settings = get_settings()
index = build_document_index(settings)  # connects and creates the collection
pipeline = IngestionPipeline.from_settings(settings)  # Docling, privacy scan, registry, tenant
outcomes = ingest_paths(["data/pdfs"], index, pipeline)

manager = Principal("dummymocktest", "u-1042", clearance="internal", roles={"people_manager"})
index.search("merit matrix", manager, k=5)  # hybrid search over what the caller may read
index.list_documents(manager)  # one DocumentInfo per readable document
admin = Principal.tenant_admin("dummymocktest")
index.delete(doc_id, admin)  # tenant admins only
```

From the command line:

```bash
uv run python scripts/ingest.py               # index documents.path (data/pdfs)
uv run python scripts/ingest.py my/folder a.pdf
uv run python scripts/ingest.py --list        # show what's indexed
uv run python scripts/ingest.py --force       # re-embed even unchanged files
uv run python scripts/ingest.py --reset       # empty the collection (all tenants) first
uv run python scripts/ingest.py --tenant other --registry other/registry.yaml other/
```

`--list` shows the tenant's documents as its administrator, with each one's classification, readers and `access_version`.

## Files

| File | Contents |
|---|---|
| [`qdrant.py`](qdrant.py) | `ensure_collection()`, `DocumentIndex`, `DocumentInfo`, `IndexResult` and the errors |
| [`security.py`](security.py) | `security_filter(principal)`, the access rule as a Qdrant filter, and `scoped()` |
| [`schema.py`](schema.py) | Vector names, payload keys and payload indexes |
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

**Payload indexes** on these fields keep filtered searches, deletes and the document list fast:
- **Access:** `metadata.tenant_id` with `is_tenant=True` (Qdrant keeps each tenant's points together, so a tenant-filtered search reads only that tenant's data), `classification_rank` (integer), `visibility`, `allowed_groups`, `allowed_roles`, `allowed_users`.
- **Document:** `metadata.doc_id`, `doc_hash`, `source`, `file_type`, `chunk_index`, `pii_scan.status`.
- **Policy:** `policy_id`, `policy_category`, `jurisdiction`, `applies_to`, `is_current`.
- **Section:** `section_id`, `chunk_type`.

The policy and section fields are for the relevance filters and the small-to-big retrieval in step 4.

`ensure_collection()` creates the collection if needed and checks an existing one. If the vector size or vector names don't match the embedding model (for example after switching from `hashing` to `bge-m3`), it raises `CollectionConfigError` with instructions, instead of failing later on the first write.

## How `index()` keeps one version of each document

```
index(document)
  ├─ stored version has the same doc_hash and pipeline_hash, the same
  │  chunk_count, and all its chunks are present?
  │    ├─ same access_hash ──► "unchanged" (nothing embedded)
  │    └─ new access rules ──► set_access(): "access_updated" (nothing embedded)
  └─ otherwise
       1. write all new chunks (upsert by chunk ID, overwriting old chunks in place)
       2. delete chunks of this doc_id that aren't in the new version   ──► "added" / "updated"
```

- **Unchanged files cost nothing.** `ingest_file()` asks `current_version()` first. If the stored version has the same file hash and the same `pipeline_hash` (chunking settings plus privacy policy) and is complete, the file isn't even loaded, scanned or embedded. A re-run with nothing to do takes under a second.
- **Settings changes are picked up.** A different chunk size or new privacy rules change the `pipeline_hash`, so unchanged files are processed again on the next run.
- **No gap during updates.** New chunks are written before stale ones are deleted, so a document never disappears from search while it's being updated. A failure part-way leaves the old and new versions side by side, never neither.
- **Self-repairing.** If a write was interrupted, the chunk count doesn't match on the next run, so the document is indexed again.
- **Shorter documents clean up after themselves.** If a file shrinks from 10 chunks to 6, or is re-chunked into fewer pieces, every chunk that isn't part of the new version is deleted (`IndexResult.removed`).

A document's identity is its `doc_id`, which comes from the tenant and the source file name (see the ingestion README). Re-uploading `report.pdf` to the same tenant therefore replaces the earlier `report.pdf`; another tenant's `report.pdf` is a different document.

## Access control

Permissions are enforced **inside the search**: every read combines the caller's security filter with whatever else it filters on. A chunk the caller can't read is never returned, counted or deleted, so it can't reach an answer, a citation or a trace. FEATURE.md section 11 has the design.

```
tenant_id == caller.tenant                              always
AND classification_rank <= caller.clearance             a ceiling no grant can lift
AND ( visibility == "tenant"                            any one grant is enough
      OR caller.user_id in allowed_users
      OR caller.groups  ∩  allowed_groups
      OR caller.roles   ∩  allowed_roles )
```

- **`Principal`** ([`../access.py`](../access.py)): tenant, user ID, clearance, groups and roles. It is built on the server from the caller's identity (API keys in step 6, JWT claims later), never from request fields. A principal without a tenant, a user or a valid clearance can't be created. `Principal.tenant_admin(tenant)` sees every document of its own tenant and nothing else; the ingestion CLI uses it.
- **One filter:** `security_filter(principal)` in [`security.py`](security.py). `scoped(principal, *conditions)` ANDs relevance filters (category, jurisdiction, ...) with it, so they can narrow the results but never widen them.
- **Deny by default:** a document without access rules has classification `restricted` and no grants, so only tenant admins see it. Chunks without access fields (indexed before step 3b) match nobody.
- **Permission changes don't re-embed.** `set_access(doc_id, policy)` overwrites the access fields on all the document's chunks in one `set_payload` call and bumps `access_version`. Ingestion calls it when a file's content is unchanged but its access rules in the registry changed. The tenant never changes, because it is part of the `doc_id`. Qdrant updates each point atomically, but not all of them as one transaction; if it fails part-way, the next ingest repeats it, because the stored `access_hash` still differs.
- **Writes are system operations.** `index()`, `set_access()` and `current_version()` work by `doc_id` for ingestion and aren't exposed to callers. `delete()` needs a tenant-admin principal.

## Reading

All of these take a `Principal` and return only what it may read.

| Method | Returns |
|---|---|
| `search(query, principal, k=4, mode="hybrid", conditions=None, dense_limit=None, sparse_limit=None)` | Dense, sparse or hybrid search over readable chunks (hybrid: two prefetches fused by Qdrant with RRF, each with the security filter). Results carry `retrieval_mode`, `retrieval_score` and `retrieval_rank`. `conditions` are relevance filters. The retrievers in [`../retrieval/`](../retrieval/README.md) build on this |
| `section_chunks(section_id, principal)` | The readable chunks of one section, in order, for small-to-big retrieval |
| `list_documents(principal)` | One `DocumentInfo` per readable document, sorted by source: `doc_id`, `source`, `title`, `chunk_count`, `doc_hash`, `ingested_at`, the privacy status, and the access fields (`tenant_id`, `classification`, `visibility`, grants, `access_version`, `access_updated_at`) |
| `get_document(doc_id, principal)` | One `DocumentInfo`, or `None` if it doesn't exist or isn't readable (the two cases look the same) |
| `count_chunks(principal, doc_id=None)` | Exact count of readable chunks, for one document or for all |
| `health()` | Collection name, status and total point count, for the operations-only `/health` |

`build_document_index()` raises `StoreError` with a hint (`docker compose up -d`) when Qdrant can't be reached.

## Tracing

With LangSmith tracing on, each `ingest_paths()` call is one trace:

```
ingest                      files, and how many were added / updated / unchanged / failed
└─ ingest_file              one per PDF: status, chunks, stale chunks removed, error
   ├─ load_document         title, pages, sections, chunks, definitions, warnings, privacy status
   │  ├─ parse              pages, blocks, tables (from the cache when the file was parsed before)
   │  └─ sanitize           blocks scanned, status, redactions by type (quarantine: error, types only)
   ├─ index_document        status, chunks, removed, access version
   │  └─ qdrant_upsert      chunks, batch size
   │     └─ embed           texts embedded, texts served from cache, dense size
   └─ set_access            only when just the permissions changed: classification,
                            visibility, number of grants, new access version
```

An unchanged file has only its `ingest_file` run, because it's skipped before loading. User IDs in grants are never traced, only counted. A quarantined file has no `index_document` run, because it never reaches the store. The runs record counts, names, statuses and timings only, never document text, vectors or local paths. `tests/test_tracing.py` checks the tree and the privacy rule with a mock LangSmith client. See [`../tracing.py`](../tracing.py).

## Tests

- `tests/test_store.py` (unit tests): uses `QdrantClient(":memory:")` and the `hashing` encoder, so no Docker or model download is needed. It covers the collection layout and validation, add, skip, update with stale-chunk removal, force, interrupted-write repair, delete, list, reset, ingesting the samples twice (the second run embeds nothing), bad files, and a hybrid search for an exact term (`LR-200`).
- `tests/test_access.py` (unit tests, in-memory Qdrant): the **leakage tests**. Two tenants each have a `handbook.pdf`, plus documents for every kind of permission: tenant-wide, group, role, a single user, a `restricted` document and an unlisted (deny-by-default) one. For 12 callers they check that:
  - list, get, count and an exact-match search for every document's unique term return only what the rule allows;
  - a grant never lifts the classification ceiling;
  - relevance conditions never widen the filter;
  - only tenant admins can delete;
  - the filter agrees with `Principal.can_read()` on every chunk;
  - a permission change updates every chunk without loading or embedding anything.
- `tests/test_store_live.py` (`-m integration`): real `bge-m3` and the real Qdrant server, using a temporary collection that is deleted afterwards. It also runs the leakage matrix against the server, with its `is_tenant` index.
