"""The Qdrant collection and ``DocumentIndex``: add, update, skip, delete, list and search.

One collection holds every chunk of every tenant as a point with:
- a named dense vector ``dense`` (cosine) and a named sparse vector ``sparse``,
- the payload LangChain's ``QdrantVectorStore`` uses: ``page_content`` plus ``metadata``
  (the chunk metadata from ingestion, including the access fields, plus ``ingested_at``,
  ``access_version`` and ``access_updated_at``).

Writes go through ``QdrantVectorStore.add_documents`` so the LangChain store, which the
retrievers use in step 4, sees exactly the same data.

Every read that returns or counts chunks takes a ``Principal`` and applies the security
filter (``security.py``). Ingestion's own bookkeeping (``index``, ``current_version``,
``set_access``) works by tenant-scoped ``doc_id`` and is not exposed to callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from hybrid_rag.access import ACCESS_FIELDS, AccessDeniedError, AccessPolicy, Principal
from hybrid_rag.embedding import EmbeddingModels
from hybrid_rag.ingestion import IngestedDocument
from hybrid_rag.store.schema import (
    CONTENT_KEY,
    DENSE_VECTOR,
    METADATA_KEY,
    PAYLOAD_INDEXES,
    SPARSE_VECTOR,
    match,
)
from hybrid_rag.store.security import scoped
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)

SearchMode = Literal["dense", "sparse", "hybrid"]


class StoreError(Exception):
    """Base class for vector store errors."""


class CollectionConfigError(StoreError):
    """The existing collection doesn't match the embedding model (e.g. a different vector size)."""


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    source: str
    title: str
    file_type: str
    chunk_count: int
    doc_hash: str
    ingested_at: str
    pii_status: str = ""  # clean / redacted / not_scanned
    pii_policy: str = ""  # privacy settings the document was scanned with
    pipeline_hash: str = ""  # chunking + privacy settings the chunks were made with
    tenant_id: str = ""
    classification: str = ""
    visibility: str = ""
    allowed_groups: list[str] = field(default_factory=list)
    allowed_roles: list[str] = field(default_factory=list)
    allowed_users: list[str] = field(default_factory=list)
    access_hash: str = ""
    access_version: int = 0  # 1 when first indexed, +1 on every permission change
    access_updated_at: str = ""


@dataclass(frozen=True)
class IndexResult:
    doc_id: str
    source: str
    # "access_updated": same content, new permissions; written without re-embedding.
    status: Literal["added", "updated", "unchanged", "access_updated"]
    chunks: int
    removed: int = 0  # stale chunks deleted from the previous version
    access_version: int = 0


def ensure_collection(client: QdrantClient, name: str, dense_dim: int) -> bool:
    """Create the hybrid collection if it doesn't exist. Returns True if it was created.

    Raises ``CollectionConfigError`` if an existing collection has the wrong layout.
    """
    if client.collection_exists(name):
        _validate_collection(client, name, dense_dim)
        created = False
    else:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dense_dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={SPARSE_VECTOR: models.SparseVectorParams()},
        )
        logger.info("Created collection %r (dense dim %d)", name, dense_dim)
        created = True

    for field_name, schema in PAYLOAD_INDEXES.items():
        client.create_payload_index(name, field_name=field_name, field_schema=schema)
    return created


def _validate_collection(client: QdrantClient, name: str, dense_dim: int) -> None:
    params = client.get_collection(name).config.params
    vectors = params.vectors if isinstance(params.vectors, dict) else {}
    dense = vectors.get(DENSE_VECTOR)
    if dense is None or SPARSE_VECTOR not in (params.sparse_vectors or {}):
        raise CollectionConfigError(
            f"Collection {name!r} exists but has no {DENSE_VECTOR!r} and {SPARSE_VECTOR!r} "
            "vectors. Use a different collection name or delete it (scripts/ingest.py --reset)."
        )
    if dense.size != dense_dim:
        raise CollectionConfigError(
            f"Collection {name!r} stores {dense.size}-dimensional vectors, but the embedding "
            f"model produces {dense_dim}. This happens after switching embedding models; "
            "re-create the collection (scripts/ingest.py --reset)."
        )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class DocumentIndex:
    """Stores ingested documents in Qdrant and keeps one version of each document."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str,
        embeddings: EmbeddingModels,
        batch_size: int = 16,
    ) -> None:
        self.client = client
        self.collection = collection
        self.embeddings = embeddings
        self.batch_size = batch_size
        ensure_collection(client, collection, embeddings.dense_dim)
        self.vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=embeddings.dense,
            sparse_embedding=embeddings.sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name=DENSE_VECTOR,
            sparse_vector_name=SPARSE_VECTOR,
            content_payload_key=CONTENT_KEY,
            metadata_payload_key=METADATA_KEY,
            validate_collection_config=False,  # done by ensure_collection, without embedding text
        )

    # ------------------------------------------------------------ writes

    @traceable(
        name="index_document",
        process_inputs=lambda i: {
            "source": i["document"].source,
            "doc_id": i["document"].doc_id,
            "chunks": len(i["document"].chunks),
            "force": i["force"],
        },
        process_outputs=lambda r: (
            {
                "status": r.status,
                "chunks": r.chunks,
                "removed": r.removed,
                "access_version": r.access_version,
            }
            if r
            else {}
        ),
    )
    def index(self, document: IngestedDocument, *, force: bool = False) -> IndexResult:
        """Add a document, replace an older version of it, or skip it if it hasn't changed.

        New chunks are written first (their IDs are deterministic, so they overwrite the old
        chunks in place), and only then are leftover chunks of the old version deleted. A
        failure part-way therefore never leaves the document missing from the index.

        If only the permissions changed, they are updated in place, without embedding.
        """
        existing = self._stored(document.doc_id)
        access_hash = document.access.hash
        if existing is not None and not force and self._is_complete(existing, document):
            if existing.access_hash == access_hash:
                logger.info("%s unchanged, skipped", document.source)
                return IndexResult(
                    document.doc_id,
                    document.source,
                    "unchanged",
                    existing.chunk_count,
                    access_version=existing.access_version,
                )
            version = self.set_access(document.doc_id, document.access)
            return IndexResult(
                document.doc_id,
                document.source,
                "access_updated",
                existing.chunk_count,
                access_version=version,
            )

        now = _now()
        if existing is None:
            version, access_updated_at = 1, now
        elif existing.access_hash != access_hash:
            version, access_updated_at = existing.access_version + 1, now
        else:
            version, access_updated_at = existing.access_version, existing.access_updated_at
        chunks = [
            Document(
                page_content=c.page_content,
                metadata={
                    **c.metadata,
                    "ingested_at": now,
                    "access_version": version,
                    "access_updated_at": access_updated_at,
                },
                id=c.id,
            )
            for c in document.chunks
        ]
        self._write_chunks(chunks, self.batch_size)
        # Leftovers are this document's chunks that aren't part of the new version: from an
        # older file, or from the same file chunked differently (new chunking/privacy rules).
        removed = self._delete_where(
            must=[match("doc_id", document.doc_id)],
            must_not=[models.HasIdCondition(has_id=[c.id for c in chunks])],
        )
        status: Literal["added", "updated"] = "added" if existing is None else "updated"
        logger.info(
            "%s %s: %d chunk(s), %d stale removed", document.source, status, len(chunks), removed
        )
        return IndexResult(
            document.doc_id, document.source, status, len(chunks), removed, access_version=version
        )

    @traceable(
        name="set_access",
        process_inputs=lambda i: {
            "doc_id": i["doc_id"],
            "classification": i["access"].classification,
            "visibility": i["access"].visibility,
            "groups": len(i["access"].allowed_groups),
            "roles": len(i["access"].allowed_roles),
            "users": len(i["access"].allowed_users),
        },
        process_outputs=lambda v: {"access_version": v},
    )
    def set_access(self, doc_id: str, access: AccessPolicy) -> int:
        """Change a document's permissions on all its chunks, without re-embedding anything.

        One ``set_payload`` call, filtered by ``doc_id``, overwrites the access fields and
        bumps ``access_version``. The tenant never changes: it is part of the ``doc_id``.
        Returns the new ``access_version``.
        """
        existing = self._stored(doc_id)
        if existing is None:
            raise StoreError(f"No document with ID {doc_id} is indexed")
        payload = {
            k: v
            for k, v in access.metadata(existing.tenant_id).items()
            if k in ACCESS_FIELDS and k != "tenant_id"
        }
        version = existing.access_version + 1
        payload |= {"access_version": version, "access_updated_at": _now()}
        self.client.set_payload(
            self.collection,
            payload=payload,
            points=models.Filter(
                must=[match("doc_id", doc_id), match("tenant_id", existing.tenant_id)]
            ),
            key=METADATA_KEY,  # update fields inside "metadata", keep the rest
            wait=True,
        )
        logger.info("%s permissions changed (access_version %d)", existing.source, version)
        return version

    def delete(self, doc_id: str, principal: Principal) -> int:
        """Delete every chunk of a document. Only tenant admins may delete, and only within
        their tenant. Returns the number of chunks deleted (0 if the document isn't visible)."""
        if not principal.admin:
            raise AccessDeniedError("Only tenant administrators can delete documents")
        return self._delete_where(must=[scoped(principal, match("doc_id", doc_id))])

    def reset(self) -> None:
        """Delete the whole collection (every tenant) and create it again, empty."""
        self.client.delete_collection(self.collection)
        ensure_collection(self.client, self.collection, self.embeddings.dense_dim)

    # ------------------------------------------------------------- reads

    def search(
        self,
        query: str,
        principal: Principal,
        *,
        k: int = 4,
        mode: SearchMode = "hybrid",
        conditions: list[models.Condition] | None = None,
        dense_limit: int | None = None,
        sparse_limit: int | None = None,
    ) -> list[Document]:
        """Search the chunks ``principal`` may read; the best ``k``, best first.

        - ``dense``: cosine similarity of bge-m3 dense vectors (meaning).
        - ``sparse``: bge-m3 lexical weights (exact terms such as "HR-TRV-003" or "COBRA").
        - ``hybrid``: both searches as prefetches (``dense_limit`` and ``sparse_limit``
          candidates, default ``k`` each), fused by Qdrant with Reciprocal Rank Fusion.

        ``conditions`` are relevance filters; they are ANDed with the security filter, which is
        applied to every prefetch and to the final query, so they can only narrow it. Each
        result's metadata gets ``retrieval_mode``, ``retrieval_score`` and ``retrieval_rank``
        (1-based).
        """
        selector = scoped(principal, *(conditions or []))
        (encoding,) = self.embeddings.encoder.encode([query])
        dense = encoding.dense
        sparse = models.SparseVector(
            indices=list(encoding.sparse.keys()), values=list(encoding.sparse.values())
        )
        common: dict[str, Any] = {
            "query_filter": selector,
            "limit": k,
            "with_payload": True,
            "with_vectors": False,
        }
        if mode == "dense":
            response = self.client.query_points(
                self.collection, query=dense, using=DENSE_VECTOR, **common
            )
        elif mode == "sparse":
            if not encoding.sparse:  # nothing to match lexically (e.g. only stop words)
                return []
            response = self.client.query_points(
                self.collection, query=sparse, using=SPARSE_VECTOR, **common
            )
        elif mode == "hybrid":
            prefetch = [
                models.Prefetch(
                    query=dense, using=DENSE_VECTOR, filter=selector, limit=dense_limit or k
                )
            ]
            if encoding.sparse:
                prefetch.append(
                    models.Prefetch(
                        query=sparse, using=SPARSE_VECTOR, filter=selector, limit=sparse_limit or k
                    )
                )
            response = self.client.query_points(
                self.collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                **common,
            )
        else:
            raise ValueError(f"Unknown search mode {mode!r}; use dense, sparse or hybrid")
        return [
            _to_document(point, mode=mode, rank=rank)
            for rank, point in enumerate(response.points, start=1)
        ]

    def section_chunks(self, section_id: str, principal: Principal) -> list[Document]:
        """Every readable chunk of one section, in order (for small-to-big retrieval)."""
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=scoped(principal, match("section_id", section_id)),
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        docs = [_to_document(p) for p in points]
        return sorted(docs, key=lambda d: d.metadata.get("chunk_index", 0))

    def get_document(self, doc_id: str, principal: Principal) -> DocumentInfo | None:
        """The document, or ``None`` if it doesn't exist or ``principal`` may not read it."""
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=scoped(principal, match("doc_id", doc_id), match("chunk_index", 0)),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return _document_info(points[0].payload) if points else None

    def list_documents(self, principal: Principal) -> list[DocumentInfo]:
        """The documents ``principal`` may read (from each one's first chunk), sorted by source."""
        documents: list[DocumentInfo] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=scoped(principal, match("chunk_index", 0)),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(_document_info(p.payload) for p in points)
            if offset is None:
                return sorted(documents, key=lambda d: d.source)

    def count_chunks(self, principal: Principal, doc_id: str | None = None) -> int:
        """Chunks ``principal`` may read: of one document, or all of them."""
        conditions = [match("doc_id", doc_id)] if doc_id else []
        selector = scoped(principal, *conditions)
        return self.client.count(self.collection, count_filter=selector, exact=True).count

    def health(self) -> dict[str, Any]:
        """Collection status for ``/health`` (operations only: no per-tenant data)."""
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "status": str(info.status.value if hasattr(info.status, "value") else info.status),
            "points": info.points_count,
        }

    # ------------------------------------------------- ingestion internals

    def current_version(
        self, doc_id: str, doc_hash: str, pipeline_hash: str
    ) -> DocumentInfo | None:
        """The stored document if it is this exact file, made with these settings, and complete.

        Lets ingestion skip an unchanged file before loading, scanning or chunking it. The
        caller compares ``access_hash`` itself, to update permissions without re-embedding.
        """
        existing = self._stored(doc_id)
        if (
            existing is not None
            and existing.doc_hash == doc_hash
            and existing.pipeline_hash == pipeline_hash
            and self._count(doc_id) == existing.chunk_count
        ):
            return existing
        return None

    def _stored(self, doc_id: str) -> DocumentInfo | None:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[match("doc_id", doc_id), match("chunk_index", 0)]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return _document_info(points[0].payload) if points else None

    def _count(self, doc_id: str) -> int:
        selector = models.Filter(must=[match("doc_id", doc_id)])
        return self.client.count(self.collection, count_filter=selector, exact=True).count

    @traceable(
        name="qdrant_upsert",
        process_inputs=lambda i: {"chunks": len(i["chunks"]), "batch_size": i["batch_size"]},
        process_outputs=lambda _: {},
    )
    def _write_chunks(self, chunks: list[Document], batch_size: int) -> None:
        """Embed (dense + sparse) and upsert; the embedding run nests under this one."""
        self.vector_store.add_documents(chunks, ids=[c.id for c in chunks], batch_size=batch_size)

    def _is_complete(self, existing: DocumentInfo, document: IngestedDocument) -> bool:
        """Same file, made with the same pipeline settings, and every chunk of it is stored.

        The chunk count guards against an interrupted write; the pipeline hash makes a change
        to the chunking or privacy settings re-process documents.
        """
        return (
            existing.doc_hash == document.doc_hash
            and existing.pipeline_hash == document.pipeline_hash
            and existing.chunk_count == len(document.chunks)
            and self._count(document.doc_id) == len(document.chunks)
        )

    def _delete_where(
        self,
        must: list[models.Condition],
        must_not: list[models.Condition] | None = None,
    ) -> int:
        selector = models.Filter(must=must, must_not=must_not)
        count = self.client.count(self.collection, count_filter=selector, exact=True).count
        if count:
            self.client.delete(
                self.collection, points_selector=models.FilterSelector(filter=selector), wait=True
            )
        return count


def _to_document(
    point: models.ScoredPoint | models.Record,
    *,
    mode: SearchMode | None = None,
    rank: int | None = None,
) -> Document:
    payload = point.payload or {}
    metadata = dict(payload.get(METADATA_KEY, {}))
    if mode is not None:
        metadata |= {
            "retrieval_mode": mode,
            "retrieval_score": getattr(point, "score", None),
            "retrieval_rank": rank,
        }
    return Document(page_content=payload.get(CONTENT_KEY, ""), metadata=metadata, id=str(point.id))


def _document_info(payload: dict[str, Any] | None) -> DocumentInfo:
    metadata = (payload or {}).get(METADATA_KEY, {})
    pii = metadata.get("pii_scan", {})
    return DocumentInfo(
        doc_id=metadata["doc_id"],
        source=metadata["source"],
        title=metadata.get("title", metadata["source"]),
        file_type=metadata["file_type"],
        chunk_count=metadata["chunk_count"],
        doc_hash=metadata["doc_hash"],
        ingested_at=metadata.get("ingested_at", ""),
        pii_status=pii.get("status", ""),
        pii_policy=pii.get("policy", ""),
        pipeline_hash=metadata.get("pipeline_hash", ""),
        tenant_id=metadata.get("tenant_id", ""),
        classification=metadata.get("classification", ""),
        visibility=metadata.get("visibility", ""),
        allowed_groups=list(metadata.get("allowed_groups", [])),
        allowed_roles=list(metadata.get("allowed_roles", [])),
        allowed_users=list(metadata.get("allowed_users", [])),
        access_hash=metadata.get("access_hash", ""),
        access_version=metadata.get("access_version", 0),
        access_updated_at=metadata.get("access_updated_at", ""),
    )
