"""The Qdrant collection and ``DocumentIndex``: add, update, skip, delete and list documents.

One collection holds every chunk as a point with:
- a named dense vector ``dense`` (cosine) and a named sparse vector ``sparse``,
- the payload LangChain's ``QdrantVectorStore`` uses: ``page_content`` plus ``metadata``
  (the chunk metadata from ingestion, and ``ingested_at``).

Writes go through ``QdrantVectorStore.add_documents`` so the LangChain store, which the
retrievers use in step 4, sees exactly the same data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models

from hybrid_rag.embedding import EmbeddingModels
from hybrid_rag.ingestion import IngestedDocument
from hybrid_rag.tracing import traceable

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"

# Fields we filter on get a payload index, so filtered searches and deletes stay fast.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    f"{METADATA_KEY}.doc_id": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.doc_hash": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.source": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.file_type": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.chunk_index": models.PayloadSchemaType.INTEGER,
    f"{METADATA_KEY}.pii_scan.status": models.PayloadSchemaType.KEYWORD,
    # Policy and section metadata, for the relevance filters and small-to-big retrieval.
    f"{METADATA_KEY}.policy_id": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.policy_category": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.jurisdiction": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.applies_to": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.is_current": models.PayloadSchemaType.BOOL,
    f"{METADATA_KEY}.section_id": models.PayloadSchemaType.KEYWORD,
    f"{METADATA_KEY}.chunk_type": models.PayloadSchemaType.KEYWORD,
}


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


@dataclass(frozen=True)
class IndexResult:
    doc_id: str
    source: str
    status: Literal["added", "updated", "unchanged"]
    chunks: int
    removed: int = 0  # stale chunks deleted from the previous version


def _field(key: str) -> str:
    return f"{METADATA_KEY}.{key}"


def _match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=_field(key), match=models.MatchValue(value=value))


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
            {"status": r.status, "chunks": r.chunks, "removed": r.removed} if r else {}
        ),
    )
    def index(self, document: IngestedDocument, *, force: bool = False) -> IndexResult:
        """Add a document, replace an older version of it, or skip it if it hasn't changed.

        New chunks are written first (their IDs are deterministic, so they overwrite the old
        chunks in place), and only then are leftover chunks of the old version deleted. A
        failure part-way therefore never leaves the document missing from the index.
        """
        existing = self.get_document(document.doc_id)
        if existing is not None and not force and self._is_complete(existing, document):
            logger.info("%s unchanged, skipped", document.source)
            return IndexResult(document.doc_id, document.source, "unchanged", existing.chunk_count)

        ingested_at = datetime.now(UTC).isoformat(timespec="seconds")
        chunks = [
            Document(
                page_content=c.page_content,
                metadata={**c.metadata, "ingested_at": ingested_at},
                id=c.id,
            )
            for c in document.chunks
        ]
        self._write_chunks(chunks, self.batch_size)
        # Leftovers are this document's chunks that aren't part of the new version: from an
        # older file, or from the same file chunked differently (new chunking/privacy rules).
        removed = self._delete_where(
            must=[_match("doc_id", document.doc_id)],
            must_not=[models.HasIdCondition(has_id=[c.id for c in chunks])],
        )
        status: Literal["added", "updated"] = "added" if existing is None else "updated"
        logger.info(
            "%s %s: %d chunk(s), %d stale removed", document.source, status, len(chunks), removed
        )
        return IndexResult(document.doc_id, document.source, status, len(chunks), removed)

    def delete(self, doc_id: str) -> int:
        """Delete every chunk of a document. Returns the number of chunks deleted."""
        return self._delete_where(must=[_match("doc_id", doc_id)])

    def reset(self) -> None:
        """Delete the whole collection and create it again, empty."""
        self.client.delete_collection(self.collection)
        ensure_collection(self.client, self.collection, self.embeddings.dense_dim)

    # ------------------------------------------------------------- reads

    def current_version(
        self, doc_id: str, doc_hash: str, pipeline_hash: str
    ) -> DocumentInfo | None:
        """The stored document if it is this exact file, made with these settings, and complete.

        Lets ingestion skip an unchanged file before loading, scanning or chunking it.
        """
        existing = self.get_document(doc_id)
        if (
            existing is not None
            and existing.doc_hash == doc_hash
            and existing.pipeline_hash == pipeline_hash
            and self.count_chunks(doc_id) == existing.chunk_count
        ):
            return existing
        return None

    def get_document(self, doc_id: str) -> DocumentInfo | None:
        points, _ = self.client.scroll(
            self.collection,
            scroll_filter=models.Filter(must=[_match("doc_id", doc_id), _match("chunk_index", 0)]),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return _document_info(points[0].payload) if points else None

    def list_documents(self) -> list[DocumentInfo]:
        """One entry per document (read from each document's first chunk), sorted by source."""
        documents: list[DocumentInfo] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=models.Filter(must=[_match("chunk_index", 0)]),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            documents.extend(_document_info(p.payload) for p in points)
            if offset is None:
                return sorted(documents, key=lambda d: d.source)

    def count_chunks(self, doc_id: str | None = None) -> int:
        count_filter = models.Filter(must=[_match("doc_id", doc_id)]) if doc_id else None
        return self.client.count(self.collection, count_filter=count_filter, exact=True).count

    def health(self) -> dict[str, Any]:
        """Collection status for ``/health``. Raises if Qdrant can't be reached."""
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "status": str(info.status.value if hasattr(info.status, "value") else info.status),
            "points": info.points_count,
        }

    # ----------------------------------------------------------- helpers

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
            and self.count_chunks(document.doc_id) == len(document.chunks)
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


def _document_info(payload: dict[str, Any] | None) -> DocumentInfo:
    metadata = (payload or {}).get(METADATA_KEY, {})
    return DocumentInfo(
        doc_id=metadata["doc_id"],
        source=metadata["source"],
        title=metadata.get("title", metadata["source"]),
        file_type=metadata["file_type"],
        chunk_count=metadata["chunk_count"],
        doc_hash=metadata["doc_hash"],
        ingested_at=metadata.get("ingested_at", ""),
        pii_status=metadata.get("pii_scan", {}).get("status", ""),
        pii_policy=metadata.get("pii_scan", {}).get("policy", ""),
        pipeline_hash=metadata.get("pipeline_hash", ""),
    )
