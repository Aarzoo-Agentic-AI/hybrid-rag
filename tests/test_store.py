from pathlib import Path

import pytest
from qdrant_client import QdrantClient, models

from hybrid_rag.config import ChunkingSettings, EmbeddingSettings
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import load_document
from hybrid_rag.store import (
    DENSE_VECTOR,
    SPARSE_VECTOR,
    CollectionConfigError,
    DocumentIndex,
    ensure_collection,
)
from tests.pdf_fixture import write_pdf

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"
CHUNKING = ChunkingSettings(chunk_size=200, chunk_overlap=40)


@pytest.fixture
def client() -> QdrantClient:
    return QdrantClient(":memory:")


@pytest.fixture
def index(client: QdrantClient) -> DocumentIndex:
    return DocumentIndex(client, "test", build_embeddings(EmbeddingSettings(provider="hashing")))


def write(tmp_path: Path, name: str, text: str) -> Path:
    """Write ``text`` as a PDF, 40 lines per page."""
    lines = text.split("\n")
    return write_pdf(tmp_path / name, [lines[i : i + 40] for i in range(0, len(lines), 40)])


def long_text(n: int) -> str:
    return "\n".join(f"Paragraph {i}: the LR-50 courier robot delivers parcels." for i in range(n))


# ------------------------------------------------------------ collection


def test_collection_has_named_dense_and_sparse_vectors(index: DocumentIndex) -> None:
    params = index.client.get_collection("test").config.params

    assert params.vectors[DENSE_VECTOR].size == 384
    assert params.vectors[DENSE_VECTOR].distance == models.Distance.COSINE
    assert SPARSE_VECTOR in params.sparse_vectors


def test_ensure_collection_is_idempotent(client: QdrantClient) -> None:
    assert ensure_collection(client, "c", 8) is True
    assert ensure_collection(client, "c", 8) is False


def test_wrong_vector_size_is_reported(client: QdrantClient) -> None:
    ensure_collection(client, "c", 8)

    with pytest.raises(CollectionConfigError, match="8-dimensional"):
        ensure_collection(client, "c", 16)


def test_collection_without_named_vectors_is_reported(client: QdrantClient) -> None:
    client.create_collection(
        "plain", vectors_config=models.VectorParams(size=8, distance=models.Distance.COSINE)
    )

    with pytest.raises(CollectionConfigError, match="no 'dense' and 'sparse'"):
        ensure_collection(client, "plain", 8)


# ---------------------------------------------------------------- index


def test_new_document_is_added(index: DocumentIndex, tmp_path: Path) -> None:
    doc = load_document(write(tmp_path, "a.pdf", long_text(20)), CHUNKING)

    result = index.index(doc)

    assert (result.status, result.chunks, result.removed) == ("added", len(doc.chunks), 0)
    assert index.count_chunks() == len(doc.chunks)
    point = index.client.retrieve("test", [doc.chunks[0].id], with_payload=True)[0]
    assert point.payload["page_content"] == doc.chunks[0].page_content
    assert point.payload["metadata"]["source"] == "a.pdf"
    assert point.payload["metadata"]["ingested_at"]


def test_unchanged_document_is_skipped_without_embedding(
    index: DocumentIndex, tmp_path: Path
) -> None:
    doc = load_document(write(tmp_path, "a.pdf", long_text(20)), CHUNKING)
    index.index(doc)
    embedded = index.embeddings.encoder.encoded_count

    result = index.index(load_document(tmp_path / "a.pdf", CHUNKING))

    assert result.status == "unchanged"
    assert index.embeddings.encoder.encoded_count == embedded


def test_changed_document_replaces_old_chunks(index: DocumentIndex, tmp_path: Path) -> None:
    path = write(tmp_path, "a.pdf", long_text(30))
    first = load_document(path, CHUNKING)
    index.index(first)

    write(tmp_path, "a.pdf", long_text(5))  # shorter: fewer chunks
    second = load_document(path, CHUNKING)
    result = index.index(second)

    assert result.status == "updated"
    assert result.removed == len(first.chunks) - len(second.chunks)
    assert index.count_chunks() == len(second.chunks)
    hashes = {
        p.payload["metadata"]["doc_hash"]
        for p in index.client.scroll("test", limit=100, with_payload=True)[0]
    }
    assert hashes == {second.doc_hash}


def test_force_reindexes_unchanged_document(index: DocumentIndex, tmp_path: Path) -> None:
    doc = load_document(write(tmp_path, "a.pdf", "Hello."), CHUNKING)
    index.index(doc)

    assert index.index(doc, force=True).status == "updated"


def test_interrupted_write_is_repaired_on_next_run(index: DocumentIndex, tmp_path: Path) -> None:
    doc = load_document(write(tmp_path, "a.pdf", long_text(30)), CHUNKING)
    index.index(doc)
    index.client.delete("test", points_selector=models.PointIdsList(points=[doc.chunks[-1].id]))

    result = index.index(doc)

    assert result.status == "updated"
    assert index.count_chunks(doc.doc_id) == len(doc.chunks)


def test_delete_removes_only_that_document(index: DocumentIndex, tmp_path: Path) -> None:
    a = load_document(write(tmp_path, "a.pdf", long_text(10)), CHUNKING)
    b = load_document(write(tmp_path, "b.pdf", "Other document."), CHUNKING)
    index.index(a)
    index.index(b)

    assert index.delete(a.doc_id) == len(a.chunks)
    assert index.delete(a.doc_id) == 0
    assert [d.source for d in index.list_documents()] == ["b.pdf"]


def test_list_and_get_documents(index: DocumentIndex, tmp_path: Path) -> None:
    index.index(load_document(write(tmp_path, "b.pdf", long_text(10)), CHUNKING))
    a = load_document(write(tmp_path, "a.pdf", "Short."), CHUNKING)
    index.index(a)

    documents = index.list_documents()

    assert [d.source for d in documents] == ["a.pdf", "b.pdf"]
    info = index.get_document(a.doc_id)
    assert info is not None
    assert (info.file_type, info.chunk_count, info.doc_hash) == ("pdf", 1, a.doc_hash)
    assert index.get_document("missing") is None


def test_reset_empties_the_collection(index: DocumentIndex, tmp_path: Path) -> None:
    index.index(load_document(write(tmp_path, "a.pdf", "Hello."), CHUNKING))

    index.reset()

    assert index.count_chunks() == 0
    assert index.list_documents() == []


def test_health_reports_point_count(index: DocumentIndex, tmp_path: Path) -> None:
    index.index(load_document(write(tmp_path, "a.pdf", "Hello."), CHUNKING))

    health = index.health()

    assert health["collection"] == "test"
    assert health["points"] == 1


# ------------------------------------------------- end to end on samples


def test_ingest_samples_then_rerun_is_a_no_op(index: DocumentIndex) -> None:
    first = ingest_paths([SAMPLES], index, ChunkingSettings())
    embedded = index.embeddings.encoder.encoded_count
    second = ingest_paths([SAMPLES], index, ChunkingSettings())

    assert all(o.ok for o in first)
    assert {o.result.status for o in first} == {"added"}
    assert {o.result.status for o in second} == {"unchanged"}
    assert index.embeddings.encoder.encoded_count == embedded
    assert len(index.list_documents()) == len(first)
    assert index.count_chunks() == sum(o.result.chunks for o in first)


def test_ingest_reports_bad_files_without_stopping(index: DocumentIndex, tmp_path: Path) -> None:
    write(tmp_path, "good.pdf", "Fine.")
    write(tmp_path, "empty.pdf", "   ")

    outcomes = {o.path.name: o for o in ingest_paths([tmp_path], index, CHUNKING)}

    assert outcomes["good.pdf"].ok
    assert not outcomes["empty.pdf"].ok
    assert "no extractable text" in outcomes["empty.pdf"].error


def test_hybrid_search_finds_exact_term(index: DocumentIndex) -> None:
    ingest_paths([SAMPLES], index, ChunkingSettings())

    hits = index.vector_store.similarity_search("Can I enter the LR-200 safety zone?", k=3)

    assert "LR-200" in hits[0].page_content
    assert hits[0].metadata["page"] == 3


def test_unchanged_file_is_skipped_before_loading(
    index: DocumentIndex, tmp_path: Path, monkeypatch
) -> None:
    write(tmp_path, "a.pdf", long_text(10))
    ingest_paths([tmp_path], index, CHUNKING)

    def fail(*args, **kwargs):
        raise AssertionError("an unchanged file must not be loaded, scanned or chunked")

    monkeypatch.setattr("hybrid_rag.indexing.load_document", fail)
    (outcome,) = ingest_paths([tmp_path], index, CHUNKING)

    assert outcome.result.status == "unchanged"


def test_changed_chunking_settings_reindex_unchanged_files(
    index: DocumentIndex, tmp_path: Path
) -> None:
    write(tmp_path, "a.pdf", long_text(10))
    ingest_paths([tmp_path], index, CHUNKING)

    bigger = ChunkingSettings(chunk_size=300, chunk_overlap=40)
    (outcome,) = ingest_paths([tmp_path], index, bigger)

    assert outcome.result.status == "updated"
    assert index.count_chunks() == outcome.result.chunks
