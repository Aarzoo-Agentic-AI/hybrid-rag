"""Real bge-m3 and real Qdrant. Run with: uv run pytest -m integration

The first run downloads the bge-m3 model (about 2.3 GB).
"""

import math
import uuid
from pathlib import Path

import pytest

from hybrid_rag.access import Principal
from hybrid_rag.config import ChunkingSettings, EmbeddingSettings, get_settings
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline
from hybrid_rag.store import DocumentIndex, create_client

ADMIN = Principal.tenant_admin("default")  # sees every document of the test tenant

pytestmark = pytest.mark.integration

SAMPLES = Path(__file__).resolve().parents[1] / "data" / "samples"


@pytest.fixture(scope="module")
def embeddings():
    return build_embeddings(EmbeddingSettings(provider="bge-m3"))


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    )


def test_bge_m3_produces_dense_and_sparse_vectors(embeddings) -> None:
    (encoding,) = embeddings.encoder.encode(["Employees receive 25 days of annual leave"])

    assert len(encoding.dense) == 1024
    assert math.isclose(math.sqrt(sum(x * x for x in encoding.dense)), 1.0, rel_tol=1e-3)
    assert 0 < len(encoding.sparse) < 50
    assert all(w > 0 for w in encoding.sparse.values())


def test_bge_m3_dense_vectors_capture_meaning(embeddings) -> None:
    leave, vacation, robot = embeddings.dense.embed_documents(
        ["How many days of annual leave do I get?", "vacation days per year", "robot arm payload"]
    )

    assert cosine(leave, vacation) > cosine(leave, robot) + 0.1


def test_samples_are_indexed_in_real_qdrant(embeddings) -> None:
    client = create_client(get_settings())
    collection = f"test_{uuid.uuid4().hex[:8]}"
    try:
        index = DocumentIndex(client, collection, embeddings)
        outcomes = ingest_paths(
            [SAMPLES], index, IngestionPipeline(ChunkingSettings(tokenizer="approx"))
        )

        assert all(o.ok for o in outcomes)
        assert len(index.list_documents(ADMIN)) == len(outcomes)
        hits = index.search("How long is parental leave?", ADMIN, k=3)
        assert any("Parental leave" in h.page_content for h in hits)
    finally:
        client.delete_collection(collection)


def test_access_rules_hold_in_real_qdrant(tmp_path: Path) -> None:
    """The leakage matrix from test_access.py, against the Qdrant server (with its payload
    indexes, including the is_tenant index), using the fast hashing encoder."""
    from tests.test_access import DOCUMENTS, EXPECTED, index_world

    client = create_client(get_settings())
    collection = f"test_{uuid.uuid4().hex[:8]}"
    try:
        index = DocumentIndex(
            client, collection, build_embeddings(EmbeddingSettings(provider="hashing"))
        )
        index_world(index, tmp_path)
        schema = client.get_collection(collection).payload_schema
        assert schema["metadata.tenant_id"].params.is_tenant is True

        for principal, visible in EXPECTED:
            assert {d.source for d in index.list_documents(principal)} == visible
            for source, term in DOCUMENTS.items():
                hits = index.search(term, principal, k=50)
                assert {h.metadata["source"] for h in hits} <= visible, (principal, term)
                assert all(principal.can_read(h.metadata) for h in hits)
                if principal.tenant_id == "acme" and source in visible:
                    assert hits[0].metadata["source"] == source
    finally:
        client.delete_collection(collection)
