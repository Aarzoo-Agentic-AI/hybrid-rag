"""Ingestion tracing, checked with a mock LangSmith client (nothing is sent anywhere)."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context
from qdrant_client import QdrantClient

from hybrid_rag.config import ChunkingSettings, EmbeddingSettings
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline
from hybrid_rag.store import DocumentIndex
from tests.pdf_fixture import write_pdf

SECRET_SENTENCE = "Salary band C4 tops out at 187000 dollars."
CHUNKING = ChunkingSettings(target_tokens=60, max_tokens=120, overlap_tokens=12, tokenizer="approx")
PIPELINE = IngestionPipeline(CHUNKING)


class Runs:
    """Collects the runs a traced call would send to LangSmith."""

    def __init__(self, client: MagicMock) -> None:
        calls = client.method_calls
        self.created = {c.kwargs["id"]: c.kwargs for c in calls if c[0] == "create_run"}
        self.updated = {c.kwargs["run_id"]: c.kwargs for c in calls if c[0] == "update_run"}
        self.everything_sent = repr(client.method_calls)

    def named(self, name: str) -> list[dict]:
        return [r for r in self.created.values() if r["name"] == name]

    def parent_name(self, run: dict) -> str | None:
        parent = self.created.get(run.get("parent_run_id"))
        return parent["name"] if parent else None

    def outputs(self, run: dict) -> dict:
        return self.updated[run["id"]]["outputs"]


@pytest.fixture
def traced():
    client = MagicMock(spec=Client)
    client.tracing_queue = None

    def run(fn):
        with tracing_context(enabled=True, client=client):
            result = fn()
        return result, Runs(client)

    return run


@pytest.fixture
def index() -> DocumentIndex:
    embeddings = build_embeddings(EmbeddingSettings(provider="hashing"))
    return DocumentIndex(QdrantClient(":memory:"), "test", embeddings)


def write_policy(tmp_path: Path) -> Path:
    lines = [SECRET_SENTENCE] + [f"Rule {i}: claims need a receipt over 25 USD." for i in range(20)]
    return write_pdf(tmp_path / "salary_bands.pdf", [lines], title="Salary Bands")


def test_ingestion_is_one_trace_with_a_run_for_every_step(traced, index, tmp_path) -> None:
    write_policy(tmp_path)

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    (root,) = runs.named("ingest")
    assert root.get("parent_run_id") is None
    assert "ingestion" in root["tags"]
    assert runs.parent_name(runs.named("ingest_file")[0]) == "ingest"
    assert runs.parent_name(runs.named("load_document")[0]) == "ingest_file"
    assert runs.parent_name(runs.named("index_document")[0]) == "ingest_file"
    assert runs.parent_name(runs.named("qdrant_upsert")[0]) == "index_document"
    (embed,) = runs.named("embed")  # dense and sparse share one model pass
    assert embed["run_type"] == "embedding"
    assert runs.parent_name(embed) == "qdrant_upsert"
    assert {r["trace_id"] for r in runs.created.values()} == {root["trace_id"]}


def test_runs_record_counts_and_statuses(traced, index, tmp_path) -> None:
    write_policy(tmp_path)

    outcomes, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    chunks = outcomes[0].result.chunks
    assert runs.outputs(runs.named("ingest")[0]) == {"files": 1, "added": 1}
    assert runs.outputs(runs.named("ingest_file")[0])["status"] == "added"
    loaded = runs.outputs(runs.named("load_document")[0])
    assert (loaded["title"], loaded["pages"], loaded["chunks"]) == ("Salary Bands", 1, chunks)
    embed = runs.named("embed")[0]
    assert embed["inputs"]["texts"] == chunks
    assert runs.outputs(embed)["dense_dim"] == 384


def test_traces_never_contain_document_text_vectors_or_local_paths(traced, index, tmp_path) -> None:
    write_policy(tmp_path)

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    assert "salary_bands.pdf" in runs.everything_sent  # file names are fine
    assert "187000" not in runs.everything_sent
    assert "receipt" not in runs.everything_sent
    assert str(tmp_path) not in runs.everything_sent


def test_unchanged_file_is_traced_without_an_embedding_run(traced, index, tmp_path) -> None:
    write_policy(tmp_path)
    ingest_paths([tmp_path], index, PIPELINE)

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    assert runs.outputs(runs.named("ingest")[0]) == {"files": 1, "unchanged": 1}
    assert runs.named("embed") == []


def test_failed_file_is_visible_in_the_trace(traced, index, tmp_path) -> None:
    write_pdf(tmp_path / "scanned.pdf", [[]])

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    file_run = runs.outputs(runs.named("ingest_file")[0])
    assert file_run["status"] == "failed"
    assert "no extractable text" in file_run["error"]
    assert runs.outputs(runs.named("ingest")[0]) == {"files": 1, "failed": 1}


def test_generator_of_paths_is_not_consumed_by_tracing(traced, index, tmp_path) -> None:
    write_policy(tmp_path)

    outcomes, runs = traced(lambda: ingest_paths((p for p in [tmp_path]), index, PIPELINE))

    assert len(outcomes) == 1
    assert runs.named("ingest")[0]["inputs"]["paths"] == "(iterable)"


def test_every_trace_processor_succeeds(traced, index, tmp_path, caplog) -> None:
    """LangSmith drops a run's inputs/outputs if a processor raises; make sure none do."""
    write_policy(tmp_path)

    traced(lambda: ingest_paths([tmp_path], index, PIPELINE, force=True))

    assert not [r for r in caplog.records if "process_" in r.getMessage()]


def test_sanitize_step_is_traced_with_counts_only(traced, index, tmp_path) -> None:
    write_pdf(tmp_path / "faq.pdf", [["Questions go to jane.doe@gmail.com or 415-555-0134."]])

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    (sanitize,) = runs.named("sanitize")
    assert runs.parent_name(sanitize) == "load_document"
    assert sanitize["inputs"] == {"source": "faq.pdf", "pages": 1}
    scan = runs.outputs(sanitize)
    assert scan["status"] == "redacted"
    assert scan["redactions"] == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1}
    assert "jane.doe" not in runs.everything_sent
    assert "555-0134" not in runs.everything_sent


def test_quarantined_file_trace_reveals_no_values(traced, index, tmp_path) -> None:
    write_pdf(tmp_path / "export.pdf", [["Direct deposit: SSN 123-45-6789."]])

    _, runs = traced(lambda: ingest_paths([tmp_path], index, PIPELINE))

    file_run = runs.outputs(runs.named("ingest_file")[0])
    assert file_run["status"] == "quarantined"
    assert "US_SSN x1" in file_run["error"]
    assert runs.outputs(runs.named("ingest")[0]) == {"files": 1, "quarantined": 1}
    assert runs.named("index_document") == []  # never reaches the store
    assert "6789" not in runs.everything_sent


def test_retrieval_traces_scores_but_no_chunk_text(traced, index, tmp_path: Path) -> None:
    from hybrid_rag.access import Principal
    from hybrid_rag.config import RetrievalSettings, Settings
    from hybrid_rag.ingestion import PolicyRegistry
    from hybrid_rag.retrieval import build_retriever
    from tests.test_retrieval import KeywordCrossEncoder

    registry = PolicyRegistry(
        {"pay.pdf": {"policy_name": "Pay", "access": {"visibility": "tenant"}}}
    )
    pipeline = IngestionPipeline(CHUNKING, registry=registry)
    ingest_paths([write_pdf(tmp_path / "pay.pdf", [[SECRET_SENTENCE]])], index, pipeline)
    settings = Settings.model_construct(
        retrieval=RetrievalSettings(fused_limit=5, top_k=2), chunking=CHUNKING
    )
    caller = Principal("default", "u-1", "restricted")
    retriever = build_retriever(
        settings, index, caller, mode="hybrid_rerank", cross_encoder=KeywordCrossEncoder()
    )

    hits, runs = traced(lambda: retriever.invoke("salary band"))

    assert SECRET_SENTENCE in hits[0].page_content
    (rerank_run,) = runs.named("rerank")
    kept = runs.outputs(rerank_run)["kept"]
    assert kept[0]["rerank_rank"] == 1 and "rerank_score" in kept[0]
    assert SECRET_SENTENCE not in repr(runs.outputs(rerank_run))
    (expand_run,) = runs.named("expand_sections")
    assert runs.outputs(expand_run) == {"results": 1, "expanded": 0}
