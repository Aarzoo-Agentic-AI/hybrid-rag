"""Retrieval: the four modes, relevance filters, reranking, small-to-big and leakage.

No models: the ``hashing`` encoder stands in for bge-m3 and ``KeywordCrossEncoder`` for the
reranker, over an in-memory Qdrant.
"""

from pathlib import Path

import pytest
from langchain_core.cross_encoders import BaseCrossEncoder
from qdrant_client import QdrantClient

from hybrid_rag.access import Principal
from hybrid_rag.config import (
    ChunkingSettings,
    EmbeddingSettings,
    RerankerSettings,
    RetrievalSettings,
    Settings,
)
from hybrid_rag.embedding import build_embeddings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline, PolicyRegistry, make_doc_id
from hybrid_rag.ingestion.tokens import ApproxTokenCounter
from hybrid_rag.retrieval import (
    MODES,
    RelevanceFilters,
    build_retriever,
    expand_sections,
    rerank,
)
from hybrid_rag.retrieval.expansion import merge_section
from hybrid_rag.store import DocumentIndex
from tests.pdf_fixture import write_pdf
from tests.test_access import ACME_ADMIN, DOCUMENTS, EMPLOYEE, EXPECTED, IDS, index_world

TOKENS = ApproxTokenCounter()


class KeywordCrossEncoder(BaseCrossEncoder):
    """Scores a pair by how many query words appear in the text; counts its calls."""

    def __init__(self) -> None:
        self.pairs = 0

    def score(self, text_pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs += len(text_pairs)
        return [
            float(sum(w.lower() in text.lower() for w in query.split()))
            for query, text in text_pairs
        ]


def settings(**retrieval) -> Settings:
    return Settings.model_construct(
        retrieval=RetrievalSettings(**{"fused_limit": 10, "top_k": 4, **retrieval}),
        reranker=RerankerSettings(),
        chunking=ChunkingSettings(tokenizer="approx"),
    )


def new_index() -> DocumentIndex:
    return DocumentIndex(
        QdrantClient(":memory:"), "r", build_embeddings(EmbeddingSettings(provider="hashing"))
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> DocumentIndex:
    index = new_index()
    index_world(index, tmp_path_factory.mktemp("retrieval"))
    return index


def retrieve(index, principal, query, mode, **kwargs):
    retriever = build_retriever(
        settings(), index, principal, mode=mode, cross_encoder=KeywordCrossEncoder(), **kwargs
    )
    return retriever.invoke(query)


# --------------------------------------------------------------- leakage


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("principal", "visible"), EXPECTED, ids=IDS)
def test_no_mode_returns_unreadable_chunks(world, mode, principal, visible) -> None:
    for term in DOCUMENTS.values():
        hits = retrieve(world, principal, f"code word {term}", mode)

        assert all(principal.can_read(h.metadata) for h in hits), (mode, term)
        assert {h.metadata["source"] for h in hits} <= visible
        for h in hits:
            if h.metadata["source"] not in visible:
                pytest.fail(f"{mode} leaked {h.metadata['source']}")


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_finds_a_readable_document_by_its_unique_term(world, mode) -> None:
    hits = retrieve(world, ACME_ADMIN, "code word QXSALARY", mode)

    assert hits[0].metadata["source"] == "salary_bands.pdf", mode
    assert hits[0].metadata["retrieval_mode"] == ("hybrid" if mode == "hybrid_rerank" else mode)


def test_relevance_filters_never_widen_access(world) -> None:
    only_salary = RelevanceFilters(doc_ids=[make_doc_id("acme", "salary_bands.pdf")])

    for mode in MODES:
        assert retrieve(world, EMPLOYEE, "QXSALARY", mode, filters=only_salary) == []


# ----------------------------------------------------------------- modes


def test_results_carry_scores_and_ranks(world) -> None:
    hits = retrieve(world, ACME_ADMIN, "code word QXMANAGER rule", "hybrid")

    assert [h.metadata["retrieval_rank"] for h in hits] == list(range(1, len(hits) + 1))
    scores = [h.metadata["retrieval_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)
    assert len(hits) == 4  # top_k


def test_hybrid_rerank_scores_fused_limit_candidates_and_keeps_top_k(world) -> None:
    encoder = KeywordCrossEncoder()
    retriever = build_retriever(
        settings(), world, ACME_ADMIN, mode="hybrid_rerank", cross_encoder=encoder
    )

    hits = retriever.invoke("QXPERSONAL code word")

    # fused_limit (10) candidates, or every chunk the caller may read if there are fewer
    assert encoder.pairs == min(10, world.count_chunks(ACME_ADMIN))
    assert len(hits) == 4  # top_k kept
    assert [h.metadata["rerank_rank"] for h in hits] == [1, 2, 3, 4]
    assert hits[0].metadata["source"] == "personal_file.pdf"
    scores = [h.metadata["rerank_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_rerank_reorders_by_cross_encoder_score(world) -> None:
    candidates = world.search("code word rule", ACME_ADMIN, k=10)
    target = next(c for c in candidates if "QXINVESTIGATE" in c.page_content)

    reranked = rerank(KeywordCrossEncoder(), "QXINVESTIGATE policy", candidates, top_n=3)

    assert reranked[0].id == target.id
    assert reranked[0].metadata["retrieval_rank"] == target.metadata["retrieval_rank"]


def test_hybrid_rerank_needs_an_enabled_reranker(world) -> None:
    s = settings()
    s.reranker = RerankerSettings(enabled=False)

    with pytest.raises(ValueError, match=r"reranker\.enabled"):
        build_retriever(s, world, ACME_ADMIN, mode="hybrid_rerank")
    assert build_retriever(s, world, ACME_ADMIN, mode="hybrid").invoke("QXHANDBOOK")


# --------------------------------------------------------------- filters


def test_superseded_versions_are_left_out_unless_asked_for(tmp_path: Path) -> None:
    index = new_index()
    registry = PolicyRegistry(
        {
            "travel_2026.pdf": {
                "policy_name": "Travel 2026",
                "is_current": False,
                "jurisdiction": "UK",
            },
            "travel_2027.pdf": {
                "policy_name": "Travel 2027",
                "policy_category": "travel",
                "jurisdiction": "US",
            },
        },
        defaults={"access": {"classification": "internal", "visibility": "tenant"}},
    )
    for name in ("travel_2026.pdf", "travel_2027.pdf"):
        write_pdf(tmp_path / name, [[f"{name}: the hotel cap is QXHOTEL per night."]])
    pipeline = IngestionPipeline(
        ChunkingSettings(tokenizer="approx"), registry=registry, tenant_id="acme"
    )
    ingest_paths([tmp_path], index, pipeline)

    def sources(filters):
        return {
            h.metadata["source"]
            for h in retrieve(index, EMPLOYEE, "QXHOTEL", "hybrid", filters=filters)
        }

    assert sources(None) == {"travel_2027.pdf"}
    assert sources(RelevanceFilters(current_only=False)) == {"travel_2026.pdf", "travel_2027.pdf"}
    assert sources(RelevanceFilters(policy_categories=["expenses"])) == set()
    assert sources(RelevanceFilters(current_only=False, jurisdictions=["UK"])) == {
        "travel_2026.pdf"
    }


def test_empty_filter_lists_are_refused() -> None:
    with pytest.raises(ValueError, match="null"):
        RelevanceFilters(jurisdictions=[])


# ------------------------------------------------------------ small-to-big


@pytest.fixture(scope="module")
def split_section(tmp_path_factory: pytest.TempPathFactory) -> tuple[DocumentIndex, list]:
    """One long numbered section, split into several overlapping parts."""
    folder = tmp_path_factory.mktemp("split")
    lines = ["1. Leave rules"] + [
        f"Rule {i}: employees on leave keep benefit number {i} unless noted." for i in range(30)
    ]
    lines.insert(25, "QXDEEP: this rule appears only in a later part of the section.")
    write_pdf(folder / "leave.pdf", [lines])
    index = new_index()
    registry = PolicyRegistry(
        {
            "leave.pdf": {
                "policy_name": "Leave",
                "access": {"classification": "internal", "visibility": "tenant"},
            }
        }
    )
    chunking = ChunkingSettings(
        target_tokens=60, max_tokens=120, overlap_tokens=20, tokenizer="approx"
    )
    pipeline = IngestionPipeline(chunking, registry=registry, tenant_id="acme")
    ingest_paths([folder], index, pipeline)
    parts = index.section_chunks(
        index.search("QXDEEP", ACME_ADMIN, k=1)[0].metadata["section_id"], ACME_ADMIN
    )
    assert len(parts) >= 3
    return index, parts


def test_a_hit_on_one_part_returns_the_whole_section(split_section) -> None:
    index, parts = split_section
    hit = index.search("QXDEEP", EMPLOYEE, k=1)

    (expanded,) = expand_sections(hit, index, EMPLOYEE, tokens=TOKENS, max_tokens=5000)

    assert expanded.metadata["section_expanded"] is True
    assert expanded.metadata["section_chunk_ids"] == [p.metadata["chunk_id"] for p in parts]
    for i in range(30):
        assert expanded.page_content.count(f"Rule {i}:") == 1, i  # every rule, overlap removed
    assert expanded.page_content.count("Policy: Leave") == 1  # one header
    assert expanded.metadata["retrieval_rank"] == hit[0].metadata["retrieval_rank"]


def test_two_hits_in_one_section_become_one_result(split_section) -> None:
    index, parts = split_section

    out = expand_sections(parts[::-1], index, EMPLOYEE, tokens=TOKENS, max_tokens=5000)

    assert len(out) == 1
    assert out[0].id == parts[-1].id  # at the better-ranked hit's position, with its scores


def test_sections_over_the_token_cap_are_not_expanded(split_section) -> None:
    index, parts = split_section

    out = expand_sections(parts[:1], index, EMPLOYEE, tokens=TOKENS, max_tokens=50)

    assert out == parts[:1]


def test_unreadable_sections_are_never_expanded(split_section) -> None:
    index, parts = split_section
    outsider = Principal("globex", "u-1", "restricted")

    assert index.section_chunks(parts[0].metadata["section_id"], outsider) == []


def test_merge_keeps_a_section_without_overlap_intact(split_section) -> None:
    _, parts = split_section

    merged = merge_section(parts[:1], parts[0])

    assert merged.page_content == parts[0].page_content


def test_definitions_and_whole_sections_are_left_alone(world) -> None:
    hits = world.search("QXHANDBOOK", ACME_ADMIN, k=3)

    assert expand_sections(hits, world, ACME_ADMIN, tokens=TOKENS, max_tokens=5000) == hits


def test_build_retriever_expands_by_default(split_section) -> None:
    index, _ = split_section
    s = settings(top_k=2)

    (first, *_) = build_retriever(s, index, EMPLOYEE, mode="sparse").invoke("QXDEEP")
    (plain, *_) = build_retriever(s, index, EMPLOYEE, mode="sparse", expand=False).invoke("QXDEEP")

    assert first.metadata.get("section_expanded") is True
    assert "section_expanded" not in plain.metadata
