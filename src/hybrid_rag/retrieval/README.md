# Retrieval: hybrid search, reranking and small-to-big

This package finds the chunks that answer a question, for one caller. Every stage runs with the caller's security filter, so nothing they can't read is ever retrieved, reranked or expanded.

```python
from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.retrieval import RelevanceFilters, build_retriever
from hybrid_rag.store import build_document_index

settings = get_settings()
index = build_document_index(settings)
employee = Principal("dummymocktest", "u-1", clearance="internal")

retriever = build_retriever(settings, index, employee)  # retrieval.mode, default hybrid_rerank
docs = retriever.invoke("How many PTO days do I get after 4 years?")

build_retriever(settings, index, employee, mode="sparse")  # dense | sparse | hybrid | hybrid_rerank
build_retriever(settings, index, employee, filters=RelevanceFilters(policy_categories=["travel"]))
```

From the command line:

```bash
uv run python scripts/search.py "How many PTO days do I get after 4 years?"
uv run python scripts/search.py "merit matrix" --as manager          # employee | manager | hr-comp | admin
uv run python scripts/search.py "COBRA" --compare --text             # all four modes, with the text
```

## The pipeline

```
question ─► PolicyRetriever ──────────────────► ScoredReranker ──────► SectionExpandingRetriever
            dense (50) ┐                         cross-encoder          a hit on part of a split
                       ├─ RRF in Qdrant ─► 20    reads question and     section becomes the whole
            sparse (50)┘   (security filter      chunk together,        section (security filter)
                            on each prefetch)    keeps the best 6
```

| `retrieval.mode` | What runs | Use |
|---|---|---|
| `dense` | bge-m3 dense vectors only, `top_k` results | Meaning, paraphrases ("time off" for "PTO") |
| `sparse` | bge-m3 lexical weights only | Exact terms: `HR-TRV-003`, `COBRA`, `401(k)` |
| `hybrid` | both, fused with Reciprocal Rank Fusion | Both kinds of question, no model in the loop |
| `hybrid_rerank` (default) | hybrid, `fused_limit` candidates, then the cross-encoder keeps `top_k` | Best quality; the slowest step |

The four modes are there so the evaluation (step 7) can show what each stage adds.

## Files

| File | Contents |
|---|---|
| [`retriever.py`](retriever.py) | `PolicyRetriever`, a LangChain `BaseRetriever` over `DocumentIndex.search()` |
| [`filters.py`](filters.py) | `RelevanceFilters`: current versions only (default), categories, jurisdictions, audience, documents |
| [`reranker.py`](reranker.py) | `LocalCrossEncoder` (a LangChain `BaseCrossEncoder`), `ScoredReranker` and `rerank()` |
| [`expansion.py`](expansion.py) | `expand_sections()`, `merge_section()` and `SectionExpandingRetriever` |
| [`__init__.py`](__init__.py) | `build_retriever()` and `MODES` |

## Why our own retriever

LangChain's `QdrantVectorStore` in hybrid mode fetches `k` candidates from each search and returns `k`. We want more candidates than results: 50 dense + 50 sparse, fused down to 20 for the reranker. So `PolicyRetriever` calls `DocumentIndex.search()`, which builds the Qdrant query itself:

- **Prefetches:** one dense and one sparse, each with its own limit (`dense_limit`, `sparse_limit`).
- **Fusion:** done by Qdrant (`FusionQuery(RRF)`), server-side.
- **Security filter:** applied to each prefetch and to the final query.
- **Scores:** each result gets `retrieval_mode`, `retrieval_score` and `retrieval_rank` in its metadata.

The reranking itself is LangChain's own: `ContextualCompressionRetriever` with our `ScoredReranker`, which is `CrossEncoderReranker` plus the scores (`rerank_score`, `rerank_rank`).

## Relevance filters

`RelevanceFilters` say which policies *apply* to the question; the security filter says which the caller may *read*. Relevance conditions are ANDed with the security filter, so whatever values a request sends, they can only narrow the results.

| Field | Default | Example |
|---|---|---|
| `current_only` | `true` | Leaves out superseded versions (`is_current: false` in the registry) |
| `policy_categories` | all | `["travel", "expenses"]` |
| `jurisdictions` | all | `["US"]` |
| `applies_to` | all | `["employees"]` |
| `doc_ids` | all | Specific documents |

An empty list is refused (`null` means no filter), because an empty list would silently match nothing.

## Reranking

- **Model:** any `sentence-transformers` cross-encoder (`reranker.model`), run locally and loaded on first use. One instance per process (`get_reranker()`).

  Measured on the development laptop (i7-1265U, no GPU), reranking 20 candidates of about 450 tokens:

  | Model | Size | Time per question | Notes |
  |---|---|---|---|
  | `cross-encoder/ms-marco-MiniLM-L-6-v2` (**default**) | 22M parameters, 90 MB | about 4 s | English; trained on web search (MS MARCO) |
  | `BAAI/bge-reranker-base` | 278M, 1.1 GB | about 28 s | English and Chinese |
  | `BAAI/bge-reranker-v2-m3` | 568M, 2.3 GB | about 60–75 s | Multilingual and strongest; for a GPU. Set `max_length: 1024` |

  On the real HR policies, whose chunks are shorter, the default takes about **1.5 s per question** warm (bge-reranker-v2-m3: about 62 s). Fewer CPU threads don't help (4 threads: 80 s). The evaluation in step 7 measures what the faster model costs in ranking quality.
- **What it reads:** each candidate's full text, including its contextual header ("Policy: … / Section: …"), so it knows where each chunk comes from.
- **Scores:** higher is more relevant. The scale depends on the model: MiniLM gives raw scores (for example 1.2 for a match and -9 for a miss), `bge-reranker` gives 0 to 1. Compare them only within one query.
- **Speed:** this is the slowest step in the query path. Dense search takes about 0.5 s warm, and sparse and hybrid under 0.05 s. `retrieval.fused_limit` (fewer candidates) trades speed for recall.

## Small-to-big

Small chunks match questions precisely, while whole sections give the LLM the full rule, with its exceptions and table rows. After reranking, `SectionExpandingRetriever` replaces a hit on one part of a split section with the whole section:

- All parts are fetched with the caller's security filter (`DocumentIndex.section_chunks`).
- They are joined in order. The contextual header appears once, and the lines one part repeats from the previous one (the chunking overlap) are dropped.
- Two hits in one section become one result, at the better hit's position, keeping its scores.
- A section over `max_section_tokens` (1,500) isn't expanded; the hit is returned as it is.
- Glossary terms (`chunk_type: definition`) are never expanded into the whole glossary.
- An expanded result has `section_expanded: true`, `section_chunk_ids` and the section's full page range.

**On the 5 HR PDFs it doesn't change anything yet:** every section fits in one chunk (up to 800 tokens). It matters for longer policies, and the tests cover it with a long synthetic section.

## Configuration

```yaml
reranker:
  enabled: true
  model: cross-encoder/ms-marco-MiniLM-L-6-v2   # or BAAI/bge-reranker-v2-m3 on a GPU
  device: cpu
  batch_size: 8
  max_length: 512
retrieval:
  mode: hybrid_rerank    # dense | sparse | hybrid | hybrid_rerank
  dense_limit: 50
  sparse_limit: 50
  fused_limit: 20        # candidates for the reranker
  top_k: 6
  expand_sections: true
  max_section_tokens: 1500
```

`hybrid_rerank` with `reranker.enabled: false` is refused with a clear error, instead of silently running another mode.

## Tracing

LangChain traces the retrievers automatically (`SectionExpandingRetriever` → `ContextualCompressionRetriever` → `PolicyRetriever`), with the query embedding as `embed`. Two runs of our own are added:

- **`rerank`:** the number of candidates, and for each kept chunk its ID, source, section, retrieval rank, rerank rank and score.
- **`expand_sections`:** the number of hits, and how many were expanded.

These two runs record IDs, ranks and scores, never chunk text. The retriever runs that LangChain records do contain the retrieved text, as query traces do (see the root README for hiding inputs and outputs).

## Tests

| File | Covers |
|---|---|
| `tests/test_retrieval.py` | The leakage matrix from step 3b (12 callers × 6 documents) through all four modes with small-to-big on; each mode finds a document by its unique term; scores and ranks; reranking of `fused_limit` candidates down to `top_k`; relevance filters (superseded versions, categories, never widening access); small-to-big (whole section, overlap removed, one header, duplicates merged, token cap, permissions, definitions) |
| `tests/test_tracing.py` | `rerank` and `expand_sections` runs carry ranks and scores but no chunk text |
| `tests/test_retrieval_live.py` (`-m integration`) | The real MiniLM and `bge-reranker-v2-m3` both rank the chunk that answers above a lexical look-alike and an unrelated chunk |

The unit tests use the `hashing` encoder and a keyword-counting cross-encoder, so no model is downloaded.
