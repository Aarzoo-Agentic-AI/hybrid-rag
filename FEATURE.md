# Hybrid RAG: Features and Design Decisions

## 1. Project decisions
- **Goal:** a portfolio project, designed so it could scale to production.
- **Data types:** PDF, TXT and CSV.
- **Interface:** API only (FastAPI), with no UI.
- **Orchestration:** LangChain. The pipeline is built from LangChain components and LCEL chains, with custom LangChain components where the project needs more than LangChain provides (failover, `bge-m3` embeddings).
- **Deployment:** a mix of self-hosted and hosted. Open-source models (embeddings, reranker) run locally; answers are generated through hosted LLM APIs (OpenAI primary, Anthropic fallback).

## 2. Chosen tech stack
| Layer | Choice |
|---|---|
| Orchestration | LangChain (`langchain-core` LCEL chains, plus `langchain-classic` for the reranking retriever) |
| Vector database | Qdrant in Docker, one collection with dense and sparse vectors, used through `langchain-qdrant` (`QdrantVectorStore`) |
| Embeddings | `bge-m3` (dense and sparse from one model), wrapped in LangChain's `Embeddings` and `SparseEmbeddings` interfaces |
| Fusion | Reciprocal Rank Fusion (RRF), done by Qdrant through `QdrantVectorStore`'s hybrid mode |
| Reranker | `bge-reranker-v2-m3` through LangChain's `CrossEncoderReranker` and `ContextualCompressionRetriever`, with our own `BaseCrossEncoder` running the model through `sentence-transformers` |
| LLM | `ChatOpenAI` (primary) + `ChatAnthropic` (fallback) behind a custom `FailoverChatModel` (see section 9) |
| Loaders and chunking | Our own LangChain `BaseLoader`s (`PdfPageLoader` on `pypdf`, `TextFileLoader`, `CsvRowLoader`) and LangChain's `RecursiveCharacterTextSplitter` |
| Tracing | LangSmith (optional, switched on with environment variables) |
| API | FastAPI |
| Packaging | Docker Compose and pytest |

**How LangChain is used.** Ready-made LangChain components are used wherever they fit. Where they don't cover what the project needs, the project adds its own components that implement LangChain's interfaces, so they still plug into any chain:
- `FailoverChatModel`, a custom chat model (retries, circuit breakers, time limits; see section 9)
- the `bge-m3` dense and sparse embedding wrapper
- the PDF, TXT and CSV loaders (`BaseLoader` subclasses) and the reranker's cross-encoder (`BaseCrossEncoder`), because `langchain-community`, which used to provide them, was sunset in May 2026
- extra metadata on documents (content hashes, page and row numbers, contextual headers)

## 3. Core retrieval: what makes it "hybrid"
- **Retrieval modes:** `QdrantVectorStore` supports dense, sparse and hybrid modes, so the evaluation can compare them directly.
- **Dense retrieval (vectors):** uses embeddings, so it finds passages with the same meaning even when the wording differs.
- **Sparse retrieval:** matches exact words, which is good for names, IDs, error codes and acronyms. Uses `bge-m3`'s sparse output.
- **Fusion:** combines the two ranked lists into one with RRF, which needs no tuning. Weighted score blending is an optional alternative.
- **Reranking:** a cross-encoder re-scores the top ~20–50 fused results down to the top 5–10. This is usually the biggest quality gain for the effort. In LangChain this is a `ContextualCompressionRetriever` with a `CrossEncoderReranker`, wrapped around the hybrid retriever. The cross-encoder itself is our own small `BaseCrossEncoder` (from `langchain-core`) that runs the model with `sentence-transformers`.

## 4. Ingestion pipeline
- **Document loaders:** `PdfPageLoader`, `TextFileLoader` and `CsvRowLoader`, our own subclasses of LangChain's `BaseLoader` (built on `pypdf` and Python's `csv` module), which all produce LangChain `Document` objects. Text encodings are detected automatically (UTF-8 first, then `charset-normalizer`).
- **Handling for each file type:**
  - **PDF:** one unit per page, split with `RecursiveCharacterTextSplitter`; chunks never cross page boundaries, so every chunk has an exact page number for citations. Words hyphenated across lines are re-joined. Pages without text (scanned pages) are skipped with a warning, since OCR isn't supported.
  - **TXT:** `RecursiveCharacterTextSplitter` with overlap; each chunk records its line numbers.
  - **CSV:** each row (or a small group of rows) becomes a chunk written as `column: value` lines, with headers kept as metadata. Empty cells and empty rows are left out.
- **Metadata extraction:** document ID, source file name (never the local path), title, file type, chunk ID and position, content hash, and the page, line or row location. Filters depend on this. The full schema is in `src/hybrid_rag/ingestion/README.md`.
- **Incremental indexing:** content hashing so unchanged documents aren't re-embedded, plus a way to delete or update documents.
- **Contextual chunk headers:** add the document name and location (for example `Document: Employee Handbook (handbook.pdf)` / `Location: page 3`) to each chunk before embedding. This is cheap and works well, and can be switched off in config.
- **Batched and idempotent:** ingestion runs in batches and can be re-run safely.

## 5. Query-side intelligence
- **Query rewriting:** turns a follow-up question in a conversation into a standalone question.
- **Multi-query expansion or HyDE:** generates variations of the query, or a hypothetical answer, to improve recall.
- **Metadata filtering:** for example "only CSV files" or "only this document".
- **Query routing:** sends a query to keyword-heavy search, semantic search, or no retrieval at all.

## 6. Generation
- **LCEL chain:** retriever → format numbered sources → `ChatPromptTemplate` → `FailoverChatModel` → output parser.
- **Grounded answers with citations:** each claim links back to its source (file, page or row).
- **"I don't know" handling:** when retrieval confidence is low, the system says so instead of guessing.
- **Streaming responses.**
- **Conversation memory:** for multi-turn chat.
- **Context management:** removes duplicate chunks, orders them well, and keeps the prompt within a token budget.

## 7. Evaluation and observability
- **Golden test set:** question → expected source or answer.
- **Retrieval metrics:** Recall@k, MRR and nDCG. Compare dense-only, sparse-only, hybrid, and hybrid plus rerank, so the results show whether the hybrid setup actually helps.
- **Answer metrics:** faithfulness and relevance, either with RAGAS or an LLM judge.
- **Tracing:** logs the query, retrieved chunks, scores, latency, token cost, and which LLM provider answered for every request.
- **LangSmith (optional):** setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` sends a trace of every chain run to LangSmith, showing each step's inputs, outputs and timing.

## 8. API and operations
- **Endpoints (FastAPI):**
  - `POST /ingest`: upload and index PDF, TXT or CSV files
  - `POST /query`: ask a question and get an answer with citations
  - `GET /documents`: list indexed documents
  - `DELETE /documents/{id}`: remove a document and its chunks
  - `GET /health`: health of the service, Qdrant and the LLM providers
- **Debug mode:** a `debug` flag on `/query` returns the retrieved chunks, scores, reranker output and provider details.
- **Config-driven pipeline:** swap models, chunk sizes, fusion settings and LLM providers through config files instead of code changes.
- **Caching:** for embeddings and repeated queries.
- **Docker Compose:** for local setup (API + Qdrant).

## 9. LLM abstraction layer with failover
**Built on LangChain's chat model interface.** All providers are LangChain chat models, so the rest of the code works with LangChain messages and never imports a provider SDK directly:
- `invoke(messages) -> AIMessage` and `stream(messages)`, which yields `AIMessageChunk`s, plus their async versions.
- Token usage comes from `AIMessage.usage_metadata`. The provider and model that answered, the latency, and whether a failover happened are added to `AIMessage.response_metadata`.

**Providers:**
- `ChatOpenAI` (`langchain-openai`), the primary. Model names come from config.
- `ChatAnthropic` (`langchain-anthropic`), the fallback: Claude Sonnet 5.5 for answers, Haiku 5.5 for query rewriting.
- A fake chat model for tests, which returns fixed responses and can simulate errors.
- The SDKs' own automatic retries are turned off (`max_retries=0`), so all retry decisions are made in one place: the failover model.

**`FailoverChatModel`.** A custom LangChain chat model (a `BaseChatModel` subclass) that wraps the provider models and tries them in the configured order. Because it is a normal chat model, it works anywhere in an LCEL chain. LangChain's built-in `with_fallbacks()` was considered, but it doesn't cover the rules below:
- **Retry first, then fail over.** Temporary errors (rate limits, server errors, overloaded responses, timeouts) are retried up to N times, with increasing waits and some randomness, before moving to the next provider.
- **No failover on caller errors.** Bad requests and authentication errors fail immediately. They are bugs or misconfiguration that switching providers wouldn't fix. Errors are classified from the OpenAI and Anthropic SDK exception types.
- **Circuit breaker for each provider.** After repeated failures, a provider is skipped for a cooldown period and then probed again. Requests don't keep waiting on a provider that's down.
- **Time limits.** Each request has its own timeout, and there's an overall deadline across all attempts.
- **Streaming.** A stream can fail over only before the first token arrives. If it fails mid-answer, the API returns an error rather than an answer stitched together from two models.

**Providers without an API key are skipped.** A provider is only active when its key is set in `.env`, so the system runs on OpenAI alone until an Anthropic key is added.

**Per-task routing in config.** Each task has its own ordered list of providers and models:
```yaml
llm:
  answer:
    - {provider: openai,    model: <configured>}
    - {provider: anthropic, model: claude-sonnet-5-5}
  query_rewrite:
    - {provider: openai,    model: <configured>}
    - {provider: anthropic, model: claude-haiku-5-5}
  retry: {max_attempts: 3, base_delay_s: 0.5}
  circuit_breaker: {failure_threshold: 5, cooldown_s: 30}
```
A small factory builds one `FailoverChatModel` for each task from this config.

**Prompts that work with either provider.** Citations and "I don't know" behaviour come from a `ChatPromptTemplate` with a numbered-sources format, not from features only one provider has. That way, answers stay consistent after a failover.

**Visibility:**
- Every response records which provider answered and whether a failover happened. The `/query` debug output includes this.
- `/health` reports each provider's circuit-breaker state.
- Logs record retries, failovers and token cost for each provider. With LangSmith switched on, each attempt also shows up in the trace.

**Tests:** unit tests use fake chat models to check retries, failover order, circuit-breaker behaviour, streaming failover, and that caller errors don't trigger failover. No API keys are needed.

## 10. Scaling to production
**Scale estimate.** Millions of documents usually means tens of millions of chunks. For 10 million chunks with `bge-m3` dense vectors (1024 dimensions):

| Storage | Approx. memory |
|---|---|
| Raw float32 vectors (10M × 4 KB) | ~40 GB |
| + HNSW graph overhead | ~45–50 GB |
| With scalar quantization (int8) | ~10–12 GB in RAM (originals kept on disk) |
| With binary quantization | ~1.5–2 GB in RAM (some accuracy loss, recovered by rescoring) |

**How Qdrant handles that scale:**
- **Quantization** shrinks the vectors held in RAM by 4x to 32x. Qdrant searches the compressed vectors, then rescores the top candidates with the full-precision originals.
- **On-disk storage** keeps vectors and payloads on SSD; only the index and quantized vectors stay in memory.
- **Distributed mode** splits a collection into shards across nodes and keeps replicas for high availability and more read capacity.
- **Payload indexes** keep filtered searches fast.
- **Multi-tenancy** lets many customers share one collection, each with a tenant ID.
- **Snapshots** for backup and recovery.

**Where the real bottlenecks are at that scale:**
1. **Embedding throughput during ingestion.** CPU embedding of tens of millions of chunks takes days. Production needs GPUs, batching and worker queues, or a hosted embedding API.
2. **Reranker latency.** The slowest step in the query path. Needs a GPU, fewer candidates, or a hosted reranker.
3. **Ingestion design.** Job queues, retries, deduplication and incremental updates rather than a single upload endpoint.
4. **Operations.** Monitoring, backups, rolling upgrades and capacity planning (or Qdrant Cloud).
5. **Retrieval quality.** As the corpus grows, more chunks look similar, so metadata filters, good chunking and reranking matter more.

**Design-for-scale choices in this project:**
- Every model, store and LLM provider sits behind a LangChain interface (`BaseChatModel`, `Embeddings`, `VectorStore`, `BaseRetriever`), so moving from local to GPU or hosted, or from one Qdrant node to a cluster, is a config change.
- Batched, idempotent ingestion using content hashes.
- Payload indexes on the fields we filter by.
- Quantization as a config option.
- An optional benchmark script measuring ingestion speed, query latency (p50/p95) and memory at around 100k chunks, with notes on how it would scale.

## 11. Advanced (later phases)
- **Graph RAG:** builds an entity and relationship graph alongside the vector and keyword indexes, as a third retriever.
- **Parent-child retrieval:** searches small chunks but sends their larger parent section to the LLM.
- **Agentic or iterative retrieval with LangGraph:** the LLM decides whether it needs another search.
- **CSV text-to-SQL:** handles calculation questions ("average price by region?") with SQL or pandas instead of retrieval.
- **GPU or hosted models:** switch embeddings and reranker to GPU or hosted APIs.
- **More LLM providers:** a third provider, such as local Ollama, as the last fallback, and routing based on cost or latency.
- **Multimodal:** tables and images in PDFs.
- **Access control:** each user only retrieves documents they're allowed to see.
- **Feedback loop:** thumbs up/down on answers, which feeds back into the evaluation set.

## 12. Phasing
1. **MVP:**
   - LangChain loaders and splitters → `bge-m3` → `QdrantVectorStore` → hybrid search with RRF → cross-encoder reranker
   - An LCEL chain that answers with citations and "I don't know" handling, through `FailoverChatModel` (OpenAI → Anthropic)
   - All endpoints from section 8
   - Evaluation script comparing dense-only, sparse-only, hybrid, and hybrid plus rerank (Recall@k, MRR)
   - Docker Compose, pytest tests, and a README with an architecture diagram
2. **Quality:** metadata filters, query rewriting, multi-query/HyDE, answer metrics.
3. **Production readiness:** incremental ingestion, caching, tracing (including LangSmith), quantization, benchmark script.
4. **Advanced:** pick from section 11.
