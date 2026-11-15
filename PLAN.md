# Hybrid RAG: MVP Implementation Plan

Each step ends with something working and tested. Features and design decisions are described in [FEATURE.md](FEATURE.md).

## Step 0: Project setup ✅
- Python 3.12 project managed with `uv`, using a `src/` layout.
- Dependencies:
  - **LangChain:** `langchain-core`, `langchain-openai`, `langchain-anthropic`, `langchain-qdrant`, `langchain-text-splitters` and `langchain-classic` (needed for the reranking retriever in LangChain 1.x). `langchain-community` was used at first but removed in step 2, because it was sunset in May 2026.
  - **Models and storage:** `qdrant-client`, `FlagEmbedding` (provides `bge-m3`), `sentence-transformers` (runs the reranker), and the `openai` and `anthropic` SDKs (their exception types are used to classify errors for failover)
  - **API and config:** FastAPI, `pypdf`, `pydantic-settings`
  - **Dev tools:** pytest, ruff and httpx
- Config: a `config.yaml` for pipeline settings, plus a `.env` file for API keys, with a `.env.example` committed instead of the real file.
- Docker Compose with Qdrant.
- Linting with ruff and a basic GitHub Actions workflow running the tests.
- **Done when:** `docker compose up` starts Qdrant and `pytest` runs.

## Step 1: LLM layer with failover (LangChain chat models) ✅
- Provider factory: builds `ChatOpenAI` and `ChatAnthropic` from config, with the SDKs' own retries turned off. Providers without an API key are skipped.
- `FailoverChatModel`, a custom LangChain `BaseChatModel`: retries with increasing waits, sorting errors into "retry/fail over" and "fail immediately", circuit breakers, time limits, and streaming failover only before the first token. Records the provider that answered in `response_metadata`.
- **Done when:** unit tests with fake chat models pass for retries, failover order, the circuit breaker, streaming and caller errors, and a small script gets an answer from OpenAI through `FailoverChatModel` (and from Anthropic once that key is added).

*This comes first because it doesn't depend on anything else and is the most distinctive part of the design.*

## Step 2: Document loading and chunking ✅
- Our own LangChain `BaseLoader`s: `PdfPageLoader` (on `pypdf`, keeps page numbers), `TextFileLoader` and `CsvRowLoader`, all producing `Document` objects. CSV rows become `column: value` text.
- `RecursiveCharacterTextSplitter` for PDF and TXT, row grouping for CSV, contextual chunk headers, metadata and content hashes.
- **Done when:** tests on sample files produce the expected chunks and metadata.

## Step 3: Embeddings and the Qdrant store
- A `bge-m3` wrapper that implements LangChain's `Embeddings` (dense) and `SparseEmbeddings` (sparse) interfaces.
- `QdrantVectorStore` (`langchain-qdrant`) in hybrid mode, plus our own code for payload indexes, batched adds, deletion by document, and skipping chunks whose hash is unchanged.
- **Done when:** ingesting sample files fills Qdrant, re-running it changes nothing, and the data is visible in the Qdrant dashboard. Tests use Qdrant's in-memory mode.

## Step 4: Hybrid retrieval and reranking
- Retrievers from `QdrantVectorStore` in dense, sparse and hybrid mode (Qdrant does the RRF fusion), plus metadata filters.
- Reranking with `ContextualCompressionRetriever` and `CrossEncoderReranker`, plus our own `BaseCrossEncoder` running `bge-reranker-v2-m3` through `sentence-transformers`.
- **Done when:** sample queries return relevant chunks, and you can switch between dense-only, sparse-only, hybrid and hybrid plus rerank. The evaluation script in step 7 needs these modes.

## Step 5: Answer generation (LCEL chain)
- An LCEL chain: retriever → format numbered sources → `ChatPromptTemplate` → `FailoverChatModel` → output parser.
- Context building: removes duplicates, keeps within a token budget, and orders the chunks.
- A confidence threshold for "I don't know" answers.
- Answer parsing that links citations back to file, page or row.
- Optional LangSmith tracing, switched on with environment variables.
- **Done when:** questions end to end return grounded answers with correct citations, and still do when failover to Anthropic is forced (once that key is added).

## Step 6: FastAPI service
- `POST /ingest`, `POST /query` (with `debug` and `stream` options), `GET /documents`, `DELETE /documents/{id}` and `GET /health`, which reports Qdrant and circuit-breaker states.
- Request validation, error handling and structured logging.
- The API itself added to Docker Compose.
- **Done when:** API tests pass, the whole flow works through Swagger UI, and `docker compose up` runs everything.

## Step 7: Evaluation
- A small golden dataset built from the sample documents.
- A script comparing the four retrieval modes on Recall@k and MRR, writing a results table.
- **Done when:** the results table is produced and added to the README.

## Step 8: Documentation and polish
- A README with an architecture diagram, quick start, API examples, evaluation results, design decisions and notes on scaling.
- **Done when:** someone could clone the repo and get it running from the README alone.

## Proposed structure
```
hybrid-rag/
├── src/hybrid_rag/
│   ├── llm/          # provider factory, FailoverChatModel
│   ├── ingestion/    # LangChain loaders, splitters, metadata
│   ├── embedding/    # bge-m3 dense + sparse LangChain wrapper
│   ├── store/        # QdrantVectorStore setup and indexing
│   ├── retrieval/    # hybrid retriever, reranker
│   ├── generation/   # prompts, LCEL chain, citations
│   ├── api/          # FastAPI routes
│   └── config.py
├── tests/
├── eval/             # golden set and evaluation script
├── data/samples/
├── config.yaml
├── docker-compose.yml
└── pyproject.toml
```

## Notes
- `bge-m3` and the reranker together download about 3 GB of model weights the first time they run.
- Steps 1 and 2 don't depend on each other.
- Phases 2–4 from [FEATURE.md](FEATURE.md) come after step 8. LangGraph comes in with the agentic retrieval feature.
- LangChain's import paths have moved between versions. The project targets LangChain 1.x, where the retriever helpers live in `langchain-classic`.
