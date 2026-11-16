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
- Our own LangChain `BaseLoader`: `PdfPageLoader` (on `pypdf`, keeps page numbers), producing `Document` objects.
- `RecursiveCharacterTextSplitter`, contextual chunk headers, metadata and content hashes.
- **Done when:** tests on sample files produce the expected chunks and metadata.

## Step 3: Embeddings and the Qdrant store ✅
- A `bge-m3` encoder behind LangChain's `Embeddings` (dense) and `SparseEmbeddings` (sparse) interfaces. Both wrappers share one cached encoder, so each chunk goes through the model once. A `hashing` provider stands in for tests and offline demos.
- `QdrantVectorStore` (`langchain-qdrant`) in hybrid mode, plus our own `DocumentIndex`: collection creation and validation, payload indexes, batched adds, skipping unchanged files (by file hash), replacing changed files without a gap (write new chunks, then delete stale ones), repairing interrupted writes, deletion by document, and listing documents.
- `scripts/ingest.py` to index files from the command line.
- **Done when:** ingesting sample files fills Qdrant, re-running it changes nothing, and the data is visible in the Qdrant dashboard. Tests use Qdrant's in-memory mode.

## Step 3a: HR policy chunking and metadata
- The HR policy PDFs in `data/pdfs/` become the main data set: HR Policy Handbook, Employee Benefits Guide, Travel Policy, Expense Policy, and the Annual Review, Merit Increase and Performance Bonus Policy (a candidate for a restricted document in step 3b).
- Layout-aware PDF parsing (Docling or `pymupdf4llm`, chosen after a quick comparison) that keeps headings and tables and removes repeated page headers and footers.
- Section-aware chunking across page breaks: a section stays whole if it fits; longer ones split on paragraphs, list items, then sentences, with overlap only inside a section. Table rows, list items, rules with their exceptions, and definitions are never split. Chunk size measured in tokens (target 300–500, maximum 800).
- The heading path and policy details in the contextual header (`Policy: Travel Policy (v4, effective 2026-01-01, UK)` / `Section: 4 Air travel > 4.2 Class of travel`).
- Policy metadata from a policy registry file: `policy_name`, `policy_category`, `section_number`, `section_path`, `version`, `effective_date`, `is_current`, `jurisdiction`, `applies_to`, `owner_department`, `review_date`, `language`.
- **Done when:** tests show sections kept whole, tables and exceptions not split, correct heading paths and page ranges, and registry metadata on every chunk.

## Step 3b: Authorization metadata
- The access fields from FEATURE.md section 11 on every chunk: `tenant_id`, `classification` and `classification_rank`, `visibility`, `allowed_groups`, `allowed_roles`, `allowed_users`, `access_version`, `access_hash` and `access_updated_at`, read from the policy registry. Deny by default.
- Tenant-scoped `doc_id` (`uuid5(tenant_id + source)`). This changes existing IDs, so the collection is re-created with `scripts/ingest.py --reset`.
- Payload indexes: `tenant_id` with `is_tenant=True`, keyword indexes on `visibility` and the allow lists, an integer index on `classification_rank`.
- `index()` compares `access_hash` as well as `doc_hash`. `set_access(doc_id, …)` changes permissions through `set_payload`, with no re-embedding.
- A `Principal` (tenant, user, groups, roles, clearance) and one function that builds the Qdrant security filter from it. `list_documents`, `get_document`, `delete` and `count_chunks` require a principal.
- `scripts/ingest.py` gains `--tenant` and `--registry`.
- **Done when:** leakage tests pass for every rule (tenant, classification ceiling, each kind of grant, deny by default), including exact-match queries and list/get; a permission change updates all chunks without embedding anything.

## Step 4: Hybrid retrieval and reranking
- Retrievers from `QdrantVectorStore` in dense, sparse and hybrid mode (Qdrant does the RRF fusion). Every retriever requires a `Principal`, and its security filter is always applied. Relevance filters (`is_current`, `jurisdiction`, `applies_to`, document, file type) are separate and optional.
- Small-to-big retrieval: matched chunks are expanded to their parent section before reranking or generation.
- Reranking with `ContextualCompressionRetriever` and `CrossEncoderReranker`, plus our own `BaseCrossEncoder` running `bge-reranker-v2-m3` through `sentence-transformers`.
- **Done when:** sample queries return relevant chunks, and you can switch between dense-only, sparse-only, hybrid and hybrid plus rerank. The evaluation script in step 7 needs these modes. Leakage tests from step 3b also pass through every retriever and the reranker.

## Step 5: Answer generation (LCEL chain)
- An LCEL chain: retriever → format numbered sources → `ChatPromptTemplate` → `FailoverChatModel` → output parser.
- Context building: removes duplicates, keeps within a token budget, and orders the chunks.
- A confidence threshold for "I don't know" answers.
- Answer parsing that links citations back to the policy, section, page and effective date.
- HR answer rules: the more specific policy wins over the global one (and the answer says so), weak evidence gives "please contact HR", and no advice on an individual's situation.
- Optional LangSmith tracing, switched on with environment variables.
- **Done when:** questions end to end return grounded answers with correct citations, and still do when failover to Anthropic is forced (once that key is added).

## Step 6: FastAPI service
- `POST /ingest`, `POST /query` (with `debug` and `stream` options), `GET /documents`, `DELETE /documents/{id}`, `PUT /documents/{id}/access` and `GET /health`, which reports Qdrant and circuit-breaker states.
- API keys mapped to principals in config; every route scoped to the caller's principal. An audit log of who asked and which chunks (and `access_version`s) were retrieved.
- Request validation, error handling and structured logging.
- The API itself added to Docker Compose.
- **Done when:** API tests pass, the whole flow works through Swagger UI, and `docker compose up` runs everything.

## Step 7: Evaluation
- A golden dataset of 50–100 HR questions, each tagged with its expected policy and section, including exceptions, table lookups, country differences and outdated versions.
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
├── data/pdfs/        # HR policy PDFs
├── data/samples/     # small generated PDF used by the tests
├── config.yaml
├── docker-compose.yml
└── pyproject.toml
```

## Notes
- `bge-m3` and the reranker together download about 3 GB of model weights the first time they run.
- Steps 1 and 2 don't depend on each other.
- Steps 3a and 3b were added after step 3, when the domain became HR policies. They are numbered this way so later step numbers stay the same.
- Phases 2–4 from [FEATURE.md](FEATURE.md) come after step 8. LangGraph comes in with the agentic retrieval feature.
- LangChain's import paths have moved between versions. The project targets LangChain 1.x, where the retriever helpers live in `langchain-classic`.
