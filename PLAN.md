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
- LangSmith tracing of every ingestion step (`ingest` → `ingest_file` → `load_document`, `index_document` → `qdrant_upsert` → `embed`), recording counts, names, statuses and timings but never document text or vectors.
- **Done when:** ingesting sample files fills Qdrant, re-running it changes nothing, and the data is visible in the Qdrant dashboard. Tests use Qdrant's in-memory mode.

## Step 3a: HR policy chunking and metadata ✅
- **Data set:** the HR policy PDFs in `data/pdfs/` are now the main data set, and the default for `scripts/ingest.py`: HR Policy Handbook, Employee Benefits Guide, Travel Policy, Expense Policy, and the Annual Review, Merit Increase and Performance Bonus Policy (a candidate for a restricted document in step 3b).
- **Parser:** Docling, chosen over `pymupdf4llm` after comparing both on the 5 policies.
  - Both kept headings and tables. Only Docling dropped the running page headers.
  - Docling is MIT-licensed; PyMuPDF is AGPL.
  - Docling runs without OCR, with accurate table structure. Its results are cached by file hash, at about 4 s per page on the first parse.
  - A `pypdf` parser with numbered-heading heuristics stays for unit tests.
- **Sections:** levels and paths come from the numbering. The cover becomes "Overview". Tables of contents and acknowledgement forms are skipped.
- **Section-aware chunking**, sized in `bge-m3` tokens (target 400, maximum 800, overlap 60):
  - a section stays whole if it fits;
  - longer ones split at paragraph, list-item or table-row boundaries, with overlap only inside the section;
  - split tables repeat their headers;
  - rules keep their exceptions and lists keep their introductions;
  - glossary rows become definition chunks (17 terms).
- **Contextual header:** the policy details and section path, e.g. `Policy: Travel Policy (HR-TRV-003, effective 2027-01-01, US)` / `Section: 5 Air travel`.
- **Policy registry** (`data/pdfs/registry.yaml`), validated: `policy_name`, `policy_id`, `policy_category`, `version`, `effective_date`, `is_current`, `jurisdiction`, `applies_to`, `owner_department`, `review_date`, `language`.
- **Section metadata on each chunk:** `section_id`, `section_number`, `section_path`, `section_part`, `page`/`page_end`, `chunk_type`. There are payload indexes for the filter fields.
- **Sanitizer:** it now runs on every paragraph, list item and table row, and on the policy name. A new acronym rule removed a false positive ("Generative AI").
- **Fingerprint:** `pipeline_hash` covers the parser, tokenizer, chunking, privacy policy and registry entry. Changing any of them re-processes only the affected files, and re-chunking reads the parse cache.
- **Tests:** Docling's output for the 5 policies is saved as test fixtures (`scripts/make_parse_fixtures.py`), so unit tests check real structure without models. An integration test confirms that live Docling still matches.
- **Done when:** tests show sections kept whole, tables and exceptions not split, correct heading paths and page ranges, and registry metadata on every chunk. ✅

## Step 3b: Authorization metadata ✅
- The access fields from FEATURE.md section 11 on every chunk: `tenant_id`, `classification` and `classification_rank`, `visibility`, `allowed_groups`, `allowed_roles`, `allowed_users`, `access_version`, `access_hash` and `access_updated_at`, read from the policy registry. Deny by default.
- Tenant-scoped `doc_id` (`uuid5(tenant_id + source)`). This changes existing IDs, so the collection is re-created with `scripts/ingest.py --reset`.
- Payload indexes: `tenant_id` with `is_tenant=True`, keyword indexes on `visibility` and the allow lists, an integer index on `classification_rank`.
- `index()` compares `access_hash` as well as `doc_hash`. `set_access(doc_id, …)` changes permissions through `set_payload`, with no re-embedding.
- A `Principal` (tenant, user, groups, roles, clearance) and one function that builds the Qdrant security filter from it. `list_documents`, `get_document`, `delete` and `count_chunks` require a principal.
- `scripts/ingest.py` gains `--tenant` and `--registry`.
- **Built:**
  - `hybrid_rag/access.py`: `AccessPolicy` (the registry's `access:` block, closed by default and validated) and `Principal` (with `tenant_admin()` and `can_read()`, the rule in Python).
  - `store/security.py`: `security_filter()` and `scoped()`.
  - `DocumentIndex.search(query, principal)`: the base for the step 4 retrievers.
  - Ingestion applies a permission-only change before loading the file (`access_updated`).
  - The tenant is `documents.tenant_id` (`dummymocktest`). In the HR set, the merit and bonus policy is `confidential`, restricted to `hr-compensation`, `people_manager` and `hr_director`; the other four are `internal` and tenant-wide.
- **Done when:** leakage tests pass for every rule (tenant, classification ceiling, each kind of grant, deny by default), including exact-match queries and list/get; a permission change updates all chunks without embedding anything. ✅ 12 callers × 6 documents in `tests/test_access.py`, and the same matrix against the Qdrant server in `tests/test_store_live.py`. A mutation check (removing the tenant or classification condition from the filter) makes them fail.

## Step 3c: Sensitive-data sanitization ✅
*Built before steps 3a and 3b, because it changes what may enter the index at all.*
- A `privacy` package: a pattern detector with validation (emails, phones, SSNs, cards with the Luhn check, IBANs with mod-97, employee IDs), and Microsoft Presidio with the small spaCy model for person names and context-based IDs. A name filter removes spaCy's false positives on policy text (31 → 0 on the HR PDFs).
- Pass ① before chunking: redact personal data with typed placeholders; quarantine documents with high-risk data (`SensitiveDataError`, reported as `QUARANTINE`, never indexed). Pass ② after chunking: re-check every chunk.
- An allowlist (exact values and patterns) for shared business contacts; a `privacy:` section in `config.yaml`.
- `pii_scan` metadata on every chunk (status, detector, policy hash, counts by type, never values), with a payload index on the status. `pipeline_hash` (chunking settings plus privacy policy) re-processes documents when either changes.
- Unchanged files are skipped before loading, so a re-run with nothing to do stays under a second. Stale chunks are now any chunk not in the new version, which also covers re-chunking the same file.
- The `sanitize` step in LangSmith traces, with counts only.
- **Done when:** the 5 HR PDFs come out clean with their content intact; planted names, emails, phones and IDs are redacted; planted SSNs and cards quarantine the document; no value appears in Qdrant, traces, logs or errors; changing the privacy settings re-indexes. ✅

## Step 4: Hybrid retrieval and reranking ✅
- Retrievers from `QdrantVectorStore` in dense, sparse and hybrid mode (Qdrant does the RRF fusion). Every retriever requires a `Principal`, and its security filter is always applied. Relevance filters (`is_current`, `jurisdiction`, `applies_to`, document, file type) are separate and optional.
- Small-to-big retrieval: matched chunks are expanded to their parent section before reranking or generation.
- Tracing: retrievers are traced automatically by LangChain; `@traceable` on the cross-encoder (scores before and after reranking) and on small-to-big expansion. Query embedding is already traced as `embed`.
- Reranking with `ContextualCompressionRetriever` and `CrossEncoderReranker`, plus our own `BaseCrossEncoder` running a cross-encoder through `sentence-transformers`: `ms-marco-MiniLM-L-6-v2` by default, `bge-reranker-v2-m3` on a GPU.
- **Done when:** sample queries return relevant chunks, and you can switch between dense-only, sparse-only, hybrid and hybrid plus rerank. The evaluation script in step 7 needs these modes. Leakage tests from step 3b also pass through every retriever and the reranker.
- **Built:**
  - `DocumentIndex.search(query, principal, mode=...)` (dense, sparse, hybrid with separate prefetch limits) and `section_chunks()`, both always scoped to the principal.
  - `hybrid_rag/retrieval/`: `PolicyRetriever`, `RelevanceFilters`, `LocalCrossEncoder` and `ScoredReranker` (keeps `rerank_score`), `SectionExpandingRetriever`, and `build_retriever(settings, index, principal, mode=...)`.
  - `scripts/search.py` to try queries as an employee, manager, HR or admin, in one mode or all four (`--compare`).
  - Before step 4: `ingestion/cleanup.py` repairs reading-order errors (the Handbook's contents page spilling into section 2), with per-file `corrections` in the registry.
- **Results:** the leakage matrix (12 callers × 6 documents) passes in all four modes with small-to-big on; removing the security filter from `search()` fails 50 tests. On the HR PDFs no section is split, so small-to-big doesn't change results yet; it's tested on a synthetic long section.
- **Reranker speed:** `bge-reranker-v2-m3` took about 60 s per query on the development laptop's CPU (20 candidates of about 450 tokens; 568M parameters). The default is now `ms-marco-MiniLM-L-6-v2` (about 1.5 s per query on the HR policies); `bge-reranker-v2-m3` stays one config line away for a GPU. Step 7 measures what each costs in quality. Warm timings for the other modes: dense 0.5 s, sparse and hybrid under 0.05 s.

## Step 5: Answer generation (LCEL chain) ✅
- An LCEL chain: retriever → format numbered sources → `ChatPromptTemplate` → `FailoverChatModel` → output parser.
- Context building: removes duplicates, keeps within a token budget, and orders the chunks.
- A confidence threshold for "I don't know" answers.
- Answer parsing that links citations back to the policy, section, page and effective date.
- Privacy on the query side: personal data in questions is redacted before it reaches the LLM, traces and logs; answers get a final pattern check before they're returned.
- HR answer rules: follow what the policies say about precedence (the Travel and Expense policies say the Handbook governs if they differ), otherwise the more specific policy wins, and the answer says so; weak evidence gives "please contact HR"; no advice on an individual's situation.
- One LangSmith trace per question: query rewrite, retrieval, reranking, context building, prompt, `FailoverChatModel` and citation parsing, each as a nested run.
- **Done when:** questions end to end return grounded answers with correct citations, and still do when failover to Anthropic is forced (once that key is added).
- **Built:** `hybrid_rag/generation/` (`AnswerService` with `answer()` and `stream()`, prompts, context building, citation parsing), `Sanitizer.redact_text()` and `check_output()`, `reasoning_effort` per model in `llm:`, and `scripts/ask.py`.
- **Results:** real `gpt-5` answers on the HR policies are grounded and correctly cited (live tests: PTO, an out-of-scope question, and the merit policy as employee and as manager). The confidence threshold (−8.5 for MiniLM) stops 7 of 9 unanswerable test questions without an LLM call and lets all 14 answerable ones through (moved to −12 in step 7, after the golden-set evaluation). `reasoning_effort: low` cut answer latency from about 26 s to about 7 s. Failover to Anthropic is tested with fake models only, because `ANTHROPIC_API_KEY` is empty.

## Step 6: FastAPI service ✅
- `POST /ingest`, `POST /query` (with `debug` and `stream` options), `GET /documents`, `DELETE /documents/{id}`, `PUT /documents/{id}/access` and `GET /health`, which reports Qdrant and circuit-breaker states.
- A LangSmith anonymizer for query traces; redaction and quarantine counts in the audit log; `POST /ingest` reports quarantined files.
- API keys mapped to principals in config; every route scoped to the caller's principal. An audit log of who asked and which chunks (and `access_version`s) were retrieved.
- Request validation, error handling and structured logging.
- Every route traced, with `request_id`, `tenant_id`, retrieval mode and the answering provider as LangSmith metadata and tags, and the `request_id` also in the logs, so a log line leads to its trace.
- The API itself added to Docker Compose.
- **Done when:** API tests pass, the whole flow works through Swagger UI, and `docker compose up` runs everything.
- **Built:** `hybrid_rag/api/` (routes, schemas, services, key store, audit log, background ingest jobs, JSON logging with request IDs), `AccessOverrides` (API permission changes survive CLI ingests), the LangSmith anonymizer client, provider and status as trace metadata, `scripts/make_api_key.py`, the `Dockerfile` (CPU-only PyTorch on Linux) and the `api` service in Docker Compose.
- **Decisions:** ingestion runs as a background job (a PDF takes minutes on CPU); API keys stored as SHA-256 hashes in a git-ignored file; the audit log keeps a hash of the question, not its text, by default; 404 for documents you may not see, the same as for missing ones.

## Step 7: Evaluation ✅
- A golden dataset of 50–100 HR questions, each tagged with its expected policy and section, including exceptions, table lookups, country differences and outdated versions.
- A script comparing the four retrieval modes on Recall@k and MRR, writing a results table.
- **Done when:** the results table is produced and added to the README.
- **Built:** `eval/golden.yaml` (94 questions: facts, table lookups, exceptions, cross-policy, definitions, follow-ups, restricted access, unanswerable; half reworded), `eval/evaluate.py` (Recall@1/3/5/10, MRR and latency per mode, by category and style; a leakage check; confidence-gate calibration; `--answers` for end-to-end checks), `hybrid_rag.evaluation`, and the results in `eval/results/` and the README.
- **Results:** `hybrid_rerank` is best (Recall@1 82.6%, MRR 0.908). `hybrid` without reranking is worse than `dense`, because sparse search is weak on reworded questions. No restricted chunk leaked. End to end, 85 of 86 answerable questions were fully correct, and 7 of 8 unanswerable ones were refused.
- **Decisions:**
  - Labels are policy sections, not chunk IDs, so the dataset survives re-chunking.
  - Answers are checked with strings rather than an LLM judge: cheap and repeatable, but strict.
  - No country-difference or superseded-version questions, because all five policies are current and US-only. The filters behind them are unit-tested.
  - `min_rerank_score` moved from −8.5 to −12, after the evaluation showed that −8.5 refused 9 answerable questions.

## Step 8: Documentation and polish ✅
- A README with an architecture diagram, quick start, API examples, evaluation results, design decisions and notes on scaling.
- **Done when:** someone could clone the repo and get it running from the README alone.
- **Built:** the README rewritten (two Mermaid diagrams, a Docker-only quick start with troubleshooting, API examples, how it works, evaluation results, 13 design decisions with trade-offs, scaling, development, limitations and roadmap); the LangSmith guide moved to `docs/langsmith.md`; a status note in `FEATURE.md`. For the Docker-only path: API keys in `secrets/` (Compose mounts the folder) and a `keys` tool service, so no local Python install is needed.
- **Verified:** from a clean copy of the repo, following the README exactly, key creation and API startup work. Not verified: indexing inside Docker, which doesn't fit in the 2 GB this machine's Docker is limited to. The README says it needs more than 2 GB and offers indexing on the host instead, which is how the project was run during development.
- **Decisions:**
  - The quick start indexes before starting the API, so only one process holds the models in memory.
  - No license: all rights reserved.

## Step 9: More chunks, runtime evaluation and agentic retrieval ✅
Added after the plan, from the README's roadmap.
- **More chunks:** an answer evaluation with 8 chunks (and a 4,000-token budget) against 6. 8 fixed the sabbatical miss and an employee's guessed restricted rule, broke nothing, and cost about 1.5 s per answer, so it's the default now.
- **Runtime evaluation:**
  - `POST /feedback`: only the asker, once, within a week. The rating and masked comment are stored; a thumbs-down also keeps the masked question. With tracing on, it's sent to the answer's trace as `user_rating`, with the trace's project ID.
  - A sampled faithfulness check (`runtime_eval`, off by default): `gpt-5-mini` in a background thread, written to the audit log and sent as `faithfulness` feedback.
  - `scripts/runtime_report.py`: by day, with thumbs-down questions drafted as golden-set candidates.
- **Agentic retrieval:** a LangGraph loop (retrieve → grade → search again) as `strategy: agentic`. Generation, gating, citations and permissions are shared with the single pass, and the grader chooses only query text.
- **Evaluation:** `--strategy` and `--rescore` (re-check saved answers after fixing a fact, without new LLM calls).
- **Results:**
  - The single pass with 8 chunks gets 106 of 106 checks right; agentic gets 104 of 106, with twice the searches and 7.7 s against 4.9 s per answer.
  - Live in Docker: feedback, its link to the trace, the faithfulness check (score 1.0) and an agentic answer all worked.
- **Decisions:**
  - 8 chunks by default.
  - Agentic retrieval opt-in, because it shows no gain on this golden set.
  - Faithfulness checks in-app rather than as LangSmith online evaluators: they work with tracing off and are tested.
  - The masked question is kept only with a thumbs-down.
- **Found by the live check:**
  - `build_services` didn't pass the agent LLM; a test now builds the real services.
  - Feedback without the trace's project ID is deprecated in LangSmith, so the project ID is looked up once and sent with every piece of feedback.

## Proposed structure
```
hybrid-rag/
├── src/hybrid_rag/
│   ├── llm/          # provider factory, FailoverChatModel
│   ├── ingestion/    # Docling/pypdf parsers, sections, chunking, policy registry
│   ├── privacy/      # sensitive-data detection and redaction
│   ├── embedding/    # bge-m3 dense + sparse LangChain wrapper
│   ├── store/        # QdrantVectorStore setup, indexing, security filter
│   ├── retrieval/    # hybrid retriever, reranker
│   ├── generation/   # prompts, LCEL chain, citations
│   ├── api/          # FastAPI routes
│   ├── access.py     # AccessPolicy and Principal
│   └── config.py
├── tests/
│   └── fixtures/     # Docling output for the HR PDFs (scripts/make_parse_fixtures.py)
├── eval/             # golden set and evaluation script
├── data/pdfs/        # HR policy PDFs and registry.yaml
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
