# hybrid-rag

Hybrid RAG for HR policies, built with LangChain: dense + sparse retrieval in Qdrant, cross-encoder reranking and LLM failover (OpenAI and Anthropic) over PDF documents, served as an API.

See [FEATURE.md](FEATURE.md) for the design and [PLAN.md](PLAN.md) for the implementation plan. Full documentation comes in step 8.

## Indexing the HR policies

```bash
docker compose up -d                  # start Qdrant
uv run python scripts/ingest.py       # index data/pdfs (unchanged files are skipped)
uv run python scripts/ingest.py --list
```

- **How the PDFs are processed:** Docling parses them, keeping headings and tables and dropping page headers. Each section becomes a chunk (split only when it's long), with its heading path, page range and the policy's metadata from [data/pdfs/registry.yaml](data/pdfs/registry.yaml).
- **First run:** it downloads Docling's layout models and `bge-m3`, and parses at about 4 s per page. Parsed PDFs are cached in `.cache/parsed/`.
- **Details:** see the [ingestion README](src/hybrid_rag/ingestion/README.md).

## Access control

Every policy belongs to a tenant (`documents.tenant_id` in [config.yaml](config.yaml), `dummymocktest` for the HR PDFs). Its `access:` block in the registry sets its classification and who may read it: the whole tenant, or named groups, roles or users. Searches are filtered by the caller's permissions inside Qdrant, so a chunk the caller can't read is never retrieved. A policy without access rules is visible only to tenant admins.

In the sample set, the merit and bonus policy is restricted to HR compensation, people managers and the HR director; the other four are visible to all employees. To change who can read a policy, edit its `access:` block and run `scripts/ingest.py` again. The permissions are updated in place, with nothing re-embedded. Details are in the [store README](src/hybrid_rag/store/README.md#access-control).

## Sensitive data

Before a PDF is chunked, personal data is replaced with placeholders such as `<PERSON>` and `<EMAIL>`. A document containing high-risk data, such as an SSN or a card number, is quarantined and not indexed. Every chunk is checked again before it is embedded. Settings are in the `privacy:` section of [config.yaml](config.yaml), and the details are in the [privacy README](src/hybrid_rag/privacy/README.md).

## Setting up LangSmith (optional tracing)

[LangSmith](https://smith.langchain.com) records a trace of every LangChain run, with each step's inputs, outputs, timing and token usage. In this project, each LLM request appears as a `FailoverChatModel` run, with every provider attempt nested under it (for example `openai/gpt-5-mini (attempt 1)`). Retries and failovers are therefore visible in the trace.

### 1. Create an API key
1. Sign up at https://smith.langchain.com. The free Developer plan is enough.
2. Go to **Settings → API Keys → Create API Key**.
3. Choose **Personal Access Token**, give it a name (for example `hybrid-rag-dev`) and copy it. It starts with `lsv2_pt_` and is only shown once.

### 2. Add it to `.env`
```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...      # paste the full key
LANGSMITH_PROJECT=hybrid-rag
```
Only in some cases:
```bash
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # if your account is in the EU region
LANGSMITH_WORKSPACE_ID=...                              # if your key has access to several workspaces
```
The project is created automatically by the first trace.

### 3. Send a test trace
```bash
uv run python scripts/llm_smoke.py --task query_rewrite
```
Then open the **hybrid-rag** project in LangSmith to see the run.

To see an ingestion trace, re-index the sample handbook:
```bash
uv run python scripts/ingest.py --force
```
It appears as one `ingest` trace: `ingest_file` for each PDF, with `load_document`, `index_document`, `qdrant_upsert` and `embed` nested inside.

### What is traced

| Trace | Runs | Status |
|---|---|---|
| Ingestion | `ingest` → `ingest_file` → `load_document`, `index_document` → `qdrant_upsert` → `embed` | Built |
| LLM call | `FailoverChatModel` → one run per provider attempt | Built |
| Query | rewrite, retrieval, reranking, prompt, LLM and citations, as one trace | Steps 4–5 |

### Notes
- **How the settings are loaded.** LangSmith reads its settings from environment variables. `get_settings()` in [config.py](src/hybrid_rag/config.py) copies the `LANGSMITH_*` and `LANGCHAIN_*` values from `.env` into the environment; no other values are copied. A value already set in your shell takes priority over `.env`.
- **To turn tracing off,** set `LANGSMITH_TRACING=false`.
- **Tests never send traces.** [tests/conftest.py](tests/conftest.py) switches tracing off for every test run.
- **Privacy.**
  - Ingestion and embedding traces contain only counts, file names, titles, statuses and timings: no document text, vectors or local paths.
  - LLM traces contain full prompts and responses, and once the RAG chain is built they include text from your documents. To keep only the structure and timings, set `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` in `.env`.
- **Troubleshooting:**

  | Error | Fix |
  |---|---|
  | `401 Unauthorized` | Check that `LANGSMITH_API_KEY` is the full key (not a placeholder), with no quotes or spaces, and that `LANGSMITH_ENDPOINT` matches your account's region |
  | `403 Forbidden` | Set `LANGSMITH_WORKSPACE_ID` to the workspace the key belongs to |
