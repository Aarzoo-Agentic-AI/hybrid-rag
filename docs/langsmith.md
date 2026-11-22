# Tracing with LangSmith (optional)

[LangSmith](https://smith.langchain.com) records a trace of every LangChain run, with each step's inputs, outputs, timing and token usage. Tracing is off unless you turn it on. Nothing in the project needs it.

## 1. Create an API key

1. Sign up at https://smith.langchain.com. The free Developer plan is enough.
2. Go to **Settings → API Keys → Create API Key**.
3. Choose **Personal Access Token**, give it a name (for example `hybrid-rag-dev`) and copy it. It starts with `lsv2_pt_` and is only shown once.

## 2. Add it to `.env`

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

The project is created by the first trace. With Docker, restart the API afterwards (`docker compose up -d`), since `.env` is read when the container starts.

## 3. Send a test trace

```bash
uv run python scripts/llm_smoke.py --task query_rewrite
```

Then open the **hybrid-rag** project in LangSmith. Re-indexing (`scripts/ingest.py --force`) produces an ingestion trace, and any question through the API or `scripts/ask.py` produces an answer trace.

## What is traced

| Trace | Runs |
|---|---|
| Ingestion | `ingest` → `ingest_file` per PDF → `load_document`, `index_document` → `qdrant_upsert` → `embed` |
| Answer | `answer_question`: the question rewrite, retrieval, reranking, sources, prompt, LLM call and citations |
| LLM call | `FailoverChatModel`, with one run per provider attempt (for example `openai/gpt-5 (attempt 1)`), so retries and failovers are visible |

**API requests** carry metadata you can filter on: `request_id` (the same as the `X-Request-ID` header, the JSON logs and the audit log), `tenant_id`, `route` and `mode`, plus `provider`, `model`, `answer_status`, `grounded` and `failed_over` once the answer is known. Tags: `route:…`, `tenant:…`, `status:…`, `provider:…`.

## Privacy

- **Ingestion and embedding traces** contain only counts, file names, titles, statuses and timings: no document text, vectors or local paths.
- **Answer traces** contain the prompt and the answer, so they include policy text. The question is redacted before tracing, and the API sends traces through an anonymizer that masks emails, phone numbers and IDs in every string as a last line of defence.
- **To keep only the structure and timings,** set `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true`.

## Notes

- **How the settings are loaded:** `get_settings()` in [config.py](../src/hybrid_rag/config.py) copies the `LANGSMITH_*` and `LANGCHAIN_*` values from `.env` into the environment; nothing else is copied. A value already set in your shell takes priority.
- **To turn tracing off,** set `LANGSMITH_TRACING=false`.
- **Tests never send traces:** [tests/conftest.py](../tests/conftest.py) turns tracing off for every test run. For evaluation runs, set `LANGSMITH_TRACING=false` yourself, or the runs go to your project.

## Troubleshooting

| Error | Fix |
|---|---|
| `401 Unauthorized` | Check that `LANGSMITH_API_KEY` is the full key (not a placeholder), with no quotes or spaces, and that `LANGSMITH_ENDPOINT` matches your account's region |
| `403 Forbidden` | Set `LANGSMITH_WORKSPACE_ID` to the workspace the key belongs to |
