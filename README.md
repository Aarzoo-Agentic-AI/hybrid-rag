# hybrid-rag

Hybrid RAG for HR policies, built with LangChain: dense + sparse retrieval in Qdrant, cross-encoder reranking and LLM failover (OpenAI and Anthropic) over PDF documents, served as an API.

See [FEATURE.md](FEATURE.md) for the design and [PLAN.md](PLAN.md) for the implementation plan. Full documentation comes in step 8.

## Running the API

```bash
uv run python scripts/make_api_key.py --demo   # demo keys: employee, manager, hr-comp, admin
docker compose up -d --build                   # Qdrant + the API (first build takes 10+ minutes)
```

Open **http://localhost:8000/docs**, click **Authorize** and paste a key. You can then ask questions (`POST /query`), list your documents, and, with the admin key, upload PDFs, delete documents and change who may read them. Each key stands for one principal, so the same question gives an employee and a manager different answers. Without Docker, run the API with `uv run uvicorn --factory hybrid_rag.api.app:create_app`. Details are in the [API README](src/hybrid_rag/api/README.md).

## Making an API key

Each key stands for one person or app (a *principal*) and decides what they may read. `scripts/make_api_key.py` prints the key **once** and saves only its SHA-256 hash in `api_keys.yaml` (git-ignored), so copy it when it's shown. A lost key can't be recovered, only replaced.

```bash
# An employee: reads internal policies
uv run python scripts/make_api_key.py alice --clearance internal

# A manager: can also read the merit policy (confidential, restricted to people managers)
uv run python scripts/make_api_key.py bob --clearance confidential --role people_manager

# HR compensation
uv run python scripts/make_api_key.py carol --clearance confidential --group hr-compensation

# A tenant admin: can upload, delete and change who may read a document
uv run python scripts/make_api_key.py ops --clearance restricted --admin
```

| Option | Meaning |
|---|---|
| `name` | A label for the key, shown in the audit log |
| `--user` | The user ID (default: the name) |
| `--tenant` | The tenant (default: `documents.tenant_id` in `config.yaml`) |
| `--clearance` | `public`, `internal`, `confidential` or `restricted`: the highest classification the key may read |
| `--group`, `--role` | Groups and roles that documents can grant access to (repeat for several) |
| `--admin` | Tenant admin |
| `--demo` | Creates four demo keys: `demo-employee`, `demo-manager`, `demo-hr-comp`, `demo-admin` |

The API reads `api_keys.yaml` at startup, so **restart it after adding a key** (`docker compose restart api`). The file is mounted into the container, so keys made on your machine work in Docker too.

**Using a key:** send the printed `hrk_…` value in the `X-API-Key` header, or paste it under **Authorize** in Swagger UI. Don't send the `sha256` value from `api_keys.yaml`: that's the hash, and it's refused with 401.

```bash
curl -s localhost:8000/query -H "X-API-Key: hrk_..." -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days do I get after 4 years?"}'
```

**Disabling a key:** set `disabled: true` on its entry in `api_keys.yaml` and restart the API. To start over with fresh demo keys, delete `api_keys.yaml` and run `--demo` again. The old keys then stop working.

## Indexing the HR policies

```bash
docker compose up -d                  # start Qdrant
uv run python scripts/ingest.py       # index data/pdfs (unchanged files are skipped)
uv run python scripts/ingest.py --list
```

- **How the PDFs are processed:** Docling parses them, keeping headings and tables and dropping page headers. Each section becomes a chunk (split only when it's long), with its heading path, page range and the policy's metadata from [data/pdfs/registry.yaml](data/pdfs/registry.yaml).
- **First run:** it downloads Docling's layout models and `bge-m3`, and parses at about 4 s per page. Parsed PDFs are cached in `.cache/parsed/`.
- **Details:** see the [ingestion README](src/hybrid_rag/ingestion/README.md).

## Searching

```bash
uv run python scripts/search.py "How many PTO days do I get after 4 years?"
uv run python scripts/search.py "merit matrix" --as manager   # employee | manager | hr-comp | admin
uv run python scripts/search.py "COBRA" --compare             # dense, sparse, hybrid, hybrid_rerank
```

Hybrid search (dense + sparse `bge-m3`, fused with RRF in Qdrant) finds 20 candidates, and a cross-encoder reranker keeps the best 6. The default reranker, `ms-marco-MiniLM-L-6-v2` (90 MB), takes about 1.5 s per question on the HR policies on a laptop CPU; `bge-reranker-v2-m3` is stronger but needs a GPU to be usable (about a minute per question on a laptop CPU). Details are in the [retrieval README](src/hybrid_rag/retrieval/README.md).

## Asking questions

```bash
uv run python scripts/ask.py "How many PTO days do I get after 4 years?"
uv run python scripts/ask.py "What merit increase does a Meets rating get?" --as manager
uv run python scripts/ask.py "And what about shorter flights?" --history "Can I fly premium economy on a 9 hour flight?"
```

Answers come only from the policies the caller may read, and every statement cites its policy, section, page and effective date. When the policies don't answer a question, the assistant says so and points to HR. Personal data in questions is masked before it reaches the LLM or the traces. Each question makes one `gpt-5` call (about 7 s); follow-ups also make a short `gpt-5-mini` call to rewrite them. Details are in the [generation README](src/hybrid_rag/generation/README.md).

## Access control

Every policy belongs to a tenant (`documents.tenant_id` in [config.yaml](config.yaml), `dummymocktest` for the HR PDFs). Its `access:` block in the registry sets its classification and who may read it: the whole tenant, or named groups, roles or users. Searches are filtered by the caller's permissions inside Qdrant, so a chunk the caller can't read is never retrieved. A policy without access rules is visible only to tenant admins.

In the sample set, the merit and bonus policy is restricted to HR compensation, people managers and the HR director; the other four are visible to all employees. To change who can read a policy, edit its `access:` block and run `scripts/ingest.py` again. The permissions are updated in place, with nothing re-embedded. Details are in the [store README](src/hybrid_rag/store/README.md#access-control).

## Sensitive data

Before a PDF is chunked, personal data is replaced with placeholders such as `<PERSON>` and `<EMAIL>`. A document containing high-risk data, such as an SSN or a card number, is quarantined and not indexed. Every chunk is checked again before it is embedded. Settings are in the `privacy:` section of [config.yaml](config.yaml), and the details are in the [privacy README](src/hybrid_rag/privacy/README.md).

## Evaluation results

94 labelled HR questions ([eval/golden.yaml](eval/golden.yaml)) are run through each retrieval mode, each asked as an employee, manager or HR-compensation principal. Ranks are per chunk, as sent to the LLM:

| Mode | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR | Median latency |
|---|---|---|---|---|---|---|
| `dense` | 77.3% | 94.8% | 98.8% | 98.8% | 0.878 | 0.20 s |
| `sparse` | 50.6% | 70.3% | 77.9% | 86.0% | 0.634 | 0.21 s |
| `hybrid` | 62.8% | 90.7% | 95.9% | 100% | 0.787 | 0.21 s |
| **`hybrid_rerank`** (default) | **82.6%** | **95.3%** | 98.3% | 100% | **0.908** | 0.92 s |

- **The reranker earns its 0.7 s.** It puts the right section first for 82.6% of questions, against 77.3% for dense search alone.
- **Hybrid without reranking is worse than dense.** Sparse search is weak on reworded questions (65.6% Recall@5 on the 45 paraphrased ones, against 91.5% on keyword ones), and fusing it in pulls good dense results down. With the reranker on top, hybrid is the best mode.
- **No leaks:** the 12 questions about the restricted merit policy were also asked by an employee, in every mode. No chunk from that policy came back.

**End to end** (`--answers`, `gpt-5`, `hybrid_rerank`): 85 of 86 answerable questions were answered, cited an expected section and contained every expected figure, and 7 of 8 unanswerable questions were refused. The failures:
- **A sabbatical question:** the right section ranked 7th, just outside the 6 chunks sent to the LLM.
- **"Vacation days in Germany":** the assistant answered with the US PTO table and a caveat, instead of saying the policies are US-only.
- **An employee asking about a raise at the top of the salary range:** nothing restricted was retrieved, but the answer guessed "paid as a lump sum", which happens to be the restricted rule.

**The evaluation changed a setting.** The confidence gate (`answer.min_rerank_score`) was −8.5, calibrated by hand on 23 questions. At that value it refused 9 of the 86 answerable questions, most with the right section ranked first, and stopped no question the LLM wouldn't refuse anyway. It is now −12: a cost floor for questions nothing comes close to, with refusals left to the LLM.

The full results, with per-category tables and every miss, are in [eval/results/](eval/results/). How to run the evaluation and add questions is in the [evaluation README](eval/README.md).

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
| Answer | `answer_question`: rewrite, retrieval, reranking, sources, prompt, LLM and citations, as one trace (with the redacted question) | Built |

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

## Side-by-Side Comparison

| Feature | Advanced RAG | Hybrid RAG | Agentic RAG |
|---|---|---|---|
| **Core Mechanism** | Adds pre-retrieval and post-retrieval optimization layers. | Merges dense vector search with sparse keyword or graph search. | Uses autonomous AI agents to plan, route, and loop through tools. |
| **Workflow Style** | Linear pipeline with enhanced processing stages. | Parallel search execution merged via ranking algorithms. | Iterative, non-linear loops with self-correction and tool use. |
| **Latency & Cost** | Moderate latency and compute cost due to reranking. | Low latency (1.5–4s) and low cost ($0.003–$0.015/query). | High, variable latency and higher cost ($0.02–$0.15/query). |
| **Best Use Case** | General accuracy improvements over basic vector search. | Exact terms, SKUs, legal text, or technical identifiers. | Complex multi-hop research, live APIs, and multi-source tasks. |