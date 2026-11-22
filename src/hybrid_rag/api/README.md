# API: the HR policy assistant over HTTP

A FastAPI service for asking questions and managing policies. Every route except `/health` needs an API key, which stands for one principal (tenant, user, groups, roles, clearance). Everything a caller sees, searches or changes is scoped to that principal.

## Running it

```bash
uv run python scripts/make_api_key.py --demo       # four demo keys; copy them, they're shown once
uv run uvicorn --factory hybrid_rag.api.app:create_app --port 8000
```

Then open **http://localhost:8000/docs** (Swagger UI), click **Authorize**, and paste a key.

With Docker (the API and Qdrant together):

```bash
uv run python scripts/make_api_key.py --demo       # creates api_keys.yaml, mounted read-only
docker compose up -d --build                       # first build: 10+ minutes; first start downloads models
docker compose exec api python scripts/ingest.py   # index data/pdfs inside the stack
```

Startup loads `bge-m3`, the reranker and Presidio (about 30 s warm, several minutes on the first start while the models download), so the first question isn't slow. `/health` answers once the API is ready.

## Routes

| Route | Key | What it does |
|---|---|---|
| `POST /query` | any | Answers a question from the policies you may read, with citations. `debug: true` adds the sources, scores, provider and rewritten question. `stream: true` returns server-sent events |
| `GET /documents` | any | The documents you may read. Admins also see who may read each one |
| `POST /ingest` | admin | Uploads PDFs (multipart, field `files`, optional `force`). Returns a job at once (202) |
| `GET /ingest/{job_id}` | admin | A job's status and each file's result: `added`, `updated`, `unchanged`, `quarantined` or `failed` |
| `DELETE /documents/{doc_id}` | admin | Deletes a document and its chunks, within your tenant |
| `PUT /documents/{doc_id}/access` | admin | Replaces a document's permissions without re-embedding (body: an `AccessPolicy`) |
| `GET /health` | none | The status of Qdrant and of each LLM provider's circuit breaker. No tenant data. 503 if Qdrant is down |

**Example:**

```bash
curl -s localhost:8000/query -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
  -d '{"question": "How many PTO days do I get after 4 years?"}'
```
```json
{"request_id": "2d509d83…", "answer": "20 PTO days per year … [1]", "status": "answered",
 "grounded": true,
 "citations": [{"number": 1, "label": "Employee Benefits Guide §7 Time Off and Leave (Employee_Benefits_Guide_2027.pdf, page 6)",
                "effective_date": "2027-01-01", "…": "…"}],
 "debug": null}
```

**Follow-up questions:** send the earlier turns in `history`, as `[{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]`, up to 20. The question is rewritten to stand on its own before retrieval.

**Streaming** (`"stream": true`) sends server-sent events:
- `event: token` events with `{"text": …}` while the answer is written;
- one `event: answer` with the same JSON as the normal response.

The final event is authoritative, because its text has passed the citation and output checks. The first token arrives after about 8 s, which is `gpt-5` thinking before it writes.

## API keys and principals

`api_keys.yaml` (`api.keys_file`, git-ignored) maps the **SHA-256 of each key** to a principal. The keys themselves are never stored.

```yaml
keys:
  - name: demo-manager
    sha256: 9f2c…
    principal: {tenant_id: dummymocktest, user_id: u-demo-manager, clearance: confidential,
                roles: [people_manager]}
```

- `scripts/make_api_key.py`:
  - `--demo` creates four demo keys (employee, manager, HR compensation, admin).
  - `NAME --user … --clearance … --group … --role … [--admin]` creates one key.
- **Disabling a key:** set `disabled: true` on its entry. Keys are read at startup, so restart the API after a change.
- **Where the principal comes from:** only this mapping, never request fields. Replacing it with JWT/OIDC claims later changes `auth.py` only.

## Errors

Every error has the same shape, with the request ID:

```json
{"error": {"code": "not_found", "message": "No such document", "request_id": "…"}}
```

| Status | When |
|---|---|
| 401 | Missing or unknown key |
| 403 | An admin route without a tenant-admin key |
| 404 | The document or job doesn't exist, **or you may not see it**. The two look the same, so nothing hidden is revealed |
| 413 / 422 | An upload over `api.max_upload_mb`, a file that isn't a PDF, an invalid body (an unknown field is an error) |
| 503 | Qdrant or every LLM provider is unavailable |

## Permission changes

`PUT /documents/{id}/access` updates every chunk in place (`set_payload`) and bumps `access_version`. Nothing is re-embedded.

The change is also saved as an **override** in `documents.access_overrides` (`.data/access_overrides.json`, git-ignored):
- **Ingests keep it:** every ingest, through the API or `scripts/ingest.py`, reads the overrides on top of the registry, so a later CLI run keeps the change instead of re-applying the registry's rules.
- **Deleting clears it:** deleting the document removes its override.
- **Back to the registry:** to return a document to the registry's rules, delete its entry from the overrides file and re-run `scripts/ingest.py`.

## Uploads

`POST /ingest` checks each file: a `.pdf` name, `%PDF-` content, at most `api.max_upload_mb` (25 MB), and at most `api.max_files_per_upload` (10). It then saves the files and queues a job.
- **One job at a time:** a single worker thread runs jobs one after another, so uploads don't compete for the CPU. A PDF takes about 2 minutes on a laptop CPU.
- **Uploads aren't kept:** the uploaded files are deleted when the job ends.
- **Your tenant only:** files are indexed for the caller's tenant.
  - **The configured tenant** (`documents.tenant_id`) uses the policy registry, so a file listed there gets its metadata and access rules.
  - **Any other file** is readable by tenant admins only (deny by default), until `PUT …/access`.
- **Quarantine:** a file containing high-risk personal data (an SSN, a card number, …) is quarantined and not indexed. The job reports the entity types, never the values.
- **Job status is kept in memory:** it's lost when the API restarts. The indexed documents aren't.
- **First upload after a start:** it loads Docling's layout models (about 10 s).

## Audit log

`logs/audit.jsonl` (`api.audit_log`, git-ignored) has one JSON line per question, upload, delete and permission change.

```json
{"ts": "…", "request_id": "…", "route": "/query", "tenant_id": "dummymocktest", "user_id": "u-1",
 "key": "demo-employee", "status_code": 200, "mode": "hybrid_rerank", "latency_ms": 9012,
 "answer_status": "answered", "grounded": true, "provider": "openai", "model": "gpt-5",
 "failed_over": false, "redactions": {"EMAIL_ADDRESS": 1}, "output_redactions": {},
 "question_sha256": "…", "question_chars": 41,
 "chunks": [{"chunk_id": "…", "doc_id": "…", "source": "Employee_Benefits_Guide_2027.pdf",
             "access_version": 1, "cited": true}]}
```

- **Which permissions applied:** `access_version` on each chunk tells you which permissions the chunk was retrieved under.
- **No question text by default:** only a hash and a length, because HR questions can be personal even after redaction. `api.audit_questions: true` adds the *redacted* question.
- **No raw personal data:** questions are redacted before anything is written, so the audit log never holds raw personal data.

## Logs and tracing

- **Request ID:** every request gets an `X-Request-ID`, kept from the request if it's valid and generated otherwise. It's returned in the response headers, put on every log line (JSON logs, `api.json_logs`), written to the audit log, and added as metadata on the LangSmith trace. So a log line or an audit entry leads to its trace.
- **Trace metadata and tags:**
  - **Metadata:** `request_id`, `tenant_id`, `route` and the retrieval `mode`, plus the answering `provider`, `model`, `answer_status`, `grounded` and `failed_over` once the answer is known.
  - **Tags:** `route:…`, `tenant:…`, `status:…`, `provider:…`.
- **The LangSmith anonymizer:** when tracing is on, the API sends traces through a LangSmith client with an anonymizer. It runs the privacy pattern check (emails, phone numbers, IDs) over every string before it leaves, as a last line of defence. Questions are already redacted before tracing.

## Files

| File | Contents |
|---|---|
| [`app.py`](app.py) | `create_app()`: startup, the request-ID middleware, error handlers |
| [`routes.py`](routes.py) | The routes and the `caller` / `admin` dependencies |
| [`schemas.py`](schemas.py) | Request and response models (the Swagger UI is built from them) |
| [`services.py`](services.py) | `Services` (index, answer service, keys, audit, jobs, overrides) and `build_services()` |
| [`auth.py`](auth.py) | `KeyStore`, `add_key()`, `hash_key()` |
| [`audit.py`](audit.py) | `AuditLog` |
| [`jobs.py`](jobs.py) | `IngestJobs`, the background ingestion worker |
| [`logs.py`](logs.py) | JSON logging with the request ID and tenant |

## Tests

`tests/test_api.py` runs the real app with in-memory Qdrant, the hashing encoder, a keyword reranker and scripted LLMs (no models, no API calls). It covers:
- **Keys:** 401 without a key or with an unknown one, 403 on admin routes, and the request ID.
- **Scoping:** document lists per caller; grants visible only to admins; delete only within your own tenant, with 404 for what you can't see.
- **Permission changes:** applied with nothing re-embedded, and kept by a later ingest.
- **Queries:** cited answers; the audit log, which keeps no question text or raw email; debug output without unreadable sources; streaming events; 422 on invalid queries; 503 when the LLM is down.
- **Uploads:** background ingest; jobs hidden from other tenants; quarantine reported without values; refused files.

## Not built

- **Rate limiting** per key.
- **Real identity:** OIDC/JWT instead of API keys.
- **A durable job queue:** job status survives restarts.
- **Clearing a permission override** through the API.
- **Several API workers** (`--workers N`): each would load its own models, and jobs and circuit breakers would be per process.
