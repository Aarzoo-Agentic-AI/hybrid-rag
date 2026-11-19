# Hybrid RAG: Features and Design Decisions

## 1. Project decisions
- **Goal:** a portfolio project, designed so it could scale to production.
- **Domain:** HR policies (employee handbook, benefits, leave, travel, expenses, conduct and similar). See section 10 for the chunking and metadata strategy this calls for.
- **Data source:** PDF files only (HR policy PDFs in `data/pdfs/`).
- **Access control:** every chunk carries authorization metadata, and every search is filtered by the caller's permissions inside Qdrant. See section 11.
- **Sensitive data:** personal data is redacted before chunking, and documents with high-risk data are quarantined, so nothing sensitive is embedded, stored or traced. See section 12.
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
| Loaders and chunking | Our own LangChain `BaseLoader` (`PdfPageLoader` on `pypdf`) and LangChain's `RecursiveCharacterTextSplitter` |
| Privacy | Our own validated patterns plus Microsoft Presidio with the spaCy model `en_core_web_sm`, before chunking (section 12) |
| Tracing | LangSmith (optional, switched on with environment variables) |
| API | FastAPI |
| Packaging | Docker Compose and pytest |

**How LangChain is used.** Ready-made LangChain components are used wherever they fit. Where they don't cover what the project needs, the project adds its own components that implement LangChain's interfaces, so they still plug into any chain:
- `FailoverChatModel`, a custom chat model (retries, circuit breakers, time limits; see section 9)
- the `bge-m3` dense and sparse embedding wrapper
- the PDF loader (a `BaseLoader` subclass) and the reranker's cross-encoder (`BaseCrossEncoder`), because `langchain-community`, which used to provide them, was sunset in May 2026
- extra metadata on documents (content hashes, page numbers, contextual headers)

## 3. Core retrieval: what makes it "hybrid"
- **Retrieval modes:** `QdrantVectorStore` supports dense, sparse and hybrid modes, so the evaluation can compare them directly.
- **Dense retrieval (vectors):** uses embeddings, so it finds passages with the same meaning even when the wording differs.
- **Sparse retrieval:** matches exact words, which is good for names, IDs, error codes and acronyms. Uses `bge-m3`'s sparse output.
- **Fusion:** combines the two ranked lists into one with RRF, which needs no tuning. Weighted score blending is an optional alternative.
- **Reranking:** a cross-encoder re-scores the top ~20–50 fused results down to the top 5–10. This is usually the biggest quality gain for the effort. In LangChain this is a `ContextualCompressionRetriever` with a `CrossEncoderReranker`, wrapped around the hybrid retriever. The cross-encoder itself is our own small `BaseCrossEncoder` (from `langchain-core`) that runs the model with `sentence-transformers`.

## 4. Ingestion pipeline
- **Layout-aware parsing:**
  - **Docling** (MIT licence) turns each PDF into headings, paragraphs, list items and tables (with rows and column headers), each with its page number. It also drops running page headers and footers.
  - Results are cached by file hash, because Docling takes about 4 s per page on CPU.
  - A fast `pypdf` parser (numbered headings only, built on our LangChain `PdfPageLoader`) stands in for tests and demos.
  - Pages without text (scanned pages) are skipped with a warning, since OCR isn't supported.
- **Section-aware chunking** (section 10):
  - one chunk per section where it fits, measured in `bge-m3` tokens;
  - splits only at paragraph, list-item or table-row boundaries, with overlap inside a section only;
  - tables repeat their headers, rules keep their exceptions, and glossary terms become their own chunks.
- **Metadata extraction:**
  - **Document and chunk:** document ID, source file name (never the local path), file type, chunk ID and position, content hash.
  - **Section:** section number, title, path and ID, and the page range.
  - **Policy:** policy metadata from the registry (section 10).
  - **Privacy and settings:** the privacy scan result (`pii_scan`), and a `pipeline_hash` of the parser, chunking, privacy and registry settings (section 12).
  - Authorization metadata (section 11) comes in step 3b.
  - The full schema is in `src/hybrid_rag/ingestion/README.md`.
- **Sensitive-data sanitization:** before chunking, personal data is redacted and documents with high-risk data are quarantined; every chunk is checked again before embedding (section 12).
- **Incremental indexing:** content hashing so unchanged documents aren't re-embedded, plus a way to delete or update documents. An unchanged file is skipped before it's even loaded; a change to the chunking or privacy settings re-processes it.
- **Contextual chunk headers:** add the policy and section path to each chunk before embedding, for example `Policy: Travel Policy (HR-TRV-003, effective 2027-01-01, US)` / `Section: 5 Air travel`. This is cheap and works well, and can be switched off in config.
- **Batched and idempotent:** ingestion runs in batches and can be re-run safely.

## 5. Query-side intelligence
- **Query rewriting:** turns a follow-up question in a conversation into a standalone question.
- **Multi-query expansion or HyDE:** generates variations of the query, or a hypothetical answer, to improve recall.
- **Metadata filtering:** for example "only this policy" or "only current versions".
- **Query routing:** sends a query to keyword-heavy search, semantic search, or no retrieval at all.

## 6. Generation
- **LCEL chain:** retriever → format numbered sources → `ChatPromptTemplate` → `FailoverChatModel` → output parser.
- **Grounded answers with citations:** each claim links back to its source (policy, section and page).
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
- **Every step is traced.** LangChain components (chat models, retrievers, LCEL chains) are traced automatically. Our own plain-Python steps use LangSmith's `@traceable`, through `hybrid_rag/tracing.py`, so ingestion and queries each appear as one trace tree:

  | Trace | Runs (nested) | Status |
  |---|---|---|
  | Ingestion | `ingest` → `ingest_file` (one per PDF) → `load_document`, `index_document` → `qdrant_upsert` → `embed` | ✅ Built |
  | LLM call | `FailoverChatModel` → one run per provider attempt | ✅ Built |
  | Query | `query` → query rewrite (LLM) → retriever (dense / sparse / hybrid) → `embed` (query) → reranker (scores before and after) → small-to-big expansion → prompt → `FailoverChatModel` → citation parser | Steps 4–5 |
  | API request | Each route's trace carries `request_id`, `tenant_id`, retrieval mode and the answering provider as metadata and tags | Step 6 |

- **What traces contain.** Ingestion and embedding runs record counts, names, statuses and timings only: never document text, vectors or local paths (a unit test checks this). Query traces do contain the question, the retrieved chunks and the answer, which is what makes them useful. For restricted HR documents, `LANGSMITH_HIDE_INPUTS=true` and `LANGSMITH_HIDE_OUTPUTS=true` keep the trace structure and timings but drop the text. Each environment uses its own LangSmith project.
- **Short-lived scripts** call `flush_traces()` before exiting, so queued runs aren't lost.

## 8. API and operations
- **Endpoints (FastAPI):**
  - `POST /ingest`: upload and index PDF files
  - `POST /query`: ask a question and get an answer with citations
  - `GET /documents`: list indexed documents
  - `DELETE /documents/{id}`: remove a document and its chunks
  - `PUT /documents/{id}/access`: change a document's permissions without re-embedding
  - `GET /health`: health of the service, Qdrant and the LLM providers
- **Authentication:** every request carries an API key that maps to a principal (tenant, user, groups, roles, clearance). Results, document lists and deletes are scoped to that principal (section 11).
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

## 10. HR policy documents: chunking, metadata and embedding
HR policies have a recognisable structure, and retrieval works best when chunks follow it rather than a fixed size.

**What's distinctive about HR policies:**

| Trait | Example | What it means for the pipeline |
|---|---|---|
| Numbered sections with headings | `4.2 Class of travel` | Chunk by section, and keep the heading path |
| Rules with conditions or exceptions | "…economy class, **unless** the flight exceeds 6 hours" | Never separate a rule from its exception |
| Tables of limits | Per-diem rates by city, benefit tiers by grade | Keep each row together with its column headers |
| Definitions used throughout | "Eligible employee means…" | Make definitions findable on their own |
| Versions and effective dates | "Effective 1 Jan 2026, replaces v3" | Only current versions answer by default |
| Variants by country or employee group | UK and US leave rules; contractors and full-time staff | Use metadata filters, not luck |
| Cross-references | "See section 6.1 for approvals" | Keep the reference, and optionally follow it |
| Many words for one thing | PTO, vacation, annual leave, holiday | Hybrid search, query rewriting and a glossary |

**Chunking:**
1. **Parse the layout, not just the text.** `pypdf` loses headings and tables, so the layout-aware parser **Docling** is used. Docling and `pymupdf4llm` were compared on the 5 policies:
   - both recovered headings and tables;
   - only Docling dropped the running page headers;
   - Docling is MIT-licensed, whereas PyMuPDF is AGPL-3.0;
   - Docling is slower, so its results are cached.
2. **Chunk by section, across page breaks.** A section is the unit and stays whole if it fits. A longer section is split on paragraphs, then list items, then sentences, with about 10–15% overlap inside the section only and none across sections. Each chunk records the page (or pages) it came from, for citations.
3. **Size, in `bge-m3` tokens:** a target of 400 when splitting, and 800 as the hard maximum. On the 5 policies, every section fits whole, at 44–488 tokens.
4. **Never split** a table row (column headers are repeated in each row chunk), a list item, a rule together with its "unless / except / provided that" clause, or a definition.
5. **The heading path goes in the contextual header.** This is the biggest improvement for the least effort:
   ```
   Policy: Travel Policy (v4, effective 2026-01-01, UK)
   Section: 4 Air travel > 4.2 Class of travel

   Employees travel economy class unless…
   ```
6. **Small-to-big retrieval.** Search over the small chunks, then give the LLM the whole section (the parent) the match came from.
7. **Definitions are indexed separately**, as glossary chunks, and can be attached to a matched chunk that uses the defined term.

**Policy metadata** (relevance metadata, separate from the authorization metadata in section 11):

| Field | Example | Used for |
|---|---|---|
| `policy_name`, `policy_category` | `Travel Policy`, `travel` | Filtering, routing, citations |
| `section_number`, `section_path` | `4.2`, `Air travel > Class of travel` | Citations ("Travel Policy §4.2") |
| `version`, `effective_date`, `is_current` | `4`, `2026-01-01`, `true` | Only current versions answer; answers state the effective date |
| `jurisdiction` | `UK`, `US-CA`, `global` | Filtered by the caller's location |
| `applies_to` | `["full_time", "part_time"]` | Filtered by the caller's employee group |
| `owner_department`, `review_date`, `language` | `HR Ops`, `2026-12-01`, `en` | Governance and flagging stale policies |

These values come from a policy registry (see section 11), not from guessing at the text.

**Embedding:**
- **`bge-m3` fits well:** multilingual (important for global HR teams), long inputs, and dense plus sparse vectors from one pass.
- **Sparse vectors matter more than usual.** Exact terms such as `401(k)`, `FMLA`, `per diem` and `£45` are where dense vectors blur; hybrid search and the reranker cover them.
- **Contextual retrieval (optional):** an LLM writes one sentence of context for each chunk before embedding. An HR corpus is small, so the one-off cost is low, and the `doc_hash` skip means unchanged files cost nothing.
- **Hypothetical questions (optional):** 3–5 questions an employee might ask about each chunk are generated and indexed, pointing back to the chunk. This suits conversational HR queries ("can I fly business to Singapore?").
- **No embedding fine-tuning** until the evaluation shows a gap that structure, headers and the reranker don't close.

**Retrieval and answers:**
- Filters on `is_current`, the caller's `jurisdiction` and `applies_to`, on top of the authorization filter.
- **Conflicting policies:** when a country-specific and a global policy both match, the more specific one wins, and the answer says so.
- Citations name the policy, section and effective date, e.g. "Travel Policy §4.2 (effective 1 Jan 2026)".
- Weak evidence gives "I don't know, please contact HR". The system doesn't give advice on an individual's situation.
- Synonyms (PTO = annual leave = vacation) are expanded through query rewriting and a small glossary.

**Evaluation:** 50–100 questions based on real HR FAQs, each tagged with its expected policy and section, including cases that test exceptions, table lookups, country differences and outdated versions. Measured with recall@k by section, then answer faithfulness.

## 11. Authorization and access control
Permissions are enforced **inside the vector search**, as a Qdrant filter built from the authenticated caller. Results are never filtered after retrieval, and the LLM is never trusted to ignore documents. A chunk the caller can't see is never retrieved, so it can't leak through the answer, the citations or the traces.

**Authorization metadata.** Permissions are set per document and copied onto every chunk:

| # | Field | Type | Example | Purpose |
|---|---|---|---|---|
| 1 | `tenant_id` | keyword, required | `"lumen"` | Hard isolation between organisations; always filtered on |
| 2 | `doc_id`, `chunk_id` | keyword | UUIDs | Tenant-scoped: `doc_id = uuid5(tenant_id + source)`, so two tenants can both have a `handbook.pdf` |
| 3 | `classification` | keyword | `"confidential"` | Sensitivity level: `public`, `internal`, `confidential`, `restricted` |
| 4 | `classification_rank` | integer | `2` | The same level as a number (0–3), so the filter can check `rank <= caller.clearance` |
| 5 | `visibility` | keyword | `"tenant"` / `"restricted"` | `tenant`: everyone in the tenant may read it (handbook, travel policy) |
| 6 | `allowed_groups` | keyword list | `["hr-compensation", "dept:finance"]` | The main way to grant access; departments are groups (`dept:…`) |
| 7 | `allowed_roles` | keyword list | `["people_manager"]` | Access by job role |
| 8 | `allowed_users` | keyword list | `["u-1042"]` | Exceptions only, such as one person's file |
| + | `access_version`, `access_hash`, `access_updated_at` | int, keyword, date | `3`, `…`, timestamp | Audit trail, and spotting permission changes without re-embedding |

**The access rule.** Classification is a ceiling that always applies; the allow lists are grants, and any one is enough:
```
tenant_id == caller.tenant
AND classification_rank <= caller.clearance
AND ( visibility == "tenant"
      OR caller.user_id    in allowed_users
      OR caller.groups     ∩  allowed_groups
      OR caller.roles      ∩  allowed_roles )
```

**Examples:**

| Policy | Visibility | Grants | Classification |
|---|---|---|---|
| Employee Handbook | tenant | – | internal |
| Travel / Expense policy | tenant | – | internal |
| Manager guide to performance reviews | restricted | role `people_manager` | internal |
| Salary bands | restricted | group `hr-compensation`, role `hr_director` | confidential |
| Investigation procedures | restricted | group `hr-employee-relations` | restricted |

**Deliberately left out:**

| Left out | Reason |
|---|---|
| A separate `allowed_departments` field | A department is a group (`dept:finance`), and identity providers report it that way. Easy to add if an HR system needs it. |
| `owner_department`, `jurisdiction`, `applies_to` as permissions | They are relevance filters (which policy applies to me), not security. A relevance option must never loosen security, so they live in the policy metadata (section 10). |
| Per-chunk or per-section permissions | Permissions are per document. A sensitive part of a policy is published as its own document. |
| Access expiry dates | Not needed for the first version; easy to add later. |

**Rules:**
1. **Permissions come from a policy registry** (a YAML file, later the upload API), never from the document's own content or metadata.
2. **Deny by default, fail closed.** A document without permissions is visible only to its owner. A caller without a tenant or clearance gets nothing.
3. **Documents store grants, not memberships.** Who belongs to a group comes from the caller's identity at query time (an API-key mapping now, JWT/OIDC claims later), so group changes never need re-indexing.
4. **Permission changes don't re-embed.** Qdrant's `set_payload`, filtered by `doc_id`, updates every chunk in one call and bumps `access_version`. Indexing compares `access_hash` as well as `doc_hash`, so a file with unchanged content but changed permissions is still updated.
5. **Security filters are built on the server** from a `Principal` (tenant, user, groups, roles, clearance), never from request fields. They are kept apart from the relevance filters.
6. **The same filter applies everywhere:** search, list, get, delete, health counts and citations.
7. **Permissions are never embedded.** They stay out of the contextual header.
8. **Payload indexes:** `tenant_id` with Qdrant's `is_tenant=True` (per-tenant index layout), keyword indexes on `visibility` and the allow lists, and an integer index on `classification_rank`.
9. **Audit log:** who asked, which `chunk_id`s were retrieved, and each one's `access_version`.
10. **Beyond Qdrant:** LangSmith traces contain document text, so each environment gets its own LangSmith project. Any answer cache is keyed by caller. The embedding cache is safe, because it is keyed by text and holds only vectors.
11. **Leakage tests:** for every rule, a caller without access gets zero chunks, even for an exact-match query, and even through list and get.

## 12. Sensitive data protection
Sensitive data is removed **before chunking** and checked again **after chunking**, before anything is embedded, stored or traced. Once text is embedded, it exists in the vectors, the Qdrant payload, snapshots, LangSmith traces, logs and caches, and removing it from all of them is much harder.

```
PDF ─► extract ─► clean ─► ① DETECT + REDACT ─► chunk ─► ② VERIFY ─► embed ─► Qdrant
                          (whole page + title)          (each chunk)
                                  │
                                  └─► high-risk finding ─► QUARANTINE (not indexed)
```

| Stage | Why |
|---|---|
| **① Before chunking** | Detection needs context. A chunk boundary can split an entity (`123-45-` / `6789`) so that neither half matches, and names are recognised better in full sentences. |
| **② After chunking** | A second line of defence. The pattern detector re-checks every chunk, header included, and stops the document if anything survived. |
| **Before embedding** | Embeddings can be partly inverted back to text, so vectors made from personal data still count as personal data. |

**What happens to each kind of data:**

| Category | Examples | Action |
|---|---|---|
| High risk: should never be in a policy | SSN, card, bank account, IBAN, passport and ITIN numbers | **Quarantine** the whole document: not indexed, reported for HR review |
| Personal data | Employee names, personal emails and phones, employee IDs | **Redact** with typed placeholders (`<PERSON>`, `<EMAIL>`, `<PHONE>`, `<EMPLOYEE_ID>`) |
| Shared business contacts | `hr@company.com`, the HR service centre number | **Keep**, through an allowlist (exact values or patterns) |
| Policy content | "$300 hotel cap", "25 days", dates, cities | **Keep**: this is the knowledge base |
| Sensitive but legitimate | Salary bands, investigation procedures | **Restrict**, don't redact: classification and access control (section 11) |

**Detection:**
- **Patterns** (always on), each with validation: emails, North American phone numbers, SSNs (valid ranges only), cards (Luhn check), IBANs (mod-97 check) and employee IDs (a configurable pattern).
- **Microsoft Presidio** with the small spaCy model (`en_core_web_sm`, 13 MB) adds person names and context-based IDs (bank account, passport, ITIN).
- **Name filter:** spaCy's person guesses are noisy on policy text ("Roth", "Bereavement", "Tier 2"). A name is accepted only if it has at least two capitalised words, no digits, and no word that also appears in lower case in the same document. On the 5 HR PDFs this took false positives from 31 to 0, with all planted test names still found. It made the small model as accurate as the 560 MB one.

**Rules:**
1. **Irreversible:** placeholders, no encryption and no lookup table. Nothing reversible is stored.
2. **Values are never recorded.** Logs, traces, errors and metadata contain entity types, counts and page numbers only.
3. **Each chunk records its scan** as `pii_scan` (status, detector, policy hash, redaction counts). There is a payload index on the status.
4. **Re-scan when rules change.** The privacy policy hash is part of each chunk's `pipeline_hash`, so changing the privacy settings, or `SCANNER_VERSION`, re-processes every document on the next ingest.
5. **Safe by default.** Without an explicit sanitizer, ingestion uses the pattern sanitizer, so nothing is ever chunked unscanned. Scanning can only be switched off explicitly (`privacy.enabled: false`), and those chunks are marked `not_scanned`.
6. **Hashes:** `doc_hash` stays on the original file bytes, for change detection. `content_hash` is computed on the redacted text.
7. **Original PDFs** are kept outside the index, in restricted storage in production. Only redacted text is indexed, and citations still point to the page.
8. **Query side (step 5):** personal data in user questions is redacted before it reaches the LLM, traces or logs, and LLM answers get a final pattern check.
9. **Traces and audit (step 6):** a LangSmith anonymizer for query traces, and redaction counts in the audit log.
10. **Erasure:** deleting a document by `doc_id` removes all its chunks. Keeping personal data out of the index in the first place keeps GDPR erasure requests simple.

**Standards this follows:** GDPR (Art. 5 data minimisation, Art. 17 erasure, Art. 25 privacy by design), CCPA/CPRA for employee data, HIPAA if benefits documents ever contain an individual's health information, NIST SP 800-122 (protecting PII by risk level), ISO/IEC 27001 and 27701, and the OWASP Top 10 for LLM Applications (2025), LLM02 Sensitive Information Disclosure.

**Known limits:**
- Single-word names, or names made of common words, aren't redacted.
- Non-US national IDs need extra recognisers.
- Scanned PDFs have no text to scan, since OCR isn't supported.

## 13. Scaling to production
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

## 14. Advanced (later phases)
- **Graph RAG:** builds an entity and relationship graph alongside the vector and keyword indexes, as a third retriever.
- **Agentic or iterative retrieval with LangGraph:** the LLM decides whether it needs another search.
- **GPU or hosted models:** switch embeddings and reranker to GPU or hosted APIs.
- **More LLM providers:** a third provider, such as local Ollama, as the last fallback, and routing based on cost or latency.
- **Multimodal:** tables and images in PDFs.
- **Identity provider integration:** JWT/OIDC claims (tenant, groups, roles, clearance) instead of the API-key mapping.
- **Contextual retrieval and hypothetical questions** (section 10), if the evaluation shows they help.
- **Feedback loop:** thumbs up/down on answers, which feeds back into the evaluation set.

## 15. Phasing
1. **MVP:**
   - LangChain loaders and splitters → `bge-m3` → `QdrantVectorStore` → hybrid search with RRF → cross-encoder reranker
   - Section-aware chunking of HR policies with heading paths and policy metadata (section 10)
   - Authorization metadata and permission-filtered search, with leakage tests (section 11)
   - Sensitive-data sanitization before chunking, with quarantine and verification (section 12)
   - An LCEL chain that answers with citations and "I don't know" handling, through `FailoverChatModel` (OpenAI → Anthropic)
   - All endpoints from section 8
   - Evaluation script comparing dense-only, sparse-only, hybrid, and hybrid plus rerank (Recall@k, MRR)
   - Docker Compose, pytest tests, and a README with an architecture diagram
2. **Quality:** metadata filters, query rewriting, multi-query/HyDE, answer metrics.
3. **Production readiness:** incremental ingestion, caching, tracing (including LangSmith), quantization, benchmark script.
4. **Advanced:** pick from section 14.
