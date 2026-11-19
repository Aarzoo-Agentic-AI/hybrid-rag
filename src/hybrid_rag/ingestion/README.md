# Ingestion

This package turns HR policy PDFs into LangChain `Document` chunks, ready to be embedded and stored. Chunks follow the policy's own structure: one chunk per section where it fits, with the heading path ("7 Annual Performance Bonus Plan > 7.3 Company Performance Factor"), the page range, and the policy's metadata from the registry. Sensitive data is removed before chunking (see the [privacy README](../privacy/README.md)). PDF is the only supported file type.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.ingestion import IngestionPipeline, citation

pipeline = IngestionPipeline.from_settings(get_settings())  # Docling, Presidio, registry
doc = pipeline.load("data/pdfs/Travel_Policy.pdf")
print(doc.title, doc.stats)  # Travel Policy {'pages': 5, 'sections': 16, 'chunks': 16, ...}
print(citation(doc.chunks[6].metadata))  # Travel Policy §5 Air travel (Travel_Policy.pdf, page 3)
```

For uploads saved under a temporary name, pass the original file name. It's used for citations, for the document ID and for the registry lookup:

```python
pipeline.load("/tmp/upload_8f3a.tmp", source="Travel_Policy.pdf")
```

`IngestionPipeline()` with no arguments (and the `load_document()` shortcut) uses the fast pypdf parser, the pattern-only sanitizer and no registry. The unit tests use this.

## How a PDF is processed

```
PDF ─► parse ──────────► registry ─► ① sanitize ─► sections ─────► chunks ──────► ② verify
       Docling: blocks   policy      redact or     heading paths   section-aware   re-check
       with pages,       metadata    quarantine    from numbering  (sizes in       each chunk
       cached by hash    per file                                  tokens)
```

| Step | File | What it does |
|---|---|---|
| Parse | [`parsers.py`](parsers.py) | Turns the PDF into **blocks** (heading, paragraph, list item, table), each with its page number |
| Policy metadata | [`registry.py`](registry.py) | Looks up the file in `data/pdfs/registry.yaml` |
| Sanitize | [`../privacy/`](../privacy/) | Runs on every paragraph, list item and table row, and on the policy name |
| Sections | [`sections.py`](sections.py) | Groups blocks under their headings and builds the heading path |
| Chunks | [`chunking.py`](chunking.py) | Section-aware chunking, headers, IDs and the metadata schema |
| Pipeline | [`pipeline.py`](pipeline.py) | `IngestionPipeline` (all of the above, the parse cache, `fingerprint()`) and `load_document()` |

## Parsers

| `documents.parser` | What it gives | Speed | Used for |
|---|---|---|---|
| `docling` (default) | Headings, paragraphs, list items and **tables with rows and column headers**. Running page headers and footers ("… \| Internal use", "Page 2") are dropped, because Docling classifies them as page furniture. | About 4 s per page on CPU, so results are **cached** in `.cache/parsed/`, keyed by file hash and parser version | Production |
| `pypdf` | Text lines. Headings are recognised only when **numbered** ("5. Air travel"). Lines repeated on half the pages and "Page N" lines are removed. Tables come out as plain lines. | Instant, no model | Unit tests, quick demos |

**Why Docling.** Both Docling and `pymupdf4llm` were compared on the 5 policies.
- **Quality:** both recovered headings and tables. Only Docling removed the running headers, and it gives the page of every element.
- **Licence:** Docling is MIT. `pymupdf4llm` builds on PyMuPDF, which is AGPL-3.0, and that would put the whole project under the AGPL.
- **Speed:** Docling is slower (OCR is off, because the policies are born-digital PDFs). The parse cache and the skip for unchanged files mean each file is parsed once.

The cache holds the extracted text of each PDF, before sanitization, so it's as sensitive as the PDFs themselves. It is git-ignored; in production, keep it in the same restricted storage as the originals.

## Sections

Heading levels come from the numbering:

| Heading | Level | Path |
|---|---|---|
| `6. Merit Increase Guidelines` | 1 | `6 Merit Increase Guidelines` |
| `6.2 Merit matrix` | 2 | `6 Merit Increase Guidelines > 6.2 Merit matrix` |
| `Special cases` (unnumbered, after `4.`) | 2 | `4 Performance Rating Scale > Special cases` |
| `Document Control` (before section 1) | 1 | `Document Control` |

- Everything before the first real heading (the cover: title, company name, policy number, effective date) becomes an **Overview** section.
- Sections whose title matches `chunking.skip_sections` aren't indexed, and neither are their sub-sections. The defaults are tables of contents ("Contents", "What's inside") and signature forms ("Acknowledgement").

## Chunking rules

Sizes are measured in **tokens of the embedding model's tokenizer** (`bge-m3`), not characters.

1. A section up to `max_tokens` (800) **stays whole**. On the 5 policies, every section fits, from 44 to 488 tokens.
2. A longer section is split near `target_tokens` (400) at **block boundaries**: paragraphs, list items and table rows, never inside one. Consecutive parts share up to `overlap_tokens` (60) of trailing blocks. There is **no overlap across sections**.
3. A split **table repeats its column headers** at the top of every part.
4. A block starting with "Unless", "Except", "However", "Provided that" or "Note" stays with the block before it, so **a rule keeps its exception**. A paragraph ending with ":" stays with what follows, so **a list keeps its introduction**.
5. Only a single block longer than `max_tokens` is split inside it, on sentences, with LangChain's `RecursiveCharacterTextSplitter` counting tokens.
6. In **definition sections** (`definition_sections`: "Definitions", "Glossary"), each row of a two-column table becomes its own chunk, `Compa-ratio: Base salary divided by the midpoint of the salary range`, with `chunk_type: definition` and `term`. The policies have 17 such terms.

## Contextual header

Each chunk starts with a short header, which is embedded with the text and tells the LLM where the passage comes from:

```
Policy: Travel Policy (HR-TRV-003, effective 2027-01-01, US)
Section: 5 Air travel

| Flight length | Cabin class | Notes |
|---|---|---|
| 6 hours or less | Economy | Lowest logical fare; avoid unnecessary connections. |
...
```

## Policy registry

[`data/pdfs/registry.yaml`](../../../data/pdfs/registry.yaml) holds the metadata for each file. It was transcribed from each policy's cover and Document Control table, and the policy owner maintains it. Ingestion never guesses these values from the text.

```yaml
defaults: {jurisdiction: US, language: en, is_current: true, applies_to: [employees, interns]}
policies:
  Travel_Policy.pdf:
    policy_name: Travel Policy
    policy_id: HR-TRV-003
    policy_category: travel
    effective_date: 2027-01-01
    owner_department: People (Chief People Officer) with Finance
    review_date: 2028-01-01
```

- Entries are validated, and a typo (`efective_date`) or an invalid value is an error, not silently ignored.
- A file that isn't listed loads with the defaults and its PDF title, plus a warning.
- The registry also supplies the policy name when the PDF's title is an image, as with the Expense Policy.
- Step 3b adds the access-control fields here.

## Chunk metadata

Flat and JSON-friendly, stored as the Qdrant payload. It never contains a local path.

| Key | Example | Notes |
|---|---|---|
| `doc_id` / `doc_hash` | `"a63a…"` / `"3d53…"` | UUID from the source name / SHA-256 of the file |
| `source`, `file_type`, `title` | `"Travel_Policy.pdf"`, `"pdf"`, `"Travel Policy"` | `title` is the policy name |
| `chunk_id`, `chunk_index`, `chunk_count` | | Deterministic UUID; position in the file |
| `chunk_type` | `"section"` / `"definition"` | |
| `section_id` | `"b41d…"` | The same for every part of a section, for small-to-big retrieval in step 4 |
| `section_number`, `section_title`, `section_path` | `"5"`, `"Air travel"`, `"5 Air travel"` | |
| `section_part`, `section_parts` | `1`, `1` | Part of a split section |
| `term` | `"Compa-ratio"` | Definition chunks only |
| `page`, `page_end` | `3`, `4` | A section can cross pages |
| `policy_name`, `policy_id`, `policy_category`, `version`, `effective_date`, `is_current`, `jurisdiction`, `applies_to`, `owner_department`, `review_date`, `language` | | From the registry |
| `content_hash` | | SHA-256 of `page_content` (after redaction) |
| `pii_scan` | `{"status": "clean", …}` | Privacy scan result (types and counts, never values) |
| `pipeline_hash` | `"6577…"` | Fingerprint of the parser, tokenizer, chunking settings, privacy policy and this file's registry entry |

`citation(metadata)` gives `Travel Policy §5 Air travel (Travel_Policy.pdf, page 3)`. An unnumbered sub-section also names its numbered parent: `HR Policy Handbook §3 Anti-Harassment and Anti-Discrimination > Prohibited conduct includes (…)`.

**When is a document processed again?** Whenever its `pipeline_hash` changes: a new parser version, other chunking settings, other privacy rules, or an edited registry entry. Otherwise an unchanged file is skipped before it's even parsed. Re-chunking reads the parse cache, so it doesn't run Docling again.

## Configuration

```yaml
documents:
  path: data/pdfs
  registry: data/pdfs/registry.yaml
  parser: docling              # or pypdf
  cache_dir: .cache/parsed
chunking:
  target_tokens: 400
  max_tokens: 800
  overlap_tokens: 60
  tokenizer: BAAI/bge-m3       # or "approx" (4 characters per token)
  contextual_headers: true
  skip_sections: ['^contents$', "^what's inside$", 'acknowledge?ment']
  definition_sections: ['definitions?', 'glossary']
```

## Exceptions

All inherit from `IngestionError`.

| Exception | When |
|---|---|
| `UnsupportedFileTypeError` | The extension isn't `.pdf` (case-insensitive) |
| `EmptyDocumentError` | No page has extractable text (scanned PDFs need OCR, which isn't supported) |
| `RegistryError` | The registry is missing, isn't valid YAML, or has an invalid entry |
| `IngestionError` | A damaged or password-protected PDF |
| `SensitiveDataError` | High-risk data found: the document is quarantined (from `hybrid_rag.privacy`) |

## Tests

| File | Covers |
|---|---|
| `tests/test_ingestion.py` | The pypdf parser (headings, wrapped lines, running headers removed), section levels and paths, skipped sections, every chunking rule, metadata and headers, IDs, the registry (validation, defaults, hash), table sanitization, the parse cache, the fingerprint |
| `tests/test_hr_policies.py` | The full pipeline on the 5 HR PDFs using **Docling's saved output** (`tests/fixtures/parsed/`), so no models are needed: registry metadata on every chunk, tables intact, nested paths, 17 definitions, boilerplate left out, size limit, page ranges, a clean privacy scan |
| `tests/test_ingestion_live.py` (`-m integration`) | Real Docling output matches the fixtures; chunk sizes in real `bge-m3` tokens |

The fixtures come from `uv run python scripts/make_parse_fixtures.py`. If a PDF changes or `DoclingParser.VERSION` is bumped, the tests fail until it's run again.
