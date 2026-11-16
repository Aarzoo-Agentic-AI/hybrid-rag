# Ingestion

This package turns PDF files into LangChain `Document` chunks, ready to be embedded and stored. Sensitive data is removed before chunking (see the [privacy README](../privacy/README.md)). Each chunk carries a contextual header, a stable ID, a content hash and its page number for citations. PDF is the only supported file type.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.ingestion import citation, find_supported_files, load_document

settings = get_settings()
for path in find_supported_files("data/samples"):
    doc = load_document(path, settings.chunking)
    print(doc.source, doc.stats, doc.warnings)
    print(citation(doc.chunks[0].metadata))  # e.g. "lumen_employee_handbook.pdf, page 1"
```

For uploads saved under a temporary name, pass the original file name; it's used for citations and for the document ID:

```python
load_document("/tmp/upload_8f3a.tmp", settings.chunking, source="report.pdf")
```

## Files

| File | Contents |
|---|---|
| [`pipeline.py`](pipeline.py) | `load_document()` and `find_supported_files()` |
| [`loaders.py`](loaders.py) | The LangChain loader `PdfPageLoader` and `load_pdf()`, which turns its pages into page-tagged units |
| [`chunking.py`](chunking.py) | Splitting, contextual headers, IDs, hashes, and `location_label()` / `citation()` |
| [`text.py`](text.py) | Whitespace clean-up and hashing |
| [`types.py`](types.py) | `FileType`, `IngestedDocument` and the exceptions |

## How a PDF is processed

```
PDF ──► PdfPageLoader ──► units ──────► sanitize ──────► chunks ─────────► final chunks ──► verify
        (LangChain        (one per      (redact or       (Recursive-        (header + IDs +   (re-check
         BaseLoader,       page,         quarantine;      Character-         hashes +          each chunk)
         pypdf)            cleaned)      privacy/)        TextSplitter)      metadata)
```

`load_document(path, chunking, sanitizer=...)` takes the sanitizer built from `settings.privacy`. Without one, it uses the pattern-only `default_sanitizer()`, so no document is ever chunked unscanned. A quarantined document raises `SensitiveDataError`, which is an `IngestionError`.

- One `Document` per page, and one unit per page that has text.
- Pages are split with overlap, and **chunks never cross pages**, so every chunk has an exact page number.
- Words hyphenated across a line break are re-joined (`hy-\nbrid` becomes `hybrid`).
- The PDF's own title is used if it has one; otherwise the file name.
- Pages without text, such as scanned pages, are skipped with a warning. If no page has text, `EmptyDocumentError` is raised, because OCR isn't supported.
- Damaged or password-protected files raise `IngestionError`.

Section-aware chunking across page breaks, with heading paths, is planned for step 3a (see [PLAN.md](../../../PLAN.md)).

### Why the loader is our own

`langchain-community`, which provided `PyPDFLoader`, was sunset in May 2026 and no longer receives fixes. `PdfPageLoader` subclasses `langchain_core.document_loaders.BaseLoader`, so it works like any LangChain loader (`load()`, `lazy_load()`), and uses `pypdf` directly.

## Contextual headers

With `chunking.contextual_headers: true` (the default), each chunk starts with a short header naming its document and page:

```
Document: Lumen Robotics Employee Handbook (lumen_employee_handbook.pdf)
Location: page 2

4. Annual leave
Full-time employees receive 25 days of paid annual leave per year, ...
```

The header is embedded together with the text, which helps retrieval for questions that mention a document. It also tells the LLM where each passage comes from. `chunk_size` applies to the text without the header.

## Chunk metadata

Metadata is flat and JSON-friendly, so it can be stored directly as a Qdrant payload. It never contains the local file path.

| Key | Example | Notes |
|---|---|---|
| `doc_id` | `"5c1f…"` | UUID, the same for every chunk of a file. It comes from the source name, so re-ingesting a file with the same name replaces it |
| `doc_hash` | `"9a0b…"` | SHA-256 of the file bytes, used to skip files that haven't changed |
| `source` | `"lumen_employee_handbook.pdf"` | File name only |
| `title` | `"Lumen Robotics Employee Handbook"` | PDF title, otherwise the file name without its extension |
| `file_type` | `"pdf"` | Always `pdf` |
| `chunk_id` | `"0e7d…"` | UUID from `doc_id` and `chunk_index`; also set as `Document.id` |
| `chunk_index` / `chunk_count` | `0` / `5` | Position within the file |
| `content_hash` | `"77c2…"` | SHA-256 of `page_content` (after redaction), including the header |
| `pii_scan` | `{"status": "redacted", …}` | Privacy scan: status, detector, policy, redaction counts by type (never values) |
| `pipeline_hash` | `"4be1…"` | Hash of the chunking settings and privacy policy the chunks were made with. When it changes, unchanged files are processed again |
| `page`, `page_label` | `3` | 1-based page number; `page_label` only when the printed label differs, e.g. `iv` |

The IDs are deterministic (UUID5 with a fixed namespace), so the same file always produces the same IDs on any machine. Step 3 relies on this for incremental indexing. HR policy metadata and authorization metadata are added in steps 3a and 3b.

## Configuration

From the `chunking:` section of [`config.yaml`](../../../config.yaml):

```yaml
chunking:
  chunk_size: 800            # characters, without the header
  chunk_overlap: 120
  contextual_headers: true
```

The splitter tries to break at paragraphs first, then lines, then sentences, then words. It keeps the full stop at the end of a sentence rather than at the start of the next chunk.

## Sample data

[`data/samples/`](../../../data/samples/) holds an employee handbook for a fictional company, Lumen Robotics. It is generated from text by `uv run python scripts/make_sample_pdf.py`, using the small PDF writer in [`tests/pdf_fixture.py`](../../../tests/pdf_fixture.py). The tests use the same writer to build PDFs on the fly, so they need no binary fixtures. Step 3a replaces it with a set of HR policies.

## Exceptions

All inherit from `IngestionError`.

| Exception | When |
|---|---|
| `UnsupportedFileTypeError` | The extension isn't `.pdf` (case-insensitive) |
| `EmptyDocumentError` | No page has extractable text |
| `IngestionError` | A damaged or password-protected PDF |
| `SensitiveDataError` | High-risk data (for example an SSN or card number) found: the document is quarantined. Defined in `hybrid_rag.privacy` |

## Tests

`uv run pytest tests/test_ingestion.py` covers page numbers, splitting within pages, size and overlap, skipped pages, headers, ID stability, the metadata schema, unsupported file types, the error cases and the sample file.
