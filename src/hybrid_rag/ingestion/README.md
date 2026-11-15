# Ingestion

This package turns PDF, TXT and CSV files into LangChain `Document` chunks, ready to be embedded and stored. Each chunk carries a contextual header, a stable ID, a content hash and its exact location in the file (page, lines or rows) for citations.

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
| [`loaders.py`](loaders.py) | The LangChain loaders (`PdfPageLoader`, `TextFileLoader`, `CsvRowLoader`) and `load_pdf()` / `load_txt()` / `load_csv()`, which turn their output into location-tagged units |
| [`chunking.py`](chunking.py) | Splitting, contextual headers, IDs, hashes, and `location_label()` / `citation()` |
| [`text.py`](text.py) | Encoding detection, whitespace clean-up and hashing |
| [`types.py`](types.py) | `FileType`, `IngestedDocument` and the exceptions |

## How a file is processed

```
file ──► loader ──► units ──────────► chunks ─────────────► final chunks
         (LangChain   (cleaned text      (split with            (header + IDs +
          BaseLoader)  + location)        RecursiveCharacter-    hashes + metadata)
                                          TextSplitter; CSV
                                          units aren't split)
```

| | Loader output | Unit | Splitting | Location in metadata |
|---|---|---|---|---|
| **PDF** | One `Document` per page (`pypdf`) | One per page that has text | Split within the page; **chunks never cross pages** | `page` (1-based), plus `page_label` if it differs, e.g. `iv` |
| **TXT** | One `Document` for the file | The whole file | Split with overlap | `line_start` / `line_end` (1-based) |
| **CSV** | One `Document` per row, as `column: value` lines | `csv_rows_per_chunk` rows | Not split: a row is never cut in half | `row_start` / `row_end` (1-based data rows) and `columns` |

### Why the loaders are our own

`langchain-community`, which provided `PyPDFLoader`, `TextLoader` and `CSVLoader`, was sunset in May 2026 and no longer receives fixes. The loaders here subclass `langchain_core.document_loaders.BaseLoader`, so they work like any LangChain loader (`load()`, `lazy_load()`), and use `pypdf` and Python's `csv` module directly.

### Details for each file type

- **PDF**
  - Words hyphenated across a line break are re-joined (`hy-\nbrid` becomes `hybrid`).
  - The PDF's own title is used if it has one; otherwise the file name.
  - Pages without text, such as scanned pages, are skipped with a warning. If no page has text, `EmptyDocumentError` is raised, because OCR isn't supported.
  - Damaged or password-protected files raise `IngestionError`.
- **TXT**
  - Blank lines are kept, so the line numbers match the original file.
- **CSV**
  - Empty cells are left out, and rows with no values at all are skipped with a warning.
  - Quoted values containing commas or line breaks are handled.
  - Values in extra columns without a header are kept as `column N: value`.
  - With `csv_rows_per_chunk` above 1, each row in a chunk is labelled `Row N`, so the LLM can cite individual rows.
- **Encoding (TXT and CSV)**
  - UTF-8 is tried first. The `utf-8-sig` variant also removes the byte-order mark that Excel adds, which would otherwise end up in the first column name.
  - If UTF-8 fails, `charset-normalizer` guesses the encoding, with cp1252 as the last resort.

## Contextual headers

With `chunking.contextual_headers: true` (the default), each chunk starts with a short header naming its document and location:

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
| `file_type` | `"pdf"` | `pdf`, `txt` or `csv` |
| `chunk_id` | `"0e7d…"` | UUID from `doc_id` and `chunk_index`; also set as `Document.id` |
| `chunk_index` / `chunk_count` | `0` / `5` | Position within the file |
| `content_hash` | `"77c2…"` | SHA-256 of `page_content`, including the header |
| `page`, `page_label` | `3` | PDF only |
| `line_start` / `line_end` | `19` / `39` | TXT only |
| `row_start` / `row_end`, `columns` | `4` / `4`, `["sku", …]` | CSV only |

The IDs are deterministic (UUID5 with a fixed namespace), so the same file always produces the same IDs on any machine. Step 3 relies on this for incremental indexing.

## Configuration

From the `chunking:` section of [`config.yaml`](../../../config.yaml):

```yaml
chunking:
  chunk_size: 800            # characters, without the header
  chunk_overlap: 120
  csv_rows_per_chunk: 1
  contextual_headers: true
```

The splitter tries to break at paragraphs first, then lines, then sentences, then words. It keeps the full stop at the end of a sentence rather than at the start of the next chunk.

## Sample data

[`data/samples/`](../../../data/samples/) holds three files about a fictional company, Lumen Robotics: an employee handbook (PDF), firmware release notes with error codes (TXT) and a product catalogue (CSV). The error codes and SKUs (such as `E-4012` and `LR-200-G2`) are exact identifiers, where keyword (sparse) search is expected to do better than dense search, which the step 7 evaluation will check.

The PDF is generated from text by `uv run python scripts/make_sample_pdf.py`, using the small PDF writer in [`tests/pdf_fixture.py`](../../../tests/pdf_fixture.py). The tests use the same writer to build PDFs on the fly, so they need no binary fixtures.

## Exceptions

All inherit from `IngestionError`.

| Exception | When |
|---|---|
| `UnsupportedFileTypeError` | The extension isn't `.pdf`, `.txt` or `.csv` (case-insensitive) |
| `EmptyDocumentError` | No extractable text, no header row or no data rows |
| `IngestionError` | A damaged or password-protected PDF, or an unreadable CSV |

## Tests

`uv run pytest tests/test_ingestion.py` covers each file type, the line and row numbers, encodings, headers, ID stability, the metadata schema, the error cases and the sample files.
