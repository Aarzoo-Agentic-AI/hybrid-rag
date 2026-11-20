# Privacy: sensitive-data sanitization

This package keeps personal and high-risk data out of the index. Every PDF is scanned **before it is chunked**, and every chunk is checked again **before it is embedded**. Sensitive values therefore never reach the vectors, the Qdrant payload, LangSmith traces or the logs.

```python
from hybrid_rag.config import get_settings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionPipeline
from hybrid_rag.privacy import build_sanitizer
from hybrid_rag.store import build_document_index

settings = get_settings()
sanitizer = build_sanitizer(settings.privacy)  # build once; Presidio loads on first use
pipeline = IngestionPipeline(settings.chunking, sanitizer=sanitizer)
outcomes = ingest_paths(["data/pdfs"], build_document_index(settings), pipeline)
[o.path.name for o in outcomes if o.quarantined]  # files that were not indexed
```

`scripts/ingest.py` does this for you through `IngestionPipeline.from_settings()`. Without a sanitizer, `IngestionPipeline()` and `load_document()` use `default_sanitizer()`, which runs the patterns only. So nothing is ever chunked without a scan by accident.

Pass ① runs on every **block** of the parsed document: each paragraph, list item and table row, plus the policy name. Table rows are scanned as a single line and split back into cells, so redaction never breaks a table. The lowercase-word list for the name filter is built from the whole document.

## Where it runs

```
PDF ─► extract ─► clean ─► ① sanitize (whole pages + title) ─► chunk ─► ② verify (each chunk) ─► embed ─► Qdrant
                                 │
                                 └─► high-risk data ─► SensitiveDataError: quarantined, not indexed
```

| Pass | Why there |
|---|---|
| **① Before chunking** | A chunk boundary can split an entity (`123-45-` / `6789`) so that neither half matches, and names are recognised better in full sentences |
| **② After chunking** | A safety net. It re-checks the finished chunks, including the contextual header, with the pattern detector, and stops if anything slipped through. It should never fire. |
| **Before embedding** | Embeddings can be partly inverted back to text, so vectors made from personal data are still personal data |

## What happens to each kind of data

| Setting | Default entities | Action |
|---|---|---|
| `redact` | `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `EMPLOYEE_ID` | Replaced with typed placeholders, `<PERSON>`, `<EMAIL>`, `<PHONE>` and `<EMPLOYEE_ID>`, so the text keeps its meaning for search and for the LLM |
| `quarantine` | `US_SSN`, `CREDIT_CARD`, `IBAN_CODE`, `US_BANK_NUMBER`, `US_PASSPORT`, `US_ITIN` | The **whole document is refused**. A policy containing these is almost certainly the wrong file, and it needs a human review, not a silent fix. An earlier indexed version, if any, is left as it was. |
| `allowlist` / `allowlist_patterns` | `hr@dummymocktest.example` | Kept. These are shared business contacts that answers should include. |

Policy content is never touched: amounts, limits, dates, places and rules ("$300 hotel cap", "25 days", "New York", "January 1, 2027") all stay. Salary *bands* and investigation procedures are legitimate but sensitive content. Access control handles those (FEATURE.md section 11), not redaction.

Redaction is **irreversible by design**: placeholders, no encryption and no lookup table. A policy assistant never needs the original values back, and with nothing reversible stored, nothing can leak.

## Detectors

| `privacy.detector` | Uses | Finds |
|---|---|---|
| `presidio` (default) | The pattern detector, plus Microsoft Presidio with the spaCy model `en_core_web_sm` (13 MB) | Everything below, plus **person names** and more ID formats (`US_BANK_NUMBER`, `US_PASSPORT`, `US_ITIN`, which need context words such as "account number") |
| `patterns` | Regular expressions only, with no model | Emails, North American phone numbers, SSNs (valid ranges only), cards (Luhn check), IBANs (mod-97 check), employee IDs (`employee_id_pattern`) |

**Person names need a filter.** On the sample policies, spaCy's guesses were all wrong: "Roth", "Bereavement", "Tier 2", "Salary", "Policy Handbook". `plausible_name()` accepts a `PERSON` only if it has at least two capitalised words, no digits, and no word that also appears in lower case in the same document. Words inside email addresses and URLs don't count. Measured on the 5 HR PDFs in `data/pdfs/`:

| spaCy model | Size | False positives without the filter | With the filter | Planted names found |
|---|---|---|---|---|
| `en_core_web_sm` | 13 MB | 31 | **0** | 5 / 5 |
| `en_core_web_md` | 40 MB | 11 | 0 | 5 / 5 |
| `en_core_web_lg` | 560 MB | 8 | 0 | 5 / 5 |

With the filter, the small model is enough. The trade-off is that a single-word name ("ask Maria") or a name made of common words ("Grace Hope") isn't redacted. Pattern-based data (emails, phones, IDs) is still caught in those sentences.

## What is recorded

Found values are **never** logged, traced, stored or included in errors. Only entity types, counts and page numbers are.

Every chunk carries a `pii_scan` entry in its metadata, which also goes into the Qdrant payload:

```json
{"status": "redacted", "detector": "patterns+presidio:en_core_web_sm",
 "policy": "9113372759a1", "redactions": {"EMAIL_ADDRESS": 1, "PERSON": 1}}
```

- `status`: `clean`, `redacted` or `not_scanned` (only when `privacy.enabled: false`). Qdrant has a payload index on it, so you can list redacted documents.
- `policy`: a hash of the privacy settings and `SCANNER_VERSION`. It is part of each chunk's `pipeline_hash`, so **changing the privacy settings re-scans and re-indexes every document** on the next ingest, even unchanged files.
- A quarantine shows up as `IngestOutcome.quarantined`, as `QUARANTINE` in `scripts/ingest.py`, and as status `quarantined` in the LangSmith `ingest_file` run. The message names the entity types and pages, never the values: `payroll.pdf quarantined, not indexed: high-risk data found (CREDIT_CARD x1, US_SSN x1) on page(s) 3.`

The `sanitize` step appears in LangSmith under `load_document`, with its counts.

## Configuration

```yaml
privacy:
  enabled: true
  detector: presidio
  spacy_model: en_core_web_sm
  score_threshold: 0.5
  redact: [PERSON, EMAIL_ADDRESS, PHONE_NUMBER, EMPLOYEE_ID]
  quarantine: [US_SSN, CREDIT_CARD, IBAN_CODE, US_BANK_NUMBER, US_PASSPORT, US_ITIN]
  employee_id_pattern: '\bEMP-?\d{5,7}\b'
  allowlist: [hr@dummymocktest.example]
  allowlist_patterns: []        # e.g. '^(hr|payroll|benefits)@yourcompany\.com$'
```

An entity can't be both redacted and quarantined; the settings refuse that.

## Performance

- **Start-up:** importing Presidio takes about 15 s the first time in a process, because it imports `torch` and `transformers`. `bge-m3` imports them anyway, so a normal ingestion run pays this once.
- **Scanning:** about 0.15 s per page after start-up.
- **Unchanged files:** they're skipped before they're loaded, so a re-run with nothing to do takes under a second.

## Files

| File | Contents |
|---|---|
| [`detectors.py`](detectors.py) | `PatternDetector`, `PresidioDetector`, `plausible_name()`, `lowercase_vocabulary()` |
| [`sanitizer.py`](sanitizer.py) | `Sanitizer` (both passes), `ScanReport`, `SensitiveDataError`, `build_sanitizer()`, `default_sanitizer()`, `disabled_sanitizer()` |

## Not covered yet

- **Questions and answers:** redacting personal data in user questions, and checking LLM answers, comes in step 5.
- **LangSmith anonymizer and audit-log counts:** these come in step 6.
- **Non-US identifiers:** national IDs from other countries (UK NI number and so on) need extra recognisers.
- **Scanned PDFs:** these aren't supported, because there's no OCR, so their text isn't scanned either.

## Tests

`tests/test_privacy.py` covers:
- each pattern, and policy content that must not match (amounts, dates, policy numbers, `1-800-XXX-XXXX`, invalid card, SSN and IBAN numbers);
- the name filter, placeholders and counts, the allowlist and allowlist patterns, and title redaction;
- quarantine without revealing values, the verification pass, the policy hash, and re-indexing when the settings change;
- that a quarantined file never reaches Qdrant;
- that all 5 HR PDFs come out clean.

`tests/test_tracing.py` checks that traces carry counts only.
