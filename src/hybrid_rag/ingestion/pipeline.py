"""The ingestion pipeline: PDF -> parse -> policy metadata -> sanitize -> sections -> chunks.

```
PDF ─► parse ──────► registry ─► repair ─► sanitize ─► sections ─► chunks ─► verify
       (Docling,     policy +    parser    ① redact /   heading     section-   ② re-check
        cached)      access      errors    quarantine   paths       aware      each chunk
```

A pipeline belongs to one tenant: document IDs are scoped to it, and every chunk carries its
``tenant_id`` and the document's access rules from the registry.

``IngestionPipeline.from_settings(settings)`` builds the production pipeline (Docling, the
Presidio sanitizer, the policy registry, the bge-m3 tokenizer). ``IngestionPipeline()``
with no arguments, and ``load_document()``, use the fast pypdf parser, the pattern
sanitizer and an empty registry: that's what the unit tests use.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

from hybrid_rag.access import AccessPolicy, validate_tenant_id
from hybrid_rag.config import ChunkingSettings, Settings
from hybrid_rag.ingestion.blocks import Block, ParsedDocument
from hybrid_rag.ingestion.chunking import SectionChunker, finalize_chunks, make_doc_id
from hybrid_rag.ingestion.cleanup import repair_blocks
from hybrid_rag.ingestion.parsers import DoclingParser, Parser, PypdfParser
from hybrid_rag.ingestion.registry import PolicyRegistry, RegistryError
from hybrid_rag.ingestion.sections import build_sections
from hybrid_rag.ingestion.text import sha256_bytes, sha256_text
from hybrid_rag.ingestion.tokens import get_token_counter
from hybrid_rag.ingestion.types import FileType, IngestedDocument
from hybrid_rag.tracing import traceable

if TYPE_CHECKING:
    from hybrid_rag.privacy import Sanitizer, ScanReport

logger = logging.getLogger(__name__)

# Bump when sectioning or chunking logic changes, so every document is re-processed.
# 3: repair_blocks (contents spill, continuations, registry corrections).
PIPELINE_VERSION = 3

_ROW = " | "  # table rows are sanitized as one line, then split back into cells


class IngestionPipeline:
    def __init__(
        self,
        chunking: ChunkingSettings | None = None,
        *,
        parser: Parser | None = None,
        sanitizer: Sanitizer | None = None,
        registry: PolicyRegistry | None = None,
        cache_dir: Path | None = None,
        tenant_id: str = "default",
    ) -> None:
        from hybrid_rag.privacy import default_sanitizer

        self.tenant_id = validate_tenant_id(tenant_id)
        self.chunking = chunking or ChunkingSettings()
        self.parser = parser or PypdfParser()
        self.sanitizer = sanitizer or default_sanitizer()
        self.registry = registry or PolicyRegistry.empty()
        self.cache_dir = cache_dir
        self.tokens = get_token_counter(self.chunking.tokenizer)
        self.chunker = SectionChunker(self.chunking, self.tokens)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        tenant_id: str | None = None,
        registry_path: Path | None = None,
    ) -> IngestionPipeline:
        """The production pipeline. ``tenant_id`` and ``registry_path`` override
        ``documents.tenant_id`` and ``documents.registry`` (``scripts/ingest.py --tenant``)."""
        from hybrid_rag.privacy import build_sanitizer

        docs = settings.documents
        registry_file = registry_path or docs.registry
        if registry_file is None:
            registry = PolicyRegistry.empty()
        elif registry_file.exists():
            registry = PolicyRegistry.load(registry_file)
        else:
            raise RegistryError(
                f"Policy registry {registry_file} not found. Create it, or set "
                "documents.registry to null in config.yaml to use defaults for every file."
            )
        return cls(
            settings.chunking,
            parser=DoclingParser() if docs.parser == "docling" else PypdfParser(),
            sanitizer=build_sanitizer(settings.privacy),
            registry=registry,
            cache_dir=docs.cache_dir,
            tenant_id=tenant_id or docs.tenant_id,
        )

    def doc_id(self, source: str) -> str:
        return make_doc_id(self.tenant_id, source)

    def access_for(self, source: str) -> tuple[AccessPolicy, list[str]]:
        """Who may read ``source``, from the registry, plus warnings. Cheap: no file access,
        so the store can apply a permission change without loading the document."""
        return self.registry.access(source)

    def fingerprint(self, source: str) -> str:
        """Everything besides the file that shapes its chunks. Stored as ``pipeline_hash``;
        when it changes, an unchanged file is processed again. Access rules are not part of
        it: they are compared separately (``access_hash``) and updated without re-embedding."""
        data = {
            "v": PIPELINE_VERSION,
            "parser": self.parser.name,
            "tokens": self.tokens.name,
            "chunking": self.chunking.model_dump(),
            "privacy": self.sanitizer.policy,
            "registry": self.registry.entry_hash(source),
        }
        return sha256_text(json.dumps(data, sort_keys=True))[:12]

    # -------------------------------------------------------------- load

    @traceable(
        name="load_document",
        run_type="parser",
        process_inputs=lambda i: {"source": i["source"] or Path(i["path"]).name},
        process_outputs=lambda d: (
            {
                "title": d.title,
                **d.stats,
                "warnings": d.warnings,
                "pii": d.pii_scan["status"],
                "redactions": d.pii_scan["redactions"],
                "classification": d.access.classification,
                "visibility": d.access.visibility,
            }
            if d
            else {}  # the call raised; LangSmith records the error
        ),
    )
    def load(self, path: str | Path, *, source: str | None = None) -> IngestedDocument:
        """Load one PDF and turn it into chunks. ``source`` is the name used in citations and
        for the document ID (pass the upload name when ``path`` is a temporary file).

        Raises ``SensitiveDataError`` (an ``IngestionError``) if the document is quarantined.
        """
        path = Path(path)
        file_type = FileType.from_path(source or path)
        source = source or path.name
        doc_hash = sha256_bytes(path.read_bytes())

        parsed = self._parse(path, doc_hash, source=source)
        policy, registry_warnings = self.registry.lookup(source, title=parsed.title)
        access, access_warnings = self.access_for(source)
        blocks, repair_warnings = repair_blocks(parsed.blocks, self.registry.corrections(source))

        # ① Sanitize before chunking: every paragraph, list item and table row, and the name.
        blocks, title, report = self._sanitize(blocks, policy.policy_name, source)
        policy = policy.model_copy(update={"policy_name": title})

        sections = build_sections(blocks, self.chunking.skip_sections)
        drafts = self.chunker.chunk(sections)
        doc_id = self.doc_id(source)
        fingerprint = self.fingerprint(source)
        chunks = finalize_chunks(
            drafts,
            doc_id=doc_id,
            doc_hash=doc_hash,
            source=source,
            title=title,
            file_type=file_type,
            policy=policy.metadata(),
            contextual_headers=self.chunking.contextual_headers,
            pii_scan=report.as_metadata(),
            pipeline_hash=fingerprint,
            access=access.metadata(self.tenant_id),
        )
        self.sanitizer.verify(chunks, source=source)  # ② re-check the finished chunks

        warnings = [*parsed.warnings, *registry_warnings, *repair_warnings, *access_warnings]
        for warning in warnings:
            logger.warning("%s: %s", source, warning)
        stats = {
            "pages": parsed.pages,
            "pages_with_text": parsed.pages_with_text,
            "sections": sum(1 for s in sections if s.blocks and not s.skipped),
            "skipped_sections": sum(1 for s in sections if s.skipped),
            "definitions": sum(1 for d in drafts if d.chunk_type == "definition"),
            "chunks": len(chunks),
        }
        logger.info("Loaded %s: %d section(s), %d chunk(s)", source, stats["sections"], len(chunks))
        return IngestedDocument(
            doc_id=doc_id,
            source=source,
            file_type=file_type,
            doc_hash=doc_hash,
            title=title,
            chunks=chunks,
            warnings=warnings,
            stats=stats,
            pii_scan=report.as_metadata(),
            pipeline_hash=fingerprint,
            tenant_id=self.tenant_id,
            access=access,
        )

    # ------------------------------------------------------------- steps

    @traceable(
        name="parse",
        process_inputs=lambda i: {"source": i["source"]},
        process_outputs=lambda p: (
            {
                "pages": p.pages,
                "blocks": len(p.blocks),
                "tables": sum(b.kind == "table" for b in p.blocks),
            }
            if p
            else {}
        ),
    )
    def _parse(self, path: Path, doc_hash: str, *, source: str) -> ParsedDocument:
        """Parse, or read the cached result for this exact file and parser version."""
        cache_file = None
        if self.cache_dir is not None:
            cache_file = self.cache_dir / f"{self.parser.name}-{doc_hash}.json"
        if cache_file is not None and cache_file.exists():
            try:
                data = json.loads(cache_file.read_text(encoding="utf-8"))
                return ParsedDocument.from_dict(data)
            except (ValueError, KeyError, TypeError):
                logger.warning("Ignoring unreadable parse cache %s", cache_file.name)
        parsed = self.parser.parse(path)
        if cache_file is not None:
            _write_atomically(cache_file, json.dumps(parsed.to_dict()))
        return parsed

    def _sanitize(
        self, blocks: list[Block], title: str, source: str
    ) -> tuple[list[Block], str, ScanReport]:
        units: list[Document] = []
        for i, block in enumerate(blocks):
            if block.kind == "table":
                lines = ([block.header] if block.header else []) + block.rows
                for r, cells in enumerate(lines):
                    units.append(_unit(_ROW.join(cells), block.page, i, r))
            else:
                units.append(_unit(block.text, block.page, i, None))
        clean, title, report = self.sanitizer.sanitize_units(units, title=title, source=source)

        out = [Block(b.kind, b.text, b.page, b.header, []) for b in blocks]
        for doc in clean:
            index, row = doc.metadata["block"], doc.metadata["row"]
            block = out[index]
            if row is None:
                block.text = doc.page_content
            elif blocks[index].header and row == 0:
                block.header = doc.page_content.split(_ROW)
            else:
                block.rows.append(doc.page_content.split(_ROW))
        return out, title, report


def _unit(text: str, page: int, block: int, row: int | None) -> Document:
    return Document(page_content=text, metadata={"page": page, "block": block, "row": row})


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_document(
    path: str | Path,
    chunking: ChunkingSettings | None = None,
    *,
    source: str | None = None,
    sanitizer: Sanitizer | None = None,
    parser: Parser | None = None,
    registry: PolicyRegistry | None = None,
) -> IngestedDocument:
    """Convenience wrapper: ``IngestionPipeline(...).load(path)``. Defaults to the pypdf
    parser and the pattern sanitizer; use ``IngestionPipeline.from_settings()`` in production."""
    pipeline = IngestionPipeline(chunking, parser=parser, sanitizer=sanitizer, registry=registry)
    return pipeline.load(path, source=source)


def find_supported_files(directory: str | Path) -> list[Path]:
    """All supported (PDF) files under ``directory`` (recursively), sorted by path."""
    suffixes = {f".{t.value}" for t in FileType}
    return sorted(
        p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in suffixes
    )
