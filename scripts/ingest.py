"""Index files into Qdrant.

Usage:
    uv run python scripts/ingest.py                    # index documents.path (data/pdfs)
    uv run python scripts/ingest.py path/to/file.pdf some/folder
    uv run python scripts/ingest.py --force            # re-embed even unchanged files
    uv run python scripts/ingest.py --reset            # empty the collection (all tenants) first
    uv run python scripts/ingest.py --list             # only list indexed documents
    uv run python scripts/ingest.py --tenant other --registry other/registry.yaml other/

Documents belong to one tenant (documents.tenant_id, or --tenant); --list shows that tenant's
documents only, as its administrator. Unchanged files are skipped before they are even parsed,
and a change to a file's access rules alone is applied without re-embedding. The first
run downloads the bge-m3 model (about 2.3 GB) unless embedding.provider is "hashing", and
Docling's layout models (a few hundred MB) unless documents.parser is "pypdf".
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from hybrid_rag.access import Principal
from hybrid_rag.config import get_settings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.ingestion import IngestionError, IngestionPipeline
from hybrid_rag.store import DocumentIndex, DocumentInfo, StoreError, build_document_index
from hybrid_rag.tracing import flush_traces, tracing_is_enabled


def readers(d: DocumentInfo) -> str:
    """Who may read the document: grant names, and a count (never IDs) of named users."""
    if d.visibility == "tenant":
        return "whole tenant"
    grants = [*d.allowed_groups, *d.allowed_roles]
    if d.allowed_users:
        grants.append(f"{len(d.allowed_users)} user(s)")
    return ", ".join(grants) or "admins only"


def print_documents(index: DocumentIndex, tenant_id: str) -> None:
    admin = Principal.tenant_admin(tenant_id)
    documents = index.list_documents(admin)
    print(
        f"\n{len(documents)} document(s), {index.count_chunks(admin)} chunk(s) "
        f"in {index.collection!r} for tenant {tenant_id!r}:"
    )
    for d in documents:
        print(
            f"  {d.source:<42} {d.chunk_count:>4} chunks  {d.pii_status or '-':<9} "
            f"{d.classification or '-':<12} {readers(d):<38} access v{d.access_version}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="files or folders (default: documents.path)")
    parser.add_argument("--force", action="store_true", help="re-index unchanged files")
    parser.add_argument(
        "--reset", action="store_true", help="delete all indexed documents (all tenants) first"
    )
    parser.add_argument("--list", action="store_true", help="list indexed documents and exit")
    parser.add_argument("--tenant", help="tenant the documents belong to (documents.tenant_id)")
    parser.add_argument(
        "--registry", type=Path, help="policy registry for these files (documents.registry)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    settings = get_settings()
    tenant_id = args.tenant or settings.documents.tenant_id
    try:
        Principal.tenant_admin(tenant_id)  # validates the tenant ID
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    try:
        index = build_document_index(settings)
    except StoreError as exc:
        print(f"error: {exc}")
        return 1

    if args.list:
        print_documents(index, tenant_id)
        return 0
    if args.reset:
        index.reset()
        print(f"Collection {index.collection!r} emptied.")

    emb = settings.embedding
    print(f"Embeddings: {emb.provider} ({emb.model}) on {emb.device}")
    try:
        pipeline = IngestionPipeline.from_settings(
            settings, tenant_id=tenant_id, registry_path=args.registry
        )
    except IngestionError as exc:
        print(f"error: {exc}")
        return 1
    print(f"Parser: {pipeline.parser.name} | tokens: {pipeline.tokens.name}")
    print(f"Privacy scan: {pipeline.sanitizer.name} (policy {pipeline.sanitizer.policy})")
    print(f"Tenant: {pipeline.tenant_id}")
    if tracing_is_enabled():
        print("LangSmith tracing is on: one 'ingest' trace (counts and names only, no text)")
    started = time.perf_counter()
    paths = args.paths or [settings.documents.path]
    outcomes = ingest_paths(paths, index, pipeline, force=args.force)
    for o in outcomes:
        if o.ok and o.result:
            r = o.result
            extra = f", {r.removed} stale removed" if r.removed else ""
            if r.status == "access_updated":
                extra += f", now access v{r.access_version} (nothing re-embedded)"
            print(f"  {r.status:<14} {r.source:<42} {r.chunks:>4} chunks{extra}")
        else:
            label = "QUARANTINE" if o.quarantined else "FAILED"
            print(f"  {label:<14} {o.path.name:<42} {o.error}")
        for w in o.warnings:
            print(f"                 warning: {w}")
    print(
        f"Done in {time.perf_counter() - started:.1f}s "
        f"(texts embedded: {index.embeddings.encoder.encoded_count})"
    )

    flush_traces()
    print_documents(index, tenant_id)
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
