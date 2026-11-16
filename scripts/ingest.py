"""Index files into Qdrant.

Usage:
    uv run python scripts/ingest.py                    # index data/samples
    uv run python scripts/ingest.py path/to/file.pdf some/folder
    uv run python scripts/ingest.py --force            # re-embed even unchanged files
    uv run python scripts/ingest.py --reset            # empty the collection first
    uv run python scripts/ingest.py --list             # only list indexed documents

Unchanged files are skipped, so running it twice is cheap. The first run downloads the
bge-m3 model (about 2.3 GB) unless embedding.provider is "hashing".
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from hybrid_rag.config import get_settings
from hybrid_rag.indexing import ingest_paths
from hybrid_rag.store import DocumentIndex, StoreError, build_document_index


def print_documents(index: DocumentIndex) -> None:
    documents = index.list_documents()
    print(
        f"\n{len(documents)} document(s), {index.count_chunks()} chunk(s) in {index.collection!r}:"
    )
    for d in documents:
        print(f"  {d.source:<36} {d.file_type:<4} {d.chunk_count:>4} chunks  {d.ingested_at}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", default=["data/samples"], help="files or folders")
    parser.add_argument("--force", action="store_true", help="re-index unchanged files")
    parser.add_argument("--reset", action="store_true", help="delete all indexed documents first")
    parser.add_argument("--list", action="store_true", help="list indexed documents and exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="  [%(levelname)s] %(message)s")

    settings = get_settings()
    try:
        index = build_document_index(settings)
    except StoreError as exc:
        print(f"error: {exc}")
        return 1

    if args.list:
        print_documents(index)
        return 0
    if args.reset:
        index.reset()
        print(f"Collection {index.collection!r} emptied.")

    emb = settings.embedding
    print(f"Embeddings: {emb.provider} ({emb.model}) on {emb.device}")
    started = time.perf_counter()
    outcomes = ingest_paths(args.paths, index, settings.chunking, force=args.force)
    for o in outcomes:
        if o.ok and o.result:
            r = o.result
            extra = f", {r.removed} stale removed" if r.removed else ""
            print(f"  {r.status:<9} {r.source:<36} {r.chunks:>4} chunks{extra}")
        else:
            print(f"  FAILED    {o.path.name:<36} {o.error}")
        for w in o.warnings:
            print(f"            warning: {w}")
    print(
        f"Done in {time.perf_counter() - started:.1f}s "
        f"(texts embedded: {index.embeddings.encoder.encoded_count})"
    )

    print_documents(index)
    return 0 if all(o.ok for o in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
