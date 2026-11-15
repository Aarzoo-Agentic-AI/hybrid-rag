"""A tiny PDF writer for tests and sample data (text only, Helvetica, US Letter).

It writes just enough of the PDF format for text extraction to work, so tests don't need
binary fixtures or an extra dependency.
"""

from __future__ import annotations

from pathlib import Path

_LINE_HEIGHT = 14
_TOP = 740
_MAX_LINES = 50


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[str]], title: str | None = None) -> Path:
    """Write ``pages`` (each a list of text lines) to ``path``. An empty list makes a blank page."""
    for lines in pages:
        if len(lines) > _MAX_LINES:
            raise ValueError(f"at most {_MAX_LINES} lines fit on a page")

    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    info_id = 4 + 2 * n_pages

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for i, lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode()
        )
        if lines:
            ops = [f"BT /F1 11 Tf {_LINE_HEIGHT} TL 72 {_TOP} Td"]
            ops += [f"({_escape(line)}) Tj T*" for line in lines]
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    objects.append(f"<< /Title ({_escape(title or '')}) /Producer (pdf_fixture) >>".encode())

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()

    path.write_bytes(bytes(out))
    return path
