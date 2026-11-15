"""Text helpers: encoding detection, clean-up and hashing."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

_FALLBACK_ENCODING = "cp1252"  # common for CSV/TXT files saved by Windows tools


def detect_encoding(path: Path) -> str:
    """Return an encoding that can decode ``path``.

    UTF-8 is tried first (``utf-8-sig`` also strips the byte-order mark Excel adds).
    Otherwise ``charset-normalizer`` guesses, with cp1252 as the last resort.
    """
    data = path.read_bytes()
    try:
        data.decode("utf-8-sig")
        return "utf-8-sig"
    except UnicodeDecodeError:
        pass
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None and best.encoding:
            return best.encoding
    except ImportError:  # pragma: no cover - charset-normalizer ships with our dependencies
        pass
    return _FALLBACK_ENCODING


_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile("[ \t\f\v\u00a0]+")  # also the no-break space (U+00A0)
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(
    text: str, *, join_hyphenated: bool = False, collapse_blank_lines: bool = True
) -> str:
    """Normalise whitespace without changing the wording.

    ``join_hyphenated`` re-joins words split across lines ("hy-\\nbrid" -> "hybrid"), which
    is common in PDF text. ``collapse_blank_lines=False`` keeps line numbers intact.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if join_hyphenated:
        text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)
    if not collapse_blank_lines:
        return text.rstrip()  # keep leading lines so line numbers stay correct
    return _MANY_BLANK_LINES.sub("\n\n", text).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))
