"""Text helpers: clean-up and hashing."""

from __future__ import annotations

import hashlib
import re

_HYPHENATED_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile("[ \t\f\v\u00a0]+")  # also the no-break space (U+00A0)
_TRAILING_SPACES = re.compile(r"[ \t]+\n")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")


def clean_text(text: str, *, join_hyphenated: bool = False) -> str:
    """Normalise whitespace without changing the wording.

    ``join_hyphenated`` re-joins words split across lines ("hy-\\nbrid" -> "hybrid"), which
    is common in PDF text.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    if join_hyphenated:
        text = _HYPHENATED_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = _TRAILING_SPACES.sub("\n", text)
    return _MANY_BLANK_LINES.sub("\n\n", text).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))
