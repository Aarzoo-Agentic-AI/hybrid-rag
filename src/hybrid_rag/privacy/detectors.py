"""Sensitive-data detectors. Each returns ``Finding``s (entity type and position, never a copy).

- ``PatternDetector``: regular expressions with validation (checksums, number ranges). Fast,
  deterministic, no model. Always used, and also used for the after-chunking check.
- ``PresidioDetector``: Microsoft Presidio with a spaCy model, which adds person names and
  more ID formats. Loaded lazily, the first time it's needed.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Finding:
    entity: str
    start: int
    end: int
    score: float = 1.0


class Detector(Protocol):
    name: str
    entities: frozenset[str]

    def detect(self, text: str) -> list[Finding]: ...


# ------------------------------------------------------------------ patterns

_EMAIL = re.compile(
    r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"
)
# North American numbers: 415-555-0134, (415) 555-0134, +1 415.555.0134
_PHONE = re.compile(
    r"(?<![\w-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\d{3}[\s.-])\d{3}[\s.-]\d{4}(?![\w-])"
)
_SSN = re.compile(r"(?<![\w-])(\d{3})[- ](\d{2})[- ](\d{4})(?![\w-])")
_CARD = re.compile(r"(?<![\w-])\d(?:[ -]?\d){12,18}(?![\w-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _valid_phone(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    if len(digits) == 11 and digits[0] == "1":
        digits = digits[1:]
    return len(digits) == 10 and digits[0] in "23456789" and digits[3] in "23456789"


def _valid_ssn(m: re.Match[str]) -> bool:
    area, group, serial = m.groups()
    return (
        area not in {"000", "666"}
        and not area.startswith("9")
        and group != "00"
        and serial != "0000"
    )


def _valid_card(match: str) -> bool:
    digits = re.sub(r"\D", "", match)
    return 13 <= len(digits) <= 19 and digits[0] in "23456" and _luhn_valid(digits)


def _valid_iban(match: str) -> bool:
    iban = match.replace(" ", "")
    if not 15 <= len(iban) <= 34:
        return False
    rearranged = iban[4:] + iban[:4]
    number = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(number) % 97 == 1


class PatternDetector:
    """Regex detectors with validation. Doesn't find names (that needs a language model)."""

    SUPPORTED = frozenset(
        {"EMAIL_ADDRESS", "PHONE_NUMBER", "EMPLOYEE_ID", "US_SSN", "CREDIT_CARD", "IBAN_CODE"}
    )
    name = "patterns"

    def __init__(self, entities: Iterable[str], employee_id_pattern: str) -> None:
        self.entities = frozenset(entities) & self.SUPPORTED
        self._employee_id = re.compile(employee_id_pattern)

    def detect(self, text: str) -> list[Finding]:
        found: list[Finding] = []

        def add(entity: str, matches: Iterable[re.Match[str]], valid=lambda m: True) -> None:
            if entity in self.entities:
                found.extend(Finding(entity, m.start(), m.end()) for m in matches if valid(m))

        add("EMAIL_ADDRESS", _EMAIL.finditer(text))
        add("PHONE_NUMBER", _PHONE.finditer(text), lambda m: _valid_phone(m.group()))
        add("US_SSN", _SSN.finditer(text), _valid_ssn)
        add("CREDIT_CARD", _CARD.finditer(text), lambda m: _valid_card(m.group()))
        add("IBAN_CODE", _IBAN.finditer(text), lambda m: _valid_iban(m.group()))
        add("EMPLOYEE_ID", self._employee_id.finditer(text))
        return found


# ------------------------------------------------------------------ Presidio

PRESIDIO_ENTITIES = frozenset(
    {
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
        "CREDIT_CARD",
        "IBAN_CODE",
        "US_BANK_NUMBER",
        "US_PASSPORT",
        "US_ITIN",
    }
)


class PresidioDetector:
    """Presidio ``AnalyzerEngine`` with a spaCy model (English). Built on first use."""

    def __init__(self, entities: Iterable[str], spacy_model: str, score_threshold: float) -> None:
        self.entities = frozenset(entities) & PRESIDIO_ENTITIES
        self.spacy_model = spacy_model
        self.score_threshold = score_threshold
        self.name = f"presidio:{spacy_model}"
        self._analyzer: Any = None
        self._lock = threading.Lock()

    def _get_analyzer(self) -> Any:
        with self._lock:
            if self._analyzer is None:
                from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
                from presidio_analyzer.nlp_engine import NlpEngineProvider

                logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
                nlp = NlpEngineProvider(
                    nlp_configuration={
                        "nlp_engine_name": "spacy",
                        "models": [{"lang_code": "en", "model_name": self.spacy_model}],
                    }
                ).create_engine()
                registry = RecognizerRegistry(supported_languages=["en"])
                registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp)
                self._analyzer = AnalyzerEngine(
                    registry=registry, nlp_engine=nlp, supported_languages=["en"]
                )
                logger.info("Loaded Presidio with spaCy model %s", self.spacy_model)
            return self._analyzer

    def detect(self, text: str) -> list[Finding]:
        if not self.entities or not text.strip():
            return []
        results = self._get_analyzer().analyze(
            text=text,
            language="en",
            entities=sorted(self.entities),
            score_threshold=self.score_threshold,
        )
        return [Finding(r.entity_type, r.start, r.end, r.score) for r in results]


# --------------------------------------------------------- name plausibility

_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_TITLES = {"dr", "mr", "mrs", "ms", "mx", "prof"}


_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)


def lowercase_vocabulary(texts: Iterable[str]) -> frozenset[str]:
    """Words that appear in lower case somewhere in the document.

    Email addresses and URLs are left out: ``maria.gonzalez@...`` must not make "Maria
    Gonzalez" look like ordinary words.
    """
    words: set[str] = set()
    for text in texts:
        text = _URL.sub(" ", _EMAIL.sub(" ", text))
        words.update(w for w in _WORD.findall(text) if w.islower())
    return frozenset(words)


def plausible_name(span: str, lowercase_words: frozenset[str]) -> bool:
    """Filter spaCy's PERSON guesses, which are noisy on policy text ("Roth", "Tier 2").

    Accepted: at least two capitalised words, no digits, and no word that also appears in lower
    case in the same document (real names rarely do; "Policy Handbook" and "Required Notices"
    do). Measured on the HR policies in data/pdfs: 31 false positives down to 0, with all
    planted test names still found.
    """
    if re.search(r"\d", span):
        return False
    words = [w for w in _WORD.findall(span) if w.lower().rstrip(".") not in _TITLES]
    if len(words) < 2 or not all(w[0].isupper() for w in words):
        return False
    return not any(w.lower() in lowercase_words for w in words)
