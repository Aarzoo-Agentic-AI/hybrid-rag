"""The policy registry: metadata for each policy file, kept next to the PDFs.

Values come from the registry, not from guessing at the text: the owner of each policy
decides its category, version, effective date, jurisdiction and audience, and who may read
it. A file that isn't listed still loads, with the registry defaults and its PDF title, and a
warning.

```yaml
defaults:                  # applied to every policy, overridden per file
  jurisdiction: US
  language: en
  access:                  # merged key by key with each entry's access block
    classification: internal
    visibility: tenant
policies:
  Travel_Policy.pdf:
    policy_id: HR-TRV-003
    policy_name: Travel Policy
    policy_category: travel
    effective_date: 2027-01-01
  Salary_Bands.pdf:
    policy_name: Salary Bands
    access:
      classification: confidential
      visibility: restricted
      allowed_groups: [hr-compensation]
```

Access is kept apart from the rest of the entry: it is not part of ``entry_hash`` (so a
permission change never re-embeds a document), and it never reaches the contextual header.
With no ``access`` block anywhere, a document is readable by tenant admins only (deny by
default).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hybrid_rag.access import AccessPolicy
from hybrid_rag.ingestion.cleanup import Correction
from hybrid_rag.ingestion.types import IngestionError

ACCESS_KEY = "access"
CORRECTIONS_KEY = "corrections"  # per-file text fixes for parser errors (see cleanup.py)
_NOT_POLICY_FIELDS = {ACCESS_KEY, CORRECTIONS_KEY}


class RegistryError(IngestionError):
    """The registry file is missing, unreadable or has invalid entries."""


class PolicyInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in the registry is an error

    policy_name: str
    policy_id: str | None = None  # e.g. "HR-TRV-003"
    policy_category: str = "general"  # travel, expenses, benefits, compensation, handbook, ...
    version: str | None = None
    effective_date: date | None = None
    is_current: bool = True  # False for superseded versions; filtered out by default later
    jurisdiction: str = "global"  # "US", "UK", "US-CA", "global"
    applies_to: list[str] = Field(default_factory=lambda: ["all_employees"])
    owner_department: str | None = None
    review_date: date | None = None
    language: str = "en"

    @field_validator("version", mode="before")
    @classmethod
    def _version_as_text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, int | float) else value  # YAML reads 1.0 as float

    def metadata(self) -> dict[str, Any]:
        """Chunk metadata: JSON-friendly (ISO dates), without empty values."""
        data = self.model_dump(mode="json")
        return {k: v for k, v in data.items() if v is not None}


class PolicyRegistry:
    def __init__(
        self,
        policies: dict[str, dict[str, Any]] | None = None,
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.defaults = dict(defaults or {})
        self._policies = {name.lower(): dict(entry) for name, entry in (policies or {}).items()}
        for name in self._policies:  # validate every entry up front, not on first use
            self.lookup(name, title=name)
            self.access(name)
            self.corrections(name)
        if not self._policies:
            self.access("")  # still validate the defaults

    @classmethod
    def empty(cls) -> PolicyRegistry:
        return cls()

    @classmethod
    def load(cls, path: Path) -> PolicyRegistry:
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except OSError as exc:
            raise RegistryError(f"Can't read the policy registry {path}: {exc}") from exc
        except yaml.YAMLError as exc:
            raise RegistryError(f"The policy registry {path} isn't valid YAML: {exc}") from exc
        unknown = set(data) - {"defaults", "policies"}
        if unknown:
            raise RegistryError(f"Unknown top-level keys in {path}: {sorted(unknown)}")
        try:
            return cls(data.get("policies") or {}, data.get("defaults") or {})
        except RegistryError as exc:
            raise RegistryError(f"{path}: {exc}") from None

    def __contains__(self, source: str) -> bool:
        return source.lower() in self._policies

    def lookup(self, source: str, *, title: str) -> tuple[PolicyInfo, list[str]]:
        """The policy for ``source``, plus warnings (for example, not in the registry)."""
        entry = self._policies.get(source.lower())
        warnings = []
        if entry is None:
            if self._policies:  # only worth a warning when a registry is in use
                warnings.append(f"{source} is not in the policy registry; using defaults")
            entry = {}
        fields = {
            k: v for k, v in {**self.defaults, **entry}.items() if k not in _NOT_POLICY_FIELDS
        }
        fields.setdefault("policy_name", title or Path(source).stem.replace("_", " "))
        try:
            return PolicyInfo(**fields), warnings
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            raise RegistryError(f"Invalid registry entry for {source}: {problems}") from None

    def access(self, source: str) -> tuple[AccessPolicy, list[str]]:
        """Who may read ``source``: the default access block, overridden key by key by the
        file's own. Unset fields take the most closed value. Returns warnings too."""
        entry = self._policies.get(source.lower()) or {}
        try:
            fields = {**(self.defaults.get(ACCESS_KEY) or {}), **(entry.get(ACCESS_KEY) or {})}
            policy = AccessPolicy(**fields)
        except (ValidationError, TypeError) as exc:
            problems = (
                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            where = f"for {source}" if source else "in the defaults"
            raise RegistryError(f"Invalid access rules {where}: {problems}") from None
        warnings = []
        if not policy.has_readers and (self._policies or self.defaults):
            warnings.append(
                f"{source} has no readers (visibility restricted, no grants): "
                "only tenant admins can see it"
            )
        return policy, warnings

    def corrections(self, source: str) -> list[Correction]:
        """Text fixes for this file's parser output. Per file only: never in ``defaults``."""
        entry = self._policies.get(source.lower()) or {}
        items = entry.get(CORRECTIONS_KEY) or []
        try:
            if not isinstance(items, list):
                raise TypeError("corrections must be a list of {find, replace} entries")
            return [Correction(**item) for item in items]
        except (ValidationError, TypeError) as exc:
            problems = (
                "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                if isinstance(exc, ValidationError)
                else str(exc)
            )
            raise RegistryError(f"Invalid corrections for {source}: {problems}") from None

    def entry_hash(self, source: str) -> str:
        """Changes whenever this file's registry entry (or the defaults) change, except for
        the access rules: those are applied without re-processing the file. Corrections
        change the text, so they are included."""
        data = {
            "defaults": _without_access(self.defaults),
            "entry": _without_access(self._policies.get(source.lower())),
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[
            :12
        ]


def _without_access(entry: dict[str, Any] | None) -> dict[str, Any] | None:
    return None if entry is None else {k: v for k, v in entry.items() if k != ACCESS_KEY}
