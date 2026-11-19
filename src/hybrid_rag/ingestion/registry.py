"""The policy registry: metadata for each policy file, kept next to the PDFs.

Values come from the registry, not from guessing at the text: the owner of each policy
decides its category, version, effective date, jurisdiction and audience. A file that isn't
listed still loads, with the registry defaults and its PDF title, and a warning.

```yaml
defaults:                  # applied to every policy, overridden per file
  jurisdiction: US
  language: en
policies:
  Travel_Policy.pdf:
    policy_id: HR-TRV-003
    policy_name: Travel Policy
    policy_category: travel
    effective_date: 2027-01-01
```
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from hybrid_rag.ingestion.types import IngestionError


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
        fields = {**self.defaults, **entry}
        fields.setdefault("policy_name", title or Path(source).stem.replace("_", " "))
        try:
            return PolicyInfo(**fields), warnings
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
            )
            raise RegistryError(f"Invalid registry entry for {source}: {problems}") from None

    def entry_hash(self, source: str) -> str:
        """Changes whenever this file's registry entry (or the defaults) change."""
        data = {"defaults": self.defaults, "entry": self._policies.get(source.lower())}
        return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()[
            :12
        ]
