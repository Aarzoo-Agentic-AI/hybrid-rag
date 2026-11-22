"""Access control: who may read which documents (FEATURE.md section 11).

Two halves:

- ``AccessPolicy``: a document's permissions, from the ``access:`` block of the policy
  registry. Copied onto every chunk. The defaults are the most closed values (classification
  ``restricted``, visibility ``restricted``, no grants): deny by default.
- ``Principal``: the authenticated caller (tenant, user, groups, roles, clearance), built on
  the server from the caller's identity, never from request fields.

The rule, enforced inside the vector search by ``hybrid_rag.store.security_filter``::

    tenant_id == caller.tenant
    AND classification_rank <= caller.clearance
    AND ( visibility == "tenant"
          OR caller.user_id in allowed_users
          OR caller.groups  ∩  allowed_groups
          OR caller.roles   ∩  allowed_roles )

``Principal.can_read()`` is the same rule in Python, used by tests to cross-check the filter.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator

Classification = Literal["public", "internal", "confidential", "restricted"]
Visibility = Literal["tenant", "restricted"]

# Ranks, so the filter can check ``classification_rank <= clearance`` with one range condition.
CLASSIFICATION_RANKS: dict[str, int] = {
    name: rank for rank, name in enumerate(get_args(Classification))
}

# The access fields on every chunk, besides ``access_version`` and ``access_updated_at``,
# which the store sets when it writes them.
ACCESS_FIELDS = (
    "tenant_id",
    "classification",
    "classification_rank",
    "visibility",
    "allowed_groups",
    "allowed_roles",
    "allowed_users",
    "access_hash",
)

_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_GRANT = re.compile(r"^\S{1,128}$")


class AccessError(Exception):
    """Base class for access-control errors."""


class AccessDeniedError(AccessError):
    """The caller isn't allowed to do this."""


def validate_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not _TENANT_ID.match(tenant_id):
        raise ValueError(
            f"Invalid tenant ID {tenant_id!r}: use 1-63 lowercase letters, digits, '-' or '_', "
            "starting with a letter or digit"
        )
    return tenant_id


def _clean_grants(values: Iterable[str]) -> list[str]:
    cleaned = sorted({str(v).strip() for v in values})
    bad = [v for v in cleaned if not _GRANT.match(v)]
    if bad:
        raise ValueError(f"grants must be non-empty and contain no spaces: {bad}")
    return cleaned


class AccessPolicy(BaseModel):
    """A document's permissions. Unset fields take the most closed value."""

    model_config = ConfigDict(extra="forbid", frozen=True)  # a typo is an error, not a grant

    classification: Classification = "restricted"
    visibility: Visibility = "restricted"
    allowed_groups: list[str] = Field(
        default_factory=list
    )  # e.g. "hr-compensation", "dept:finance"
    allowed_roles: list[str] = Field(default_factory=list)  # e.g. "people_manager"
    allowed_users: list[str] = Field(default_factory=list)  # exceptions only, e.g. "u-1042"

    @field_validator("allowed_groups", "allowed_roles", "allowed_users", mode="before")
    @classmethod
    def _grants(cls, value: Any) -> Any:
        if isinstance(value, str):  # "allowed_roles: hr_director" in YAML
            value = [value]
        return _clean_grants(value) if isinstance(value, list | tuple | set) else value

    @property
    def classification_rank(self) -> int:
        return CLASSIFICATION_RANKS[self.classification]

    @property
    def has_readers(self) -> bool:
        """False if only tenant admins can read the document."""
        return self.visibility == "tenant" or bool(
            self.allowed_groups or self.allowed_roles or self.allowed_users
        )

    @property
    def hash(self) -> str:
        """Changes whenever the permissions change. Stored as ``access_hash``."""
        data = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:12]

    def metadata(self, tenant_id: str) -> dict[str, Any]:
        """The access fields for every chunk of the document (see ``ACCESS_FIELDS``)."""
        return {
            "tenant_id": validate_tenant_id(tenant_id),
            "classification": self.classification,
            "classification_rank": self.classification_rank,
            "visibility": self.visibility,
            "allowed_groups": list(self.allowed_groups),
            "allowed_roles": list(self.allowed_roles),
            "allowed_users": list(self.allowed_users),
            "access_hash": self.hash,
        }

    def describe(self) -> str:
        """Short summary for logs and the CLI; grant names, never user IDs."""
        grants = [f"group {g}" for g in self.allowed_groups]
        grants += [f"role {r}" for r in self.allowed_roles]
        if self.allowed_users:
            grants.append(f"{len(self.allowed_users)} user(s)")
        who = (
            "whole tenant" if self.visibility == "tenant" else (", ".join(grants) or "admins only")
        )
        return f"{self.classification}, {who}"


@dataclass(frozen=True)
class Principal:
    """The authenticated caller. Every read of the index needs one.

    ``admin`` is a tenant administrator (the ingestion CLI, later the admin API): it sees every
    document of its own tenant, whatever the classification or grants, and nothing outside it.
    """

    tenant_id: str
    user_id: str
    clearance: Classification
    groups: frozenset[str] = field(default_factory=frozenset)
    roles: frozenset[str] = field(default_factory=frozenset)
    admin: bool = False

    def __post_init__(self) -> None:
        validate_tenant_id(self.tenant_id)
        if not isinstance(self.user_id, str) or not self.user_id.strip():
            raise ValueError("A principal needs a user ID")
        if self.clearance not in CLASSIFICATION_RANKS:
            raise ValueError(
                f"Unknown clearance {self.clearance!r}; use one of {list(CLASSIFICATION_RANKS)}"
            )
        # Accept any iterable, store frozensets (so principals are hashable and immutable).
        object.__setattr__(self, "groups", frozenset(self.groups))
        object.__setattr__(self, "roles", frozenset(self.roles))

    @classmethod
    def tenant_admin(cls, tenant_id: str) -> Principal:
        return cls(tenant_id, user_id="system:admin", clearance="restricted", admin=True)

    @property
    def clearance_rank(self) -> int:
        return CLASSIFICATION_RANKS[self.clearance]

    def can_read(self, metadata: Mapping[str, Any]) -> bool:
        """The access rule in Python, for one chunk's metadata. Missing fields deny."""
        if metadata.get("tenant_id") != self.tenant_id:
            return False
        if self.admin:
            return True
        rank = metadata.get("classification_rank")
        if not isinstance(rank, int) or rank > self.clearance_rank:
            return False
        return (
            metadata.get("visibility") == "tenant"
            or self.user_id in metadata.get("allowed_users", [])
            or bool(self.groups & set(metadata.get("allowed_groups", [])))
            or bool(self.roles & set(metadata.get("allowed_roles", [])))
        )


class AccessOverrides:
    """Permission changes made through the API, in a JSON file (``documents.access_overrides``).

    The registry says who may read each policy; an override set with
    ``PUT /documents/{id}/access`` replaces that for one document, until it is cleared. Every
    ingest (API or ``scripts/ingest.py``) reads the overrides on top of the registry, so a CLI
    run never silently undoes a change made through the API. Keyed by tenant, then source.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()

    def _read(self) -> dict[str, dict[str, dict[str, Any]]]:
        if self.path is None or not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AccessError(f"Can't read access overrides {self.path}: {exc}") from exc

    def get(self, tenant_id: str, source: str) -> AccessPolicy | None:
        entry = self._read().get(tenant_id, {}).get(source.lower())
        return AccessPolicy(**entry) if entry is not None else None

    def set(self, tenant_id: str, source: str, policy: AccessPolicy | None) -> None:
        """Store an override, or remove it with ``None`` (back to the registry's rules)."""
        if self.path is None:
            raise AccessError("No access overrides file is configured")
        with self._lock:
            data = self._read()
            tenant = data.setdefault(tenant_id, {})
            if policy is None:
                tenant.pop(source.lower(), None)
            else:
                tenant[source.lower()] = policy.model_dump(mode="json")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
