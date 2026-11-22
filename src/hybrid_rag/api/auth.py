"""API keys -> principals.

Each request sends ``X-API-Key``. The keys file (``api.keys_file``, git-ignored) stores only
the SHA-256 of each key, with the principal it stands for:

```yaml
keys:
  - name: demo-manager
    sha256: 9f2c...                  # sha256 of the key; the key itself is never stored
    principal: {tenant_id: dummymocktest, user_id: u-demo-manager, clearance: confidential,
                roles: [people_manager]}
```

``scripts/make_api_key.py`` creates a key, prints it once and adds its hash here. The
principal comes only from this mapping, never from request fields. A later step would swap
this for JWT/OIDC claims without changing the routes.
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from hybrid_rag.access import Classification, Principal

KEY_PREFIX = "hrk_"  # makes leaked keys easy to recognise (and to scan for)


class PrincipalSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tenant_id: str
    user_id: str
    clearance: Classification
    groups: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    admin: bool = False

    def principal(self) -> Principal:
        return Principal(
            self.tenant_id,
            self.user_id,
            self.clearance,
            groups=frozenset(self.groups),
            roles=frozenset(self.roles),
            admin=self.admin,
        )


class KeyEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    principal: PrincipalSpec
    disabled: bool = False


@dataclass(frozen=True)
class Caller:
    key_name: str
    principal: Principal


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


class KeyStore:
    def __init__(self, entries: list[KeyEntry]) -> None:
        self._by_hash: dict[str, Caller] = {}
        for e in entries:
            if e.sha256 in self._by_hash:
                raise ValueError(f"Duplicate API key hash for {e.name!r}")
            if not e.disabled:
                self._by_hash[e.sha256] = Caller(e.name, e.principal.principal())

    @classmethod
    def load(cls, path: Path) -> KeyStore:
        if not path.exists():
            return cls([])  # no keys: every authenticated route answers 401
        if path.is_dir():
            # Docker creates a directory when a bind-mounted file doesn't exist yet.
            raise ValueError(
                f"The API keys file {path} is a directory. Delete it and create the keys with "
                "`docker compose run --rm keys --demo` (or scripts/make_api_key.py)"
            )
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([KeyEntry(**e) for e in data.get("keys") or []])

    def __len__(self) -> int:
        return len(self._by_hash)

    def authenticate(self, key: str | None) -> Caller | None:
        if not key:
            return None
        # A dict lookup on the hash: the comparison never touches the stored secret itself.
        return self._by_hash.get(hash_key(key.strip()))


def add_key(path: Path, name: str, principal: dict[str, Any]) -> str:
    """Create a key for ``principal``, store its hash in ``path`` and return the key."""
    spec = PrincipalSpec(**principal)
    spec.principal()  # validates tenant, user and clearance
    data: dict[str, Any] = {"keys": []}
    if path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {"keys": []}
    keys = data.setdefault("keys", [])
    if any(k.get("name") == name for k in keys):
        raise ValueError(f"A key named {name!r} already exists in {path}")
    key = new_key()
    keys.append(
        {"name": name, "sha256": hash_key(key), "principal": spec.model_dump(exclude_defaults=True)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# API key hashes -> principals. Created by scripts/make_api_key.py. Never commit.\n"
        + yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    return key
