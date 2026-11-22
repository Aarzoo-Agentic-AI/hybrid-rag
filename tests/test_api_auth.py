"""API keys and access overrides."""

from pathlib import Path

import pytest
import yaml

from hybrid_rag.access import AccessOverrides, AccessPolicy
from hybrid_rag.api.auth import KEY_PREFIX, KeyStore, add_key, hash_key

SPEC = {"tenant_id": "acme", "user_id": "u-1", "clearance": "internal"}


def test_keys_are_stored_only_as_hashes(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"

    key = add_key(path, "alice", SPEC)

    assert key.startswith(KEY_PREFIX)
    assert key not in path.read_text()
    assert hash_key(key) in path.read_text()
    caller = KeyStore.load(path).authenticate(key)
    assert caller.key_name == "alice" and caller.principal.user_id == "u-1"


def test_unknown_disabled_and_empty_keys_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "keys.yaml"
    key = add_key(path, "alice", SPEC)
    data = yaml.safe_load(path.read_text())
    data["keys"][0]["disabled"] = True
    path.write_text(yaml.safe_dump(data))

    store = KeyStore.load(path)

    assert store.authenticate(key) is None
    assert store.authenticate("hrk_unknown") is None
    assert store.authenticate("") is None and store.authenticate(None) is None
    assert len(KeyStore.load(tmp_path / "missing.yaml")) == 0


def test_invalid_key_entries_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        add_key(tmp_path / "k.yaml", "bad", {**SPEC, "clearance": "top-secret"})
    add_key(tmp_path / "k.yaml", "alice", SPEC)
    with pytest.raises(ValueError, match="already exists"):
        add_key(tmp_path / "k.yaml", "alice", SPEC)


def test_overrides_are_kept_per_tenant_and_source(tmp_path: Path) -> None:
    overrides = AccessOverrides(tmp_path / "o.json")
    policy = AccessPolicy(classification="internal", visibility="tenant")

    overrides.set("acme", "Salary.PDF", policy)

    assert overrides.get("acme", "salary.pdf") == policy
    assert overrides.get("globex", "salary.pdf") is None
    overrides.set("acme", "salary.pdf", None)
    assert overrides.get("acme", "salary.pdf") is None
    assert AccessOverrides(tmp_path / "missing.json").get("acme", "x.pdf") is None
