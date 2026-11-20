"""The security filter: the access rule from ``hybrid_rag.access`` as a Qdrant filter.

Every read of the index (search, list, get, count, delete) combines this filter with whatever
else it filters on, so a chunk the caller may not read is never returned, counted or deleted.
Relevance filters (policy category, jurisdiction, ...) are added *next to* it with ``scoped``,
so they can narrow the results but never widen them.
"""

from __future__ import annotations

from qdrant_client import models

from hybrid_rag.access import Principal
from hybrid_rag.store.schema import field_key, match


def security_filter(principal: Principal) -> models.Filter:
    """The Qdrant filter for everything ``principal`` may read."""
    if not isinstance(principal, Principal):  # fail closed on anything but a real principal
        raise TypeError(f"A Principal is required, not {type(principal).__name__}")
    must: list[models.Condition] = [match("tenant_id", principal.tenant_id)]
    if principal.admin:
        return models.Filter(must=must)

    grants: list[models.Condition] = [
        match("visibility", "tenant"),
        match("allowed_users", principal.user_id),  # matches if the list contains the value
    ]
    if principal.groups:
        grants.append(_any("allowed_groups", principal.groups))
    if principal.roles:
        grants.append(_any("allowed_roles", principal.roles))
    must += [
        models.FieldCondition(
            key=field_key("classification_rank"), range=models.Range(lte=principal.clearance_rank)
        ),
        models.Filter(should=grants),  # at least one grant
    ]
    return models.Filter(must=must)


def scoped(
    principal: Principal,
    *conditions: models.Condition,
    must_not: list[models.Condition] | None = None,
) -> models.Filter:
    """The security filter AND the given conditions."""
    return models.Filter(must=[security_filter(principal), *conditions], must_not=must_not)


def _any(key: str, values: frozenset[str]) -> models.FieldCondition:
    return models.FieldCondition(key=field_key(key), match=models.MatchAny(any=sorted(values)))
