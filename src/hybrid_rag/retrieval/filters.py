"""Relevance filters: which policies apply to the question (not who may read them).

These narrow a search to the policies that matter: current versions only, one category, one
jurisdiction, an audience, or specific documents. They are always ANDed with the security
filter (``hybrid_rag.store.scoped``), so they can narrow the results but never widen them,
whatever values a request sends.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator
from qdrant_client import models

from hybrid_rag.store.schema import field_key, match


class RelevanceFilters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    current_only: bool = True  # leave out superseded policy versions (is_current: false)
    policy_categories: list[str] | None = None  # e.g. ["travel", "expenses"]
    jurisdictions: list[str] | None = None  # e.g. ["US", "global"]
    applies_to: list[str] | None = None  # e.g. ["employees"]
    doc_ids: list[str] | None = None  # restrict to these documents

    @field_validator("policy_categories", "jurisdictions", "applies_to", "doc_ids")
    @classmethod
    def _not_empty(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and not value:
            raise ValueError("use null for no filter; an empty list would match nothing")
        return value

    def conditions(self) -> list[models.Condition]:
        """Qdrant conditions, all of which must hold."""
        conditions: list[models.Condition] = []
        if self.current_only:
            conditions.append(match("is_current", True))
        for key, values in (
            ("policy_category", self.policy_categories),
            ("jurisdiction", self.jurisdictions),
            ("applies_to", self.applies_to),
            ("doc_id", self.doc_ids),
        ):
            if values:
                conditions.append(
                    models.FieldCondition(key=field_key(key), match=models.MatchAny(any=values))
                )
        return conditions
