"""The collection layout: vector names, payload keys and payload indexes."""

from __future__ import annotations

from typing import Any

from qdrant_client import models

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"
CONTENT_KEY = "page_content"
METADATA_KEY = "metadata"

_KEYWORD = models.PayloadSchemaType.KEYWORD
_INTEGER = models.PayloadSchemaType.INTEGER

# Fields we filter on get a payload index, so filtered searches and deletes stay fast.
PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType | models.KeywordIndexParams] = {
    # Access control (FEATURE.md section 11). ``is_tenant`` lets Qdrant store each tenant's
    # points together, so every (tenant-filtered) search reads only that tenant's data.
    f"{METADATA_KEY}.tenant_id": models.KeywordIndexParams(
        type=models.KeywordIndexType.KEYWORD, is_tenant=True
    ),
    f"{METADATA_KEY}.classification_rank": _INTEGER,
    f"{METADATA_KEY}.visibility": _KEYWORD,
    f"{METADATA_KEY}.allowed_groups": _KEYWORD,
    f"{METADATA_KEY}.allowed_roles": _KEYWORD,
    f"{METADATA_KEY}.allowed_users": _KEYWORD,
    # Document.
    f"{METADATA_KEY}.doc_id": _KEYWORD,
    f"{METADATA_KEY}.doc_hash": _KEYWORD,
    f"{METADATA_KEY}.source": _KEYWORD,
    f"{METADATA_KEY}.file_type": _KEYWORD,
    f"{METADATA_KEY}.chunk_index": _INTEGER,
    f"{METADATA_KEY}.pii_scan.status": _KEYWORD,
    # Policy and section metadata, for the relevance filters and small-to-big retrieval.
    f"{METADATA_KEY}.policy_id": _KEYWORD,
    f"{METADATA_KEY}.policy_category": _KEYWORD,
    f"{METADATA_KEY}.jurisdiction": _KEYWORD,
    f"{METADATA_KEY}.applies_to": _KEYWORD,
    f"{METADATA_KEY}.is_current": models.PayloadSchemaType.BOOL,
    f"{METADATA_KEY}.section_id": _KEYWORD,
    f"{METADATA_KEY}.chunk_type": _KEYWORD,
}


def field_key(key: str) -> str:
    """The payload path of a chunk metadata field."""
    return f"{METADATA_KEY}.{key}"


def match(key: str, value: Any) -> models.FieldCondition:
    return models.FieldCondition(key=field_key(key), match=models.MatchValue(value=value))
