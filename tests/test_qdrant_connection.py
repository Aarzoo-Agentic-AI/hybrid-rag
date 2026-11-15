import pytest
from qdrant_client import QdrantClient

from hybrid_rag.config import get_settings


@pytest.mark.integration
def test_qdrant_is_reachable() -> None:
    """Needs Qdrant running: `docker compose up -d`."""
    settings = get_settings()
    client = QdrantClient(url=settings.qdrant.url, timeout=5)

    # Raises if the server is unreachable.
    client.get_collections()
