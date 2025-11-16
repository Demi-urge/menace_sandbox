"""Backfill embeddings for patches and enhancements."""

from vector_service.embedding_backfill import EmbeddingBackfill


def upgrade() -> None:
    """Embed existing patch and enhancement records."""
    EmbeddingBackfill().run(dbs=["patch", "enhancement"])


if __name__ == "__main__":
    upgrade()
