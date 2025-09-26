from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from vector_service.stack_ingestion import (
    StackDatasetStreamer,
    StackIngestionConfig,
    StackMetadataStore,
)
class _FakeEmbedder:
    def encode(self, texts):  # pragma: no cover - trivial helper
        return [[float(len(text)), 1.0, 0.0, -1.0] for text in texts]


class _FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.records.append(
            {
                "kind": kind,
                "id": record_id,
                "vector": list(vector),
                "origin_db": origin_db,
                "metadata": dict(metadata or {}),
            }
        )


class _StubVectorService:
    def __init__(self, embedder: _FakeEmbedder, store: _FakeVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: dict,
        *,
        origin_db: str | None = None,
        metadata: dict | None = None,
    ):
        vec = self.embedder.encode([record.get("text", "")])[0]
        self.store.add(kind, record_id, vec, origin_db=origin_db, metadata=metadata)
        return vec


def _dataset(records):
    def _loader(*args, **kwargs):
        return list(records)

    return _loader


def _streamer(tmp_path: Path, records: list[dict]) -> tuple[StackDatasetStreamer, _FakeVectorStore]:
    cache_dir = tmp_path / "cache"
    vector_path = tmp_path / "vectors.ann"
    metadata_path = tmp_path / "meta.db"
    config = StackIngestionConfig(
        dataset_name="dummy",
        split="train",
        allowed_languages={"python"},
        max_lines=2,
        vector_dim=4,
        vector_backend="annoy",
        vector_metric="angular",
        vector_store_path=vector_path,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
    )
    store = StackMetadataStore(config.metadata_path)
    fake_store = _FakeVectorStore()
    service = _StubVectorService(_FakeEmbedder(), fake_store)
    streamer = StackDatasetStreamer(
        config=config,
        metadata_store=store,
        vector_service=service,
        dataset_loader=_dataset(records),
        vector_store_factory=lambda cfg: fake_store,
    )
    return streamer, fake_store


def test_stack_streamer_filters_languages_and_chunks(tmp_path):
    records = [
        {
            "content": "print('hi')\nprint('bye')\nprint('done')",
            "language": "python",
            "path": "src/app.py",
            "repo_name": "example/repo",
            "id": "0",
        },
        {
            "content": "console.log('skip');",
            "language": "javascript",
            "path": "web/app.js",
            "repo_name": "example/repo",
            "id": "1",
        },
    ]
    streamer, store = _streamer(tmp_path, records)
    count = streamer.process()
    assert count == 2
    assert len(store.records) == 2
    for rec in store.records:
        assert rec["kind"] == "stack"
        assert "text" not in rec["metadata"]
        assert rec["metadata"]["language"] == "python"
        assert rec["metadata"]["start_line"] <= rec["metadata"]["end_line"]

    conn = sqlite3.connect(streamer.config.metadata_path)
    try:
        rows = conn.execute("SELECT repo, path, language FROM chunks").fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert all("print" not in str(row) for row in rows)


def test_stack_streamer_resumes_from_metadata(tmp_path):
    records = [
        {
            "content": "print('hi')\nprint('bye')",
            "language": "python",
            "path": "src/app.py",
            "repo_name": "example/repo",
            "id": "0",
        }
    ]
    streamer, store = _streamer(tmp_path, records)
    first = streamer.process()
    assert first == 1
    assert len(store.records) == 1

    # Recreate streamer sharing the same metadata store to simulate restart
    streamer2, store2 = _streamer(tmp_path, records)
    resumed = streamer2.process()
    assert resumed == 0
    assert store2.records == []


def test_stack_streamer_async_entrypoint(tmp_path):
    records = [
        {
            "content": "print('hi')\nprint('bye')",
            "language": "python",
            "path": "src/app.py",
            "repo_name": "example/repo",
            "id": "0",
        }
    ]
    streamer, store = _streamer(tmp_path, records)
    result = asyncio.run(streamer.process_async(limit=1))
    assert result == 1
    assert len(store.records) == 1
