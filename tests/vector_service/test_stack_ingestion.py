import asyncio
import json
import sqlite3
from pathlib import Path

import pytest

from vector_service.stack_ingestion import (
    SQLiteVectorStore,
    StackDatasetStreamer,
    StackIngestionConfig,
    StackMetadataStore,
)


class _StubVectorService:
    """Lightweight stand-in for :class:`SharedVectorService`."""

    def __init__(self, store: SQLiteVectorStore) -> None:
        self.store = store

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: dict,
        *,
        origin_db: str | None = None,
        metadata: dict | None = None,
    ) -> list[float]:
        text = str(record.get("text", ""))
        vector = [float(len(text))]
        self.store.add(kind, record_id, vector, origin_db=origin_db, metadata=metadata)
        return vector


def _dataset(records: list[dict]):
    def _loader(*args, **kwargs):
        return list(records)

    return _loader


def _streamer(tmp_path: Path, records: list[dict]) -> tuple[StackDatasetStreamer, SQLiteVectorStore]:
    cache_dir = tmp_path / "cache"
    vector_path = tmp_path / "embeddings.db"
    metadata_path = tmp_path / "meta.db"
    config = StackIngestionConfig(
        dataset_name="dummy",
        split="train",
        allowed_languages={"python"},
        max_lines=2,
        vector_store_path=vector_path,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
    )
    metadata_store = StackMetadataStore(config.metadata_path)
    vector_store = SQLiteVectorStore(config.vector_store_path)
    service = _StubVectorService(vector_store)
    streamer = StackDatasetStreamer(
        config=config,
        metadata_store=metadata_store,
        vector_service=service,
        dataset_loader=_dataset(records),
        vector_store_factory=lambda cfg: vector_store,
    )
    return streamer, vector_store


def _fetch_embeddings(db_path: Path) -> list[tuple[str, str, str, str, str, str]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, kind, origin_db, repo, path, language, vector FROM embeddings"
        ).fetchall()
    finally:
        conn.close()
    return rows


@pytest.fixture(autouse=True)
def _enable_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STACK_STREAMING", "1")
    yield
    monkeypatch.delenv("STACK_STREAMING", raising=False)


def test_stack_streamer_filters_languages_and_chunks(tmp_path: Path) -> None:
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
    try:
        count = streamer.process()
    finally:
        store.close()
    assert count == 2

    rows = _fetch_embeddings(streamer.config.vector_store_path)
    assert len(rows) == 2
    for row in rows:
        _, kind, origin_db, repo, path, language, vector_json = row
        assert kind == "stack"
        assert origin_db == "stack"
        assert repo == "example/repo"
        assert language == "python"
        assert path.endswith("app.py")
        vector = json.loads(vector_json)
        assert isinstance(vector, list)
        assert all(isinstance(value, float) for value in vector)
        assert "print" not in vector_json

    conn = sqlite3.connect(streamer.config.metadata_path)
    try:
        chunk_rows = conn.execute(
            "SELECT repo, path, language FROM chunks"
        ).fetchall()
    finally:
        conn.close()
    assert len(chunk_rows) == 2
    assert all("print" not in str(row) for row in chunk_rows)


def test_stack_streamer_resumes_from_metadata(tmp_path: Path) -> None:
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
    try:
        first = streamer.process()
    finally:
        store.close()
    assert first == 1

    rows_first = _fetch_embeddings(streamer.config.vector_store_path)
    assert len(rows_first) == 1

    streamer2, store2 = _streamer(tmp_path, records)
    try:
        resumed = streamer2.process()
    finally:
        store2.close()
    assert resumed == 0
    rows_second = _fetch_embeddings(streamer2.config.vector_store_path)
    assert rows_second == rows_first


def test_stack_streamer_async_entrypoint(tmp_path: Path) -> None:
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
    try:
        result = asyncio.run(streamer.process_async(limit=1))
    finally:
        store.close()
    assert result == 1

    rows = _fetch_embeddings(streamer.config.vector_store_path)
    assert len(rows) == 1
