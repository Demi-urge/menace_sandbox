from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import pytest

from vector_service.stack_ingestion import (
    StackDatasetStreamer,
    StackIngestionConfig,
    StackMetadataStore,
)


class _StubVectorStore:
    """In-memory stand-in for the production FAISS/Annoy vector store."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Iterable[float],
        *,
        origin_db: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.records.append(
            {
                "kind": kind,
                "record_id": record_id,
                "vector": list(vector),
                "origin_db": origin_db,
                "metadata": dict(metadata or {}),
            }
        )

    def query(self, vector: Iterable[float], top_k: int = 5) -> List[tuple[str, float]]:
        raise NotImplementedError

    def load(self) -> None:  # pragma: no cover - interface compatibility
        return None


class _StubVectorService:
    """Returns a predictable embedding proportional to the text length."""

    def __init__(self, store: _StubVectorStore) -> None:
        self.store = store

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: Dict[str, Any],
        *,
        origin_db: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> List[float]:
        text = str(record.get("text", ""))
        vector = [float(len(text))]
        self.store.add(kind, record_id, vector, origin_db=origin_db, metadata=metadata)
        return vector


def _dataset(records: List[Dict[str, Any]]):
    def _loader(*args: Any, **kwargs: Any) -> Iterator[Dict[str, Any]]:
        return iter(records)

    return _loader


def _build_streamer(tmp_path: Path, records: List[Dict[str, Any]]):
    cache_dir = tmp_path / "cache"
    vector_path = tmp_path / "embeddings.faiss"
    metadata_path = tmp_path / "stack_metadata.db"
    document_cache = tmp_path / "documents"
    config = StackIngestionConfig(
        dataset_name="dummy/stack",
        split="train",
        allowed_languages={"python"},
        max_lines=2,
        chunk_overlap=0,
        vector_store_path=vector_path,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        document_cache=document_cache,
    )
    metadata_store = StackMetadataStore(config.metadata_path)
    vector_store = _StubVectorStore()
    service = _StubVectorService(vector_store)
    streamer = StackDatasetStreamer(
        config=config,
        metadata_store=metadata_store,
        vector_service=service,
        dataset_loader=_dataset(records),
        vector_store_factory=lambda cfg: vector_store,
    )
    return streamer, vector_store, document_cache


@pytest.fixture(autouse=True)
def _enable_streaming(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
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
    streamer, store, document_cache = _build_streamer(tmp_path, records)
    count = streamer.process()
    assert count == 2

    assert len(store.records) == 2
    for item in store.records:
        assert item["kind"] == "code"
        assert item["origin_db"] == "stack"
        metadata = item["metadata"]
        assert metadata["repo"] == "example/repo"
        assert metadata["language"] == "python"
        assert metadata["path"].endswith("app.py")
        assert len(metadata["summary_hash"]) == 40
        assert "print" not in metadata["summary_hash"]

    conn = sqlite3.connect(streamer.config.metadata_path)
    try:
        rows = conn.execute(
            "SELECT repo, path, language, summary_hash, snippet_path FROM chunks"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    for repo, path, language, summary_hash, snippet_path in rows:
        assert repo == "example/repo"
        assert language == "python"
        assert path.endswith("app.py")
        assert len(summary_hash) == 40
        assert snippet_path
        snippet_file = document_cache / snippet_path
        assert snippet_file.exists()
        cached_text = snippet_file.read_text(encoding="utf-8")
        assert any(token in cached_text for token in ("print('hi')", "print('bye')", "print('done')"))
        assert len(cached_text) <= 1200

    streamer.metadata_store.close()


def test_stack_streamer_resumes_from_high_water_mark(tmp_path: Path) -> None:
    record = {
        "content": "print('hi')\nprint('bye')",
        "language": "python",
        "path": "src/app.py",
        "repo_name": "example/repo",
        "id": "0",
    }
    streamer, store, _ = _build_streamer(tmp_path, [record])
    first = streamer.process()
    assert first == 1

    streamer.metadata_store.close()

    streamer2, store2, _ = _build_streamer(tmp_path, [record])
    resumed = streamer2.process()
    assert resumed == 0
    assert len(store2.records) == 0


def test_stack_streamer_drops_raw_payloads(tmp_path: Path) -> None:
    record = {
        "content": "print('hi')\nprint('bye')",
        "language": "python",
        "path": "src/app.py",
        "repo_name": "example/repo",
        "id": "0",
    }
    streamer, store, _ = _build_streamer(tmp_path, [record])
    count = streamer.process()
    assert count == 1

    # Ensure the stub store never saw raw content in metadata
    stored = store.records[0]
    assert "text" not in stored["metadata"]
    assert stored["metadata"]["summary_hash"] != record["content"]

    conn = sqlite3.connect(streamer.config.metadata_path)
    try:
        chunk_rows = conn.execute(
            "SELECT summary_hash FROM chunks"
        ).fetchall()
    finally:
        conn.close()
    assert len(chunk_rows) == 1
    (summary_hash,) = chunk_rows[0]
    assert summary_hash != record["content"]

    # Async helper still processes and returns chunk counts
    streamer.metadata_store.close()
    new_record = {
        "content": "print('new')",
        "language": "python",
        "path": "src/app.py",
        "repo_name": "example/repo",
        "id": "1",
    }
    streamer_async, _, _ = _build_streamer(tmp_path, [new_record])
    result = asyncio.run(streamer_async.process_async(limit=1))
    assert result == 1

