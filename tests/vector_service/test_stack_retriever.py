from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from vector_service.stack_retriever import StackRetriever


class FakeVectorStore:
    def __init__(self, ids, vectors, meta):
        self.ids = list(ids)
        self.vectors = [list(vec) for vec in vectors]
        self.meta = list(meta)

    def query(self, vector, top_k: int = 5):  # pragma: no cover - simple deterministic
        return [(self.ids[i], 0.1) for i in range(min(top_k, len(self.ids)))]


class ErrorVectorStore:
    def query(self, vector, top_k: int = 5):  # pragma: no cover - simple failure
        raise RuntimeError("boom")


class RecordingVectorService:
    def __init__(self, vector):
        self.vector = list(vector)
        self.calls: list[tuple[str, dict]] = []

    def vectorise(self, kind: str, record: dict):
        self.calls.append((kind, dict(record)))
        return list(self.vector)


def _metadata_path(tmp_path: Path, rows: list[tuple[str, str, str, str, int, int]]) -> Path:
    db_path = tmp_path / "stack_metadata.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            checksum TEXT PRIMARY KEY,
            repo TEXT NOT NULL,
            path TEXT NOT NULL,
            language TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.executemany(
        "INSERT OR REPLACE INTO chunks (checksum, repo, path, language, start_line, end_line)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db_path


def _stack_meta(repo: str, path: str, language: str, license_name: str, text: str):
    return {
        "metadata": {
            "repo": repo,
            "path": path,
            "language": language,
            "license": license_name,
            "text": text,
            "redacted": True,
        }
    }


def test_retrieve_orders_and_normalises_metadata(tmp_path):
    db_path = _metadata_path(
        tmp_path,
        [
            ("chunk-a", "owner/repo", "./src/foo.py", "Python", 10, 20),
            ("chunk-b", "owner/repo", "tests/bar.py", "Python", 1, 5),
        ],
    )
    store = FakeVectorStore(
        ["chunk-a", "chunk-b"],
        [[1.0, 0.0], [0.0, 1.0]],
        [
            _stack_meta("owner/repo", "./src/foo.py", "Python", "MIT", "safe text"),
            _stack_meta("owner/repo", "tests/bar.py", "Python", "Apache-2.0", "other text"),
        ],
    )
    retriever = StackRetriever(
        vector_store=store,
        metadata_path=db_path,
        similarity_threshold=0.0,
    )

    results = retriever.retrieve([1.0, 0.0], k=2)

    assert [r["identifier"] for r in results] == ["chunk-a", "chunk-b"]
    meta = results[0]["metadata"]
    assert meta["path"] == "src/foo.py"
    assert meta["repo"] == "owner/repo"
    assert meta["language"] == "Python"
    assert meta["size"] == 11
    assert results[0]["text"] == "safe text"

    filtered = retriever.retrieve([1.0, 0.0], keywords=["foo"])
    assert [r["identifier"] for r in filtered] == ["chunk-a"]


def test_retrieve_blocks_denylisted_licenses(tmp_path):
    db_path = _metadata_path(
        tmp_path,
        [("chunk-a", "owner/repo", "file.py", "Python", 1, 3)],
    )
    store = FakeVectorStore(
        ["chunk-a"],
        [[1.0, 0.0]],
        [_stack_meta("owner/repo", "file.py", "Python", "GPL-3.0", "unsafe")],
    )
    retriever = StackRetriever(
        vector_store=store,
        metadata_path=db_path,
        similarity_threshold=0.0,
    )

    assert retriever.retrieve([1.0, 0.0], k=1) == []


def test_retrieve_handles_store_errors(tmp_path):
    db_path = _metadata_path(
        tmp_path,
        [("chunk-a", "owner/repo", "file.py", "Python", 1, 3)],
    )
    retriever = StackRetriever(
        vector_store=ErrorVectorStore(),
        metadata_path=db_path,
        similarity_threshold=0.0,
    )

    assert retriever.retrieve([1.0, 0.0], k=1) == []


def test_embed_query_uses_shared_vector_service(tmp_path):
    db_path = _metadata_path(
        tmp_path,
        [("chunk-a", "owner/repo", "file.py", "Python", 1, 3)],
    )
    store = FakeVectorStore(["chunk-a"], [[1.0, 0.0]], [_stack_meta("owner/repo", "file.py", "Python", "MIT", "text")])
    service = RecordingVectorService([0.1, 0.2])
    retriever = StackRetriever(
        vector_store=store,
        vector_service=service,
        metadata_path=db_path,
        similarity_threshold=0.0,
    )

    embedding = retriever.embed_query("hello world")

    assert embedding == [0.1, 0.2]
    assert service.calls == [("stack", {"text": "hello world"})]
