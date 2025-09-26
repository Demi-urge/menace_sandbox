from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Mapping

import pytest

from vector_service.stack_ingestor import StackIngestor, StackMetadataStore


class DummyDataset:
    def __init__(self, samples: Iterable[Mapping[str, object]]) -> None:
        self._samples = [dict(sample) for sample in samples]

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        for sample in self._samples:
            yield dict(sample)


class DummyVectorStore:
    def __init__(self) -> None:
        self.records: List[dict] = []

    def add(
        self,
        kind: str,
        record_id: str,
        vector: Iterable[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, object] | None = None,
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
        return []

    def load(self) -> None:
        return None


@pytest.fixture
def sample_records() -> List[Mapping[str, object]]:
    return [
        {
            "language": "Python",
            "repo_name": "repo/example",
            "path": "a.py",
            "content": "print('hi')\nprint('bye')",
        },
        {
            "language": "python",
            "repo_name": "repo/example",
            "path": "b.py",
            "content": "def foo():\n    return 1",
        },
    ]


def _patch_load_dataset(monkeypatch: pytest.MonkeyPatch, samples: Iterable[Mapping[str, object]]) -> None:
    def fake_load_dataset(name: str, **kwargs):  # type: ignore[override]
        assert kwargs.get("streaming") is True
        return DummyDataset(samples)

    monkeypatch.setattr("vector_service.stack_ingestor.load_dataset", fake_load_dataset)


def test_stack_ingestor_batches_and_persists_metadata(tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load_dataset(monkeypatch, sample_records)

    calls: List[List[str]] = []

    def fake_embeddings(texts: List[str], **_: object) -> List[List[float]]:
        calls.append(list(texts))
        return [[float(len(text)), float(text.count("\n") + 1)] for text in texts]

    metadata_path = tmp_path / "meta.db"
    metadata_store = StackMetadataStore(metadata_path, namespace="test")
    vector_store = DummyVectorStore()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        metadata_path=metadata_path,
        index_path=tmp_path / "stack.index",
        vector_backend="annoy",
        embedding_fn=fake_embeddings,
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    processed = ingestor.ingest(resume=False)

    assert processed == 2
    assert len(vector_store.records) == 4
    assert all(set(entry["metadata"].keys()) == {"repo", "path", "language", "hash"} for entry in vector_store.records)
    assert all(len(batch) <= 2 for batch in calls)

    cur = metadata_store.conn.cursor()
    cur.execute(f"PRAGMA table_info({metadata_store.metadata_table})")
    columns = {row[1] for row in cur.fetchall()}
    assert "chunk_hash" in columns
    assert "content" not in columns

    cur.execute(f"SELECT chunk_hash FROM {metadata_store.metadata_table}")
    hashes = {row[0] for row in cur.fetchall()}
    assert all("print" not in chunk_hash for chunk_hash in hashes)


def test_stack_ingestor_resume_skips_existing(tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_load_dataset(monkeypatch, sample_records)

    metadata_path = tmp_path / "meta.db"
    metadata_store = StackMetadataStore(metadata_path, namespace="resume")
    vector_store = DummyVectorStore()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        metadata_path=metadata_path,
        index_path=tmp_path / "stack.index",
        vector_backend="annoy",
        embedding_fn=lambda texts, **_: [[float(len(t))] for t in texts],
        vector_store=vector_store,
        metadata_store=metadata_store,
    )

    first_processed = ingestor.ingest(resume=False)
    initial_records = len(vector_store.records)

    second_processed = ingestor.ingest(resume=True)

    assert first_processed == 2
    assert second_processed == 0
    assert len(vector_store.records) == initial_records

