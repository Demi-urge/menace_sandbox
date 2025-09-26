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


class DummyVectorService:
    def __init__(self) -> None:
        self.calls: List[dict] = []

    def vectorise_and_store(
        self,
        kind: str,
        record_id: str,
        record: Mapping[str, object],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> List[float]:
        self.calls.append(
            {
                "kind": kind,
                "record_id": record_id,
                "record": dict(record),
                "origin_db": origin_db,
                "metadata": dict(metadata or {}),
            }
        )
        text = str(record.get("text", ""))
        return [float(len(text)), float(text.count("\n") + 1)]


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


def _patch_load_dataset(monkeypatch: pytest.MonkeyPatch, samples: Iterable[Mapping[str, object]]):
    def fake_load_dataset(name: str, **kwargs):  # type: ignore[override]
        assert kwargs.get("streaming") is True
        return DummyDataset(samples)

    monkeypatch.setattr("vector_service.stack_ingest.load_dataset", fake_load_dataset)
    return fake_load_dataset


def test_stack_ingestor_batches_and_persists_metadata(tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _patch_load_dataset(monkeypatch, sample_records)

    metadata_path = tmp_path / "meta.db"
    metadata_store = StackMetadataStore(metadata_path, namespace="test")
    vector_service = DummyVectorService()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        metadata_store=metadata_store,
        vector_service=vector_service,
        dataset_loader=loader,
    )

    processed = ingestor.ingest(resume=False)

    assert processed == 2
    assert len(vector_service.calls) == 4
    assert all(call["metadata"].keys() >= {"repo", "path", "language", "chunk_hash"} for call in vector_service.calls)
    assert all("text" in call["record"] for call in vector_service.calls)

    cur = metadata_store.conn.cursor()
    cur.execute(f"PRAGMA table_info({metadata_store.metadata_table})")
    columns = {row[1] for row in cur.fetchall()}
    assert {"chunk_hash", "vector"}.issubset(columns)
    assert "content" not in columns

    cur.execute(f"SELECT chunk_hash FROM {metadata_store.metadata_table}")
    hashes = {row[0] for row in cur.fetchall()}
    assert all("print" not in chunk_hash for chunk_hash in hashes)

    cur.execute(f"SELECT vector FROM {metadata_store.metadata_table}")
    stored_vectors = [row[0] for row in cur.fetchall()]
    assert all(isinstance(blob, (bytes, bytearray)) for blob in stored_vectors)


def test_stack_ingestor_resume_skips_existing(tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch) -> None:
    loader = _patch_load_dataset(monkeypatch, sample_records)

    metadata_path = tmp_path / "meta.db"
    metadata_store = StackMetadataStore(metadata_path, namespace="resume")
    vector_service = DummyVectorService()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        metadata_store=metadata_store,
        vector_service=vector_service,
        dataset_loader=loader,
    )

    first_processed = ingestor.ingest(resume=False)
    initial_records = len(vector_service.calls)

    second_processed = ingestor.ingest(resume=True)

    assert first_processed == 2
    assert second_processed == 0
    assert len(vector_service.calls) == initial_records

