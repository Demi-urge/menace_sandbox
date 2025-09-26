from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Mapping
import hashlib

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
        self.vector_store = None

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
        vector = [float(len(text)), float(text.count("\n") + 1)]
        store = getattr(self, "vector_store", None)
        if store is not None:
            try:
                store.add(kind.lower(), record_id, vector, origin_db=origin_db, metadata=metadata)
            except Exception:
                pass
        return vector


class DummyVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.records: List[tuple[str, str, List[float]]] = []

    def add(
        self,
        kind: str,
        record_id: str,
        vector: List[float],
        *,
        origin_db: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.records.append((kind, record_id, list(vector)))


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


def test_stack_ingestor_filters_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    python_sample = {
        "language": "python",
        "repo_name": "stack/project",
        "path": "main.py",
        "content": "line 1\nline 2\nline 3\nline 4 extra",
    }
    other_samples = [
        {
            "language": "java",
            "repo_name": "stack/project",
            "path": "Main.java",
            "content": "public class Main {}",
        },
        {
            "language": "python",
            "repo_name": "stack/project",
            "path": "empty.py",
            "content": "",  # skipped because content is blank
        },
    ]
    samples = [python_sample, *other_samples]
    loader = _patch_load_dataset(monkeypatch, samples)

    metadata_path = tmp_path / "stack-meta.db"
    metadata_store = StackMetadataStore(metadata_path, namespace="filters")
    vector_service = DummyVectorService()
    ingestor = StackIngestor(
        languages=("python",),
        max_lines=3,
        max_bytes=40,
        chunk_lines=2,
        batch_size=2,
        metadata_store=metadata_store,
        vector_service=vector_service,
        dataset_loader=loader,
    )

    processed = ingestor.ingest(resume=False)

    assert processed == 1
    assert len(vector_service.calls) == 2  # two chunks after truncation
    assert {call["record"]["language"] for call in vector_service.calls} == {"python"}
    assert all(len(call["record"]["text"].encode("utf-8")) <= 40 for call in vector_service.calls)

    chunk_lookup = {
        call["record_id"]: call["record"]["text"]
        for call in vector_service.calls
    }

    cur = metadata_store.conn.cursor()
    cur.execute(
        f"SELECT embedding_id, chunk_hash, file_lines, chunk_lines FROM {metadata_store.metadata_table}"
    )
    rows = cur.fetchall()
    assert len(rows) == len(chunk_lookup)
    for embedding_id, chunk_hash, file_lines, chunk_lines in rows:
        text = chunk_lookup[embedding_id]
        assert hashlib.sha256(text.encode("utf-8")).hexdigest() == chunk_hash
        assert file_lines <= 3
        assert 0 < chunk_lines <= 2

    cur.execute(
        f"SELECT file_id, language FROM {metadata_store.progress_table}"
    )
    progress = cur.fetchall()
    assert progress == [("stack/project:main.py", "python")]


def test_stack_ingestor_reuses_supplied_vector_store(
    tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _patch_load_dataset(monkeypatch, sample_records)

    store = DummyVectorStore(tmp_path / "custom.index")
    vector_service = DummyVectorService()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        vector_store=store,
        vector_service=vector_service,
        dataset_loader=loader,
    )

    processed = ingestor.ingest(resume=False)

    assert processed == 2
    assert ingestor.vector_store is store
    assert vector_service.vector_store is store
    assert len(store.records) == 4


def test_stack_ingestor_honours_index_path(
    tmp_path: Path, sample_records, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader = _patch_load_dataset(monkeypatch, sample_records)

    created: dict[str, object] = {}

    class StubStore(DummyVectorStore):
        pass

    def fake_get_stack_store():
        return None

    def fake_create_store(
        dim: int,
        path: Path,
        *,
        backend: str | None = None,
        metric: str = "angular",
    ):
        created["args"] = (dim, Path(path), backend, metric)
        return StubStore(Path(path))

    monkeypatch.setattr("vector_service.stack_ingest.get_stack_vector_store", fake_get_stack_store)
    monkeypatch.setattr("vector_service.stack_ingest.create_vector_store", fake_create_store)

    custom_index = tmp_path / "stack.index"
    vector_service = DummyVectorService()
    ingestor = StackIngestor(
        languages=("python",),
        chunk_lines=1,
        batch_size=2,
        index_path=custom_index,
        vector_service=vector_service,
        dataset_loader=loader,
    )

    processed = ingestor.ingest(resume=False)

    assert processed == 2
    assert "args" in created
    assert created["args"][1] == custom_index
    assert ingestor.vector_store is not None
    assert Path(ingestor.vector_store.path) == custom_index
    assert vector_service.vector_store is ingestor.vector_store
    assert len(getattr(ingestor.vector_store, "records", [])) == 4

