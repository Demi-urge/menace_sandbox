"""Tests for the ``IntentClusterer`` helper.

The tests build a small temporary repository with a few modules so the
``IntentClusterer`` can index them and answer semantic queries.  Embeddings are
deterministic and very small which keeps the tests fast while still exercising
the full indexing and retrieval pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import embeddable_db_mixin as edm
import intent_clusterer as ic
import intent_db
import intent_vectorizer as iv
import json
import pytest
from db_router import LOCAL_TABLES, init_db_router


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch) -> None:
    """Provide a predictable embedding for both modules and queries.

    The embedding is a simple 3‑dimensional vector counting occurrences of the
    words ``auth``, ``pay`` and ``help``.  This makes similarity checks trivial
    and removes the dependency on external embedding services.
    """

    def _fake(text: str, model: str | None = None) -> List[float]:
        lower = text.lower()
        return [
            float(lower.count("auth")),
            float(lower.count("pay")),
            float(lower.count("help")),
        ]

    monkeypatch.setattr(edm, "governed_embed", _fake)
    monkeypatch.setattr(iv, "governed_embed", _fake)
    monkeypatch.setattr(iv, "SentenceTransformer", None)
    # ``IntentClusterer.index_modules`` writes to ``embeddings.jsonl`` via
    # ``persist_embedding`` which would pollute the repository.  Replace it with
    # a no‑op to keep the workspace clean during tests.
    monkeypatch.setattr(ic, "persist_embedding", lambda *a, **k: None)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with three modules.

    Two modules relate to authentication while ``payment.py`` handles payments.
    A ``module_map.json`` file places ``auth.py`` and ``helper.py`` into the
    same synthetic synergy cluster so the cluster search API can be exercised.
    """

    files = {
        "auth.py": (
            '"""Authentication module"""\n'
            "# handles login\n\n"
            "def login():\n"
            '    """login user"""\n'
            "    pass\n"
        ),
        "helper.py": (
            '"""Authentication helper"""\n'
            "# provides help\n\n"
            "def assist():\n"
            '    """assist auth"""\n'
            "    pass\n"
        ),
        "payment.py": (
            '"""Payment processor"""\n'
            "# handles payments\n\n"
            "def pay():\n"
            '    """process pay"""\n'
            "    pass\n"
        ),
    }
    for name, content in files.items():
        (tmp_path / name).write_text(content)

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    # Map auth.py and helper.py into the same cluster (id 1); payment.py is
    # assigned to a different cluster to make the distinction clear.
    (data_dir / "module_map.json").write_text(
        json.dumps({"auth": 1, "helper": 1, "payment": 2})
    )
    return tmp_path


class DummyRetriever:
    """In‑memory vector store used by the tests.

    ``IntentClusterer`` normally interacts with ``UniversalRetriever``.  The
    dummy replacement stores vectors in a list and performs a basic dot product
    search which is sufficient for these tests.
    """

    def __init__(self) -> None:
        self.items: List[dict] = []

    def register_db(self, *args, **kwargs) -> None:  # pragma: no cover - trivial
        pass

    def add_vector(self, vector: Iterable[float], metadata: dict) -> None:
        self.items.append({"vector": list(vector), "metadata": dict(metadata)})

    def search(self, vector: Iterable[float], top_k: int = 10) -> List[dict]:
        def score(item: dict) -> float:
            vec = item["vector"]
            return sum(a * b for a, b in zip(vector, vec))

        return sorted(self.items, key=score, reverse=True)[:top_k]


@pytest.fixture
def clusterer(sample_repo: Path, tmp_path: Path) -> ic.IntentClusterer:
    """Return a fresh ``IntentClusterer`` instance for each test."""

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )
    return ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())


def test_index_repository_stores_embeddings(clusterer: ic.IntentClusterer, sample_repo: Path) -> None:
    """Ensure ``index_repository`` writes vectors for each module."""

    clusterer.index_repository(sample_repo)
    cur = clusterer.conn.execute("SELECT module_path, vector FROM intent_embeddings")
    rows = cur.fetchall()
    paths = {row[0] for row in rows}
    expected = {
        str(sample_repo / "auth.py"),
        str(sample_repo / "helper.py"),
        str(sample_repo / "payment.py"),
    }
    assert expected.issubset(paths)
    assert all(row[1] for row in rows)  # vectors are persisted


def test_find_modules_related_to_prompts(clusterer: ic.IntentClusterer, sample_repo: Path) -> None:
    """Queries should surface the expected module paths."""

    clusterer.index_repository(sample_repo)

    res = clusterer.find_modules_related_to("authentication help", top_k=1)
    assert res and Path(res[0]["path"]).name == "helper.py"

    res = clusterer.find_modules_related_to("process payment", top_k=1)
    assert res and Path(res[0]["path"]).name == "payment.py"


def test_cluster_lookup_uses_synergy_groups(clusterer: ic.IntentClusterer, sample_repo: Path) -> None:
    """Synthetic synergy groups should be searchable as clusters."""

    clusterer.index_repository(sample_repo)
    res = clusterer.find_clusters_related_to("auth help", top_k=1)
    assert res and res[0]["origin"] == "cluster"
    assert res[0]["path"].startswith("cluster:1")
    assert "label" in res[0] and "auth" in res[0]["label"].lower()

