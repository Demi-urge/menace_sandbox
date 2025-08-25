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
    """Create a temporary repository with several modules.

    Two modules relate to authentication while ``payment.py`` handles payments.
    ``authpay.py`` mixes both intents.  A ``module_map.json`` file places
    ``auth.py`` and ``helper.py`` into the same synthetic synergy cluster so the
    cluster search API can be exercised.
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
        "authpay.py": (
            '"""Authentication and payment handler"""\n'
            "# handles auth and pay\n\n"
            "def authpay():\n"
            '    """auth pay"""\n'
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
        item = {"vector": list(vector), "metadata": dict(metadata)}
        path = item["metadata"].get("path")
        if path:
            for idx, existing in enumerate(self.items):
                if existing["metadata"].get("path") == path:
                    self.items[idx] = item
                    break
            else:
                self.items.append(item)
        else:
            self.items.append(item)

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


@pytest.fixture
def clustered_clusterer(sample_repo: Path, tmp_path: Path) -> ic.IntentClusterer:
    """Clustered ``IntentClusterer`` with modules in multiple clusters."""

    db = ic.ModuleVectorDB(
        index_path=tmp_path / "idx.ann", metadata_path=tmp_path / "idx.json"
    )
    clusterer = ic.IntentClusterer(db=db, retriever=DummyRetriever())
    # ``index_modules`` avoids loading synergy groups so KMeans clustering is used
    clusterer.index_modules(list(sample_repo.glob("*.py")))
    # ``threshold=0`` ensures each module belongs to all clusters
    clusterer.cluster_intents(2, threshold=0.0)
    return clusterer


def test_index_repository_stores_embeddings(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
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


def test_cluster_lookup_uses_synergy_groups(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """Synthetic synergy groups should be searchable as clusters."""

    clusterer.index_repository(sample_repo)
    res = clusterer.find_clusters_related_to("auth help", top_k=5)
    cluster_items = [r for r in res if r.get("origin") == "cluster"]
    assert cluster_items
    assert cluster_items[0]["path"].startswith("cluster:1")
    assert "label" in cluster_items[0] and "auth" in cluster_items[0]["label"].lower()
    # ``cluster_label`` should expose the persisted label
    label = clusterer.cluster_label(1)
    assert label and "auth" in label.lower()


def test_cluster_intents_adds_cluster_metadata(
    clustered_clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """Clustering should persist rich metadata for each cluster."""

    # At least one module should belong to more than one cluster
    assert any(len(cids) > 1 for cids in clustered_clusterer.clusters.values())

    cluster_items = [
        item
        for item in clustered_clusterer.retriever.items
        if item["metadata"].get("kind") == "cluster"
    ]
    assert cluster_items
    auth = str(sample_repo / "auth.py")
    helper = str(sample_repo / "helper.py")
    assert any(
        auth in ci["metadata"]["members"] and helper in ci["metadata"]["members"]
        for ci in cluster_items
    )
    # ``intent_text`` and ``label`` should be exposed via the retriever metadata
    assert all(
        ci["metadata"].get("label") and ci["metadata"].get("intent_text")
        for ci in cluster_items
    )
    # Metadata should also be persisted in the SQLite table
    entry = cluster_items[0]["metadata"]["path"]
    row = clustered_clusterer.conn.execute(
        "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
        (entry,),
    ).fetchone()
    meta = json.loads(row[0])
    assert meta.get("intent_text")
    assert set(meta.get("members", [])) == set(cluster_items[0]["metadata"]["members"])
    # Labels should also be retrievable via ``cluster_label``
    cid = cluster_items[0]["metadata"].get("cluster_id")
    if cid is not None:
        lbl = clustered_clusterer.cluster_label(cid)
        assert lbl


def test_get_cluster_intents_returns_summary_and_vector(
    clustered_clusterer: ic.IntentClusterer,
) -> None:
    """``get_cluster_intents`` should yield non-empty text and vectors."""

    cid = clustered_clusterer.clusters[next(iter(clustered_clusterer.clusters))][0]
    text, vec = clustered_clusterer.get_cluster_intents(cid)
    assert text and vec


def test_query_and_find_helpers_respect_thresholds(
    clustered_clusterer: ic.IntentClusterer,
) -> None:
    """Query and helper functions should honour thresholds and surface clusters."""

    res = clustered_clusterer.query("authentication help", threshold=0.1)
    assert res and res[0].path and res[0].cluster_ids

    # High threshold filters out even perfect matches
    assert clustered_clusterer.query("authentication help", threshold=0.99) == []

    mods = clustered_clusterer.find_modules_related_to(
        "authentication help", top_k=5, include_clusters=True
    )
    mod_entry = next(m for m in mods if m.get("path"))
    assert mod_entry.get("cluster_ids")
    cluster_entry = next(m for m in mods if m.get("origin") == "cluster")
    assert cluster_entry.get("cluster_ids")

    clusters = clustered_clusterer.find_clusters_related_to("authentication help", top_k=5)
    assert clusters and clusters[0]["origin"] == "cluster"


def test_mixed_intent_module_gets_multiple_cluster_ids(
    sample_repo: Path, tmp_path: Path
) -> None:
    """Modules covering several intents should surface all cluster IDs."""

    db = ic.ModuleVectorDB(
        index_path=tmp_path / "mix.ann", metadata_path=tmp_path / "mix.json"
    )
    clusterer = ic.IntentClusterer(db=db, retriever=DummyRetriever())
    clusterer.index_modules(list(sample_repo.glob("*.py")))
    clusterer.cluster_intents(2, threshold=0.0)

    mixed_path = str(sample_repo / "authpay.py")
    assert len(clusterer.clusters[mixed_path]) > 1

    res = clusterer.query("auth pay", top_k=5, include_clusters=False)
    match = next(m for m in res if m.path and Path(m.path).name == "authpay.py")
    assert len(match.cluster_ids) > 1


def test_index_repository_updates_cluster_membership(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    clusterer.index_repository(sample_repo)
    retr = clusterer.retriever

    def members(cid: int) -> List[str]:
        path = f"cluster:{cid}"
        for item in retr.items:
            if item["metadata"].get("path") == path:
                return list(item["metadata"].get("members", []))
        return []

    auth = str(sample_repo / "auth.py")
    helper = str(sample_repo / "helper.py")
    pay = str(sample_repo / "payment.py")
    assert set(members(1)) == {auth, helper}
    assert set(members(2)) == {pay}

    mapping = json.loads(
        (sample_repo / "sandbox_data" / "module_map.json").read_text()
    )
    mapping["helper"] = 2
    (sample_repo / "sandbox_data" / "module_map.json").write_text(json.dumps(mapping))

    clusterer.index_repository(sample_repo)
    assert set(members(1)) == {auth}
    assert set(members(2)) == {pay, helper}
