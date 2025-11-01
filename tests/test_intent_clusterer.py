"""Tests for the ``IntentClusterer`` helper.

The tests build a small temporary repository with a few modules so the
``IntentClusterer`` can index them and answer semantic queries.  Embeddings are
deterministic and very small which keeps the tests fast while still exercising
the full indexing and retrieval pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import sys
import types

# Provide a minimal stub for ``sentence_transformers`` to avoid importing heavy
# dependencies such as ``torch`` during test collection.
st_stub = types.ModuleType("sentence_transformers")
st_stub.SentenceTransformer = None
sys.modules.setdefault("sentence_transformers", st_stub)

import menace_sandbox.embeddable_db_mixin as edm  # noqa: E402
import intent_clusterer as ic  # noqa: E402
import intent_db  # noqa: E402
import intent_vectorizer as iv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import pytest  # noqa: E402
from db_router import LOCAL_TABLES, init_db_router  # noqa: E402


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
    monkeypatch.setattr(ic, "governed_embed", _fake)
    monkeypatch.setattr(iv, "SentenceTransformer", None)
    # ``IntentClusterer.index_modules`` writes to ``embeddings.jsonl`` via
    # ``persist_embedding`` which would pollute the repository.  Replace it with
    # a no‑op to keep the workspace clean during tests.
    monkeypatch.setattr(ic, "persist_embedding", lambda *a, **k: None)
    # Use a deterministic summariser to avoid network access
    monkeypatch.setattr(
        ic, "summarise_texts", lambda texts, **_: "auth helper summary"
    )


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    """Create a temporary repository with several modules.

    Two modules relate to authentication while ``payment.py`` handles payments.  # path-ignore
    ``authpay.py`` mixes both intents.  A ``module_map.json`` file places  # path-ignore
    ``auth.py`` and ``helper.py`` into the same synthetic synergy cluster so the  # path-ignore
    cluster search API can be exercised.
    """

    files = {
        "auth.py": (  # path-ignore
            '"""Authentication module"""\n'
            "# handles login\n\n"
            "def login():\n"
            '    """login user"""\n'
            "    pass\n"
        ),
        "helper.py": (  # path-ignore
            '"""Authentication helper"""\n'
            "# provides help\n\n"
            "def assist():\n"
            '    """assist auth"""\n'
            "    pass\n"
        ),
        "authpay.py": (  # path-ignore
            '"""Authentication and payment handler"""\n'
            "# handles auth and pay\n\n"
            "def authpay():\n"
            '    """auth pay"""\n'
            "    pass\n"
        ),
        "payment.py": (  # path-ignore
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
    # Map auth.py and helper.py into the same cluster (id 1); payment.py is  # path-ignore
    # assigned to a different cluster to make the distinction clear.
    (data_dir / "module_map.json").write_text(
        json.dumps({"auth": 1, "helper": 1, "payment": 2})
    )
    return tmp_path


def test_derive_cluster_label_uses_summariser(monkeypatch):
    """``derive_cluster_label`` should prefer the summariser when available."""

    monkeypatch.setattr(
        ic, "summarise_texts", lambda texts, **_: "auth summary phrase"
    )
    label, summary = ic.derive_cluster_label(["auth stuff", "more auth"])
    assert label == "auth summary phrase"
    assert summary == "auth summary phrase"


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


class DummyVectorService:
    """Minimal stand-in for the vector service layer.

    The service records any vectors passed to :meth:`add_vector` and performs a
    simple dot-product search.  A ``search_called`` flag allows tests to verify
    that queries are delegated to this service.
    """

    def __init__(self) -> None:
        self.items: List[dict] = []
        self.search_called = False

    def add_vector(self, vector: Iterable[float], metadata: dict) -> None:
        self.items.append({"vector": list(vector), "metadata": dict(metadata)})

    def search(self, vector: Iterable[float], top_k: int = 10) -> List[dict]:
        self.search_called = True

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
    clusterer.index_modules(list(sample_repo.glob("*.py")))  # path-ignore
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
        str(sample_repo / "auth.py"),  # path-ignore
        str(sample_repo / "helper.py"),  # path-ignore
        str(sample_repo / "payment.py"),  # path-ignore
    }
    assert expected.issubset(paths)
    assert all(row[1] for row in rows)  # vectors are persisted


def test_find_modules_related_to_prompts(clusterer: ic.IntentClusterer, sample_repo: Path) -> None:
    """Queries should surface the expected module paths."""

    clusterer.index_repository(sample_repo)

    res = clusterer.find_modules_related_to("authentication help", top_k=2)
    assert any(Path(r.path).name == "helper.py" for r in res if r.path)  # path-ignore

    res = clusterer.find_modules_related_to("process payment", top_k=2)
    assert any(Path(r.path).name == "payment.py" for r in res if r.path)  # path-ignore


def test_vector_service_storage_and_query(sample_repo: Path, tmp_path: Path) -> None:
    """Vectors should be stored in and retrieved via ``vector_service``."""

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "vs.db"), str(tmp_path / "vs.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "vs.db",
        vector_index_path=tmp_path / "vs.index",
        router=router,
    )
    vs = DummyVectorService()
    clusterer = ic.IntentClusterer(intent_db=db, vector_service=vs)

    clusterer.index_repository(sample_repo)
    # ``add_vector`` should have been called for each module (and clusters)
    assert vs.items

    res = clusterer.find_modules_related_to("auth help", top_k=2)
    assert vs.search_called
    assert any(Path(r.path).name == "helper.py" for r in res if r.path)  # path-ignore


def test_cluster_lookup_uses_synergy_groups(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """Synthetic synergy groups should be searchable as clusters."""

    clusterer.index_repository(sample_repo)
    res = clusterer.find_clusters_related_to("auth help", top_k=5)
    cluster_items = res
    assert cluster_items
    assert cluster_items[0].path.startswith("cluster:1")
    assert cluster_items[0].label and "auth" in cluster_items[0].label.lower()
    assert cluster_items[0].summary == "auth helper summary"
    assert cluster_items[0].category == "authentication"
    # ``cluster_label`` should expose the persisted label and summary
    label, summary = clusterer.cluster_label(1)
    assert label == "auth helper summary"
    assert summary == "auth helper summary"


def test_cluster_label_updates_persisted_metadata(
    clustered_clusterer: ic.IntentClusterer, monkeypatch
) -> None:
    """``cluster_label`` should persist derived labels and summaries."""

    clusterer = clustered_clusterer
    entry = "cluster:1"
    # start with outdated metadata: label only
    meta = {"label": "old", "intent_text": "auth helper"}
    row = clusterer.conn.execute(
        "SELECT vector FROM intent_embeddings WHERE module_path = ?",
        (entry,),
    ).fetchone()
    vec = row[0] if row else None
    clusterer.conn.execute(
        "REPLACE INTO intent_embeddings (module_path, vector, metadata) VALUES (?, ?, ?)",
        (entry, vec, json.dumps(meta)),
    )
    clusterer.conn.commit()
    clusterer.db._metadata[entry] = dict(meta)  # type: ignore[attr-defined]
    clusterer._cluster_cache.pop(1, None)
    assert clusterer._get_cluster_summary(1) is None
    t, _ = clusterer.get_cluster_intents(1)
    assert t
    clusterer._cluster_cache.pop(1, None)

    monkeypatch.setattr(
        ic,
        "derive_cluster_label",
        lambda *a, **k: ("new label", "new summary"),
    )
    rebuild_called = {"r": 0, "s": 0}
    monkeypatch.setattr(
        clusterer.db,
        "_rebuild_index",
        lambda: rebuild_called.__setitem__("r", rebuild_called["r"] + 1),
    )
    monkeypatch.setattr(
        clusterer.db,
        "save_index",
        lambda: rebuild_called.__setitem__("s", rebuild_called["s"] + 1),
    )

    label, summary = clusterer.cluster_label(1)
    assert label == "new label"
    assert summary == "new summary"
    row = clusterer.conn.execute(
        "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
        (entry,),
    ).fetchone()
    data = json.loads(row[0])
    assert data.get("label") == "new label"
    assert data.get("summary") == "new summary"
    assert clusterer.db._metadata[entry]["label"] == "new label"  # type: ignore[attr-defined]
    assert clusterer.db._metadata[entry]["summary"] == "new summary"  # type: ignore[attr-defined]
    assert rebuild_called["r"] == 1 and rebuild_called["s"] == 1


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
    auth = str(sample_repo / "auth.py")  # path-ignore
    helper = str(sample_repo / "helper.py")  # path-ignore
    assert any(
        auth in ci["metadata"]["members"] and helper in ci["metadata"]["members"]
        for ci in cluster_items
    )
    # ``intent_text``, ``label`` and ``summary`` should be exposed via the retriever metadata
    assert all(
        ci["metadata"].get("label")
        and ci["metadata"].get("intent_text")
        and ci["metadata"].get("summary") is not None
        and ci["metadata"].get("category")
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
    assert meta.get("summary") is not None
    assert meta.get("category")
    assert set(meta.get("members", [])) == set(cluster_items[0]["metadata"]["members"])
    # Labels should also be retrievable via ``cluster_label``
    cid = cluster_items[0]["metadata"].get("cluster_id")
    if cid is not None:
        lbl, summ = clustered_clusterer.cluster_label(cid)
        assert lbl
        assert summ == "" or isinstance(summ, str)


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
    assert res[0].category == "authentication"

    # High threshold filters out even perfect matches
    assert clustered_clusterer.query("authentication help", threshold=1.1) == []

    mods = clustered_clusterer.find_modules_related_to(
        "authentication help", top_k=5, include_clusters=True
    )
    mod_entry = next(m for m in mods if m.path)
    assert mod_entry.cluster_ids
    assert mod_entry.category == "authentication"
    cluster_entry = next(m for m in mods if m.origin == "cluster")
    assert cluster_entry.cluster_ids
    assert cluster_entry.category == "authentication"

    clusters = clustered_clusterer.find_clusters_related_to("authentication help", top_k=5)
    assert clusters and clusters[0].origin == "cluster" and clusters[0].category == "authentication"


def test_mixed_intent_module_gets_multiple_cluster_ids(
    sample_repo: Path, tmp_path: Path
) -> None:
    """Modules covering several intents should surface all cluster IDs."""

    db = ic.ModuleVectorDB(
        index_path=tmp_path / "mix.ann", metadata_path=tmp_path / "mix.json"
    )
    clusterer = ic.IntentClusterer(db=db, retriever=DummyRetriever())
    clusterer.index_modules(list(sample_repo.glob("*.py")))  # path-ignore
    clusterer.cluster_intents(2, threshold=0.0)

    mixed_path = str(sample_repo / "authpay.py")  # path-ignore
    assert len(clusterer.clusters[mixed_path]) > 1

    res = clusterer.query("auth pay", top_k=5, include_clusters=False)
    match = next(m for m in res if m.path and Path(m.path).name == "authpay.py")  # path-ignore
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

    auth = str(sample_repo / "auth.py")  # path-ignore
    helper = str(sample_repo / "helper.py")  # path-ignore
    pay = str(sample_repo / "payment.py")  # path-ignore
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


def _fetch_entry(clusterer: ic.IntentClusterer, path: Path) -> tuple[list, dict]:
    row = clusterer.conn.execute(
        "SELECT vector, metadata FROM intent_embeddings WHERE module_path = ?",
        (str(path),),
    ).fetchone()
    vec = pickle.loads(row[0]) if row and row[0] else []
    meta = json.loads(row[1]) if row and row[1] else {}
    return vec, meta


def test_incremental_update_and_prune(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    clusterer.index_repository(sample_repo)

    auth_path = sample_repo / "auth.py"  # path-ignore
    helper_path = sample_repo / "helper.py"  # path-ignore
    pay_path = sample_repo / "payment.py"  # path-ignore

    auth_vec1, auth_meta1 = _fetch_entry(clusterer, auth_path)
    helper_vec1, helper_meta1 = _fetch_entry(clusterer, helper_path)

    helper_path.write_text(
        helper_path.read_text().replace("helper", "helper pay")
    )
    pay_path.unlink()

    clusterer.index_repository(sample_repo)

    auth_vec2, auth_meta2 = _fetch_entry(clusterer, auth_path)
    helper_vec2, helper_meta2 = _fetch_entry(clusterer, helper_path)

    assert auth_meta2["mtime"] == auth_meta1["mtime"]
    assert auth_vec2 == auth_vec1
    assert helper_meta2["mtime"] != helper_meta1["mtime"]
    assert helper_vec2 != helper_vec1

    rows = clusterer.conn.execute(
        "SELECT module_path FROM intent_embeddings"
    ).fetchall()
    paths = {row[0] for row in rows}
    assert str(pay_path) not in paths

    retr_paths = {item["metadata"].get("path") for item in clusterer.retriever.items}
    assert "cluster:2" not in retr_paths


def test_index_modules_records_vectors(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """``index_modules`` should store vectors and register paths."""

    paths = list(sample_repo.glob("*.py"))  # path-ignore
    clusterer.index_modules(paths)

    for path in paths:
        spath = str(path)
        assert spath in clusterer.module_ids
        assert spath in clusterer.vectors

    stored = {item["metadata"].get("path") for item in clusterer.retriever.items}
    assert {str(p) for p in paths} <= stored


def test_cluster_intents_dynamic_count(
    clusterer: ic.IntentClusterer, sample_repo: Path, monkeypatch
) -> None:
    """When ``n_clusters`` is ``None`` the optimal count is used."""

    clusterer.index_modules(list(sample_repo.glob("*.py")))  # path-ignore

    called: dict[str, bool] = {}

    def _fake_optimal(self, vectors: List[List[float]]) -> int:
        called["yes"] = True
        return 2

    monkeypatch.setattr(ic.IntentClusterer, "_optimal_cluster_count", _fake_optimal)

    clusters = clusterer.cluster_intents(n_clusters=None, threshold=0.0)
    assert called.get("yes") is True
    unique = {cid for ids in clusters.values() for cid in ids}
    assert len(unique) == 2


def test_search_related_returns_modules_and_clusters(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """``_search_related`` should surface both module and cluster entries."""

    clusterer.index_modules(list(sample_repo.glob("*.py")))  # path-ignore
    clusterer.cluster_intents(2, threshold=0.0)
    results = clusterer._search_related("authentication help", top_k=5)

    assert any(Path(r.path).name == "auth.py" for r in results if r.path)  # path-ignore
    assert any(r.origin == "cluster" for r in results)


def test_find_modules_related_to_filters_clusters(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """``find_modules_related_to`` should optionally include clusters."""

    clusterer.index_repository(sample_repo)

    mods_only = clusterer.find_modules_related_to("authentication help", top_k=5)
    assert all(r.origin != "cluster" for r in mods_only)

    with_clusters = clusterer.find_modules_related_to(
        "authentication help", top_k=5, include_clusters=True
    )
    assert any(r.origin == "cluster" for r in with_clusters)


def test_fresh_instance_queries_clusters(sample_repo: Path, tmp_path: Path) -> None:
    """A new ``IntentClusterer`` should use stored cluster mappings."""

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "fresh.db"), str(tmp_path / "fresh.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "fresh.db",
        vector_index_path=tmp_path / "fresh.index",
        router=router,
    )
    retr = DummyRetriever()
    clusterer = ic.IntentClusterer(intent_db=db, retriever=retr)
    clusterer.index_repository(sample_repo)
    clusterer.cluster_intents(2, threshold=0.0)

    fresh = ic.IntentClusterer(intent_db=db, retriever=retr)
    res = fresh.find_clusters_related_to("auth help", top_k=5)
    assert res and res[0].path.startswith("cluster:1")


def test_extract_intent_text_parses_docstrings_and_comments(tmp_path: Path) -> None:
    """Ensure ``extract_intent_text`` collects docs, names and comments."""

    module = tmp_path / "sample.py"  # path-ignore
    module.write_text(
        '"""Top level doc"""\n'
        "# pre foo\n\n"
        "def foo():\n"
        '    """Foo docs"""\n'
        "    pass\n\n"
        "# pre bar class\n"
        "class Bar:\n"
        '    """Bar docs"""\n'
        "    pass\n"
    )

    text = ic.extract_intent_text(module)
    assert "Top level doc" in text
    assert "Foo docs" in text
    assert "Bar docs" in text
    assert "foo" in text
    assert "Bar" in text
    assert "pre foo" in text
    assert "pre bar class" in text


def test_index_modules_and_repository_populate_ids_vectors_db(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """``index_modules`` and ``index_repository`` should populate mappings."""

    paths = list(sample_repo.glob("*.py"))  # path-ignore
    expected = {str(p) for p in paths}

    # ``index_modules`` populates in-memory maps
    clusterer.index_modules(paths)
    assert expected <= set(clusterer.module_ids)
    assert expected <= set(clusterer.vectors)

    # database table not yet populated
    count = clusterer.conn.execute(
        "SELECT COUNT(*) FROM intent_embeddings"
    ).fetchone()[0]
    assert count == 0

    # ``index_repository`` performs incremental update and stores rows
    clusterer.index_repository(sample_repo)
    rows = {
        row[0]
        for row in clusterer.conn.execute(
            "SELECT module_path FROM intent_embeddings "
            "WHERE module_path NOT LIKE 'cluster:%'"
        ).fetchall()
    }
    assert rows == expected


def test_index_clusters_creates_metadata_and_embeddings(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """``_index_clusters`` should aggregate vectors and persist metadata."""

    paths = list(sample_repo.glob("*.py"))  # path-ignore
    clusterer.index_modules(paths)

    members = [str(sample_repo / "auth.py"), str(sample_repo / "helper.py")]  # path-ignore
    clusterer._index_clusters({"1": members})

    row = clusterer.conn.execute(
        "SELECT vector, metadata FROM intent_embeddings WHERE module_path = ?",
        ("cluster:1",),
    ).fetchone()
    assert row is not None
    vec = list(pickle.loads(row[0]))
    assert vec == pytest.approx([1.5, 0.0, 1.0])
    meta = json.loads(row[1])
    assert meta["members"] == sorted(members)
    assert meta["cluster_id"] == 1
    assert meta["label"] == "auth helper summary"
    assert any(
        item["metadata"].get("path") == "cluster:1" for item in clusterer.retriever.items
    )


def test_search_helpers_return_expected_matches(
    clusterer: ic.IntentClusterer, sample_repo: Path
) -> None:
    """Search helpers should surface relevant modules and clusters."""

    clusterer.index_repository(sample_repo)

    query = "auth help help help help"
    results = clusterer._search_related(query, top_k=5)
    assert results and results[0].path.endswith("helper.py")  # path-ignore
    assert any(r.origin == "cluster" for r in results)

    mods = clusterer.find_modules_related_to(query, top_k=2)
    assert mods and mods[0].path.endswith("helper.py")  # path-ignore
    assert all(m.origin != "cluster" for m in mods)

    clusters = clusterer.find_clusters_related_to("auth help", top_k=2)
    assert clusters and clusters[0].cluster_ids == [1]
