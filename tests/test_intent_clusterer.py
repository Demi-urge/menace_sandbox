import json
import pytest
from pathlib import Path

import embeddable_db_mixin as edm
import intent_vectorizer as iv
import intent_clusterer as ic
import intent_db
from db_router import init_db_router, LOCAL_TABLES


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    def _fake(text: str, model=None) -> list[float]:
        lower = text.lower()
        return [
            float(lower.count("auth")),
            float(lower.count("help")),
            float(lower.count("pay")),
        ]

    monkeypatch.setattr(edm, "governed_embed", _fake)
    monkeypatch.setattr(iv, "governed_embed", _fake)


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
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
    return tmp_path


@pytest.fixture
def indexed_clusterer(sample_repo: Path, tmp_path: Path):
    LOCAL_TABLES.add("intent")
    router = init_db_router(
        "intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db")
    )
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def register_db(self, *args, **kwargs):
            pass

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_modules(sample_repo.glob("*.py"))
    return clusterer, sample_repo


def test_intent_extraction(sample_repo: Path):
    vectorizer = iv.IntentVectorizer()
    text = vectorizer.bundle(sample_repo / "auth.py")
    assert "Authentication module" in text
    assert "handles login" in text
    assert "login" in text


def test_clustering_output(indexed_clusterer):
    clusterer, repo = indexed_clusterer
    clusters = clusterer.cluster_intents(2)
    auth_path = str(repo / "auth.py")
    helper_path = str(repo / "helper.py")
    payment_path = str(repo / "payment.py")
    assert clusters[auth_path] == clusters[helper_path]
    assert clusters[payment_path] != clusters[auth_path]


def test_natural_language_query(indexed_clusterer):
    clusterer, repo = indexed_clusterer
    res = clusterer.find_modules_related_to("authentication help", top_k=1)
    assert res and Path(res[0]["path"]).name == "helper.py"
    assert res[0]["origin"] == "module"
    res2 = clusterer.find_modules_related_to("process payment", top_k=1)
    assert res2 and Path(res2[0]["path"]).name == "payment.py"
    assert res2[0]["origin"] == "module"


def test_synergy_cluster_embeddings_and_query(tmp_path: Path, monkeypatch):
    (tmp_path / "a.py").write_text('"""alpha"""')
    (tmp_path / "b.py").write_text('"""beta"""')

    def _fake(text: str, model=None) -> list[float]:
        lower = text.lower()
        return [float(lower.count("alpha")), float(lower.count("beta"))]

    monkeypatch.setattr(edm, "governed_embed", _fake)

    class DummyGrapher:
        root = tmp_path

        def get_synergy_cluster(self, module_name: str, threshold: float):
            return {"a", "b"}

    monkeypatch.setattr(intent_db, "ModuleSynergyGrapher", lambda: DummyGrapher())

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def register_db(self, *args, **kwargs):
            pass

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_modules([tmp_path / "a.py", tmp_path / "b.py"])
    db.index_synergy_cluster("a", 0.5)

    res = clusterer.find_clusters_related_to("alpha beta", top_k=1)
    assert res and res[0]["path"].startswith("cluster:a")
    assert res[0]["origin"] == "cluster"


def test_module_map_cluster_embeddings(tmp_path: Path, monkeypatch):
    def _fake(text: str, model=None) -> list[float]:
        lower = text.lower()
        return [float(lower.count("alpha")), float(lower.count("beta"))]

    monkeypatch.setattr(edm, "governed_embed", _fake)

    (tmp_path / "a.py").write_text('"""alpha"""')
    (tmp_path / "b.py").write_text('"""beta"""')
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "module_map.json").write_text(json.dumps({"a": 1, "b": 1}))

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = intent_db.IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def register_db(self, *args, **kwargs):
            pass

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_repository(tmp_path)

    res = clusterer.find_clusters_related_to("alpha beta", top_k=1)
    assert res and res[0]["origin"] == "cluster"
    assert res[0]["path"].startswith("cluster:1")

    res2 = clusterer.find_modules_related_to(
        "alpha beta", top_k=3, include_clusters=True
    )
    origins = {r["origin"] for r in res2}
    assert "module" in origins and "cluster" in origins
