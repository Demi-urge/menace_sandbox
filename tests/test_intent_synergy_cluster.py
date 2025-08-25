import embeddable_db_mixin as edm
import intent_db
import intent_clusterer as ic
from intent_db import IntentDB
from intent_clusterer import IntentClusterer
from db_router import init_db_router, LOCAL_TABLES
from pathlib import Path
import pytest


def _fake(text, model=None):
    lower = text.lower()
    return [float(lower.count("alpha")), float(lower.count("beta"))]


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(edm, "governed_embed", _fake)


def test_synergy_cluster_index_and_search(tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text('"""alpha"""')
    (tmp_path / "b.py").write_text('"""beta"""')

    class DummyGrapher:
        root = tmp_path

        def get_synergy_cluster(self, module_name, threshold):
            return {"a", "b"}

    monkeypatch.setattr(intent_db, "ModuleSynergyGrapher", lambda: DummyGrapher())

    LOCAL_TABLES.add("intent")
    router = init_db_router("intent", str(tmp_path / "intent.db"), str(tmp_path / "intent.db"))
    db = IntentDB(
        path=tmp_path / "intent.db",
        vector_index_path=tmp_path / "intent.index",
        router=router,
    )

    class DummyRetriever:
        def register_db(self, *args, **kwargs):
            pass

    clusterer = IntentClusterer(
        intent_db=db,
        retriever=DummyRetriever(),
        vector_service=object(),
    )
    clusterer.index_modules([tmp_path / "a.py", tmp_path / "b.py"])
    db.index_synergy_cluster("a", 0.5)

    res = clusterer.find_modules_related_to("alpha beta", top_k=1)
    assert res and res[0]["path"].startswith("cluster:a")
