import pytest
from pathlib import Path

import json

import embeddable_db_mixin as edm
import intent_db
import intent_clusterer as ic
from db_router import init_db_router, LOCAL_TABLES


def _fake(text: str, model=None) -> list[float]:
    lower = text.lower()
    return [float(lower.count("alpha")), float(lower.count("beta"))]


@pytest.fixture(autouse=True)
def patch_embed(monkeypatch):
    monkeypatch.setattr(edm, "governed_embed", _fake)


def test_synergy_cluster_embeddings_and_query(tmp_path: Path, monkeypatch):
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
        def __init__(self):
            self.items = []

        def register_db(self, *args, **kwargs):
            pass

        def add_vector(self, vector, metadata):
            self.items.append({"vector": list(vector), "metadata": dict(metadata)})

        def search(self, vector, top_k=10):
            return self.items

    clusterer = ic.IntentClusterer(intent_db=db, retriever=DummyRetriever())
    clusterer.index_repository(tmp_path)

    res = clusterer.find_clusters_related_to("alpha beta", top_k=5)
    assert res and res[0]["origin"] == "cluster"
    members = {str(tmp_path / "a.py"), str(tmp_path / "b.py")}
    assert set(res[0]["members"]) == members
    assert res[0]["intent_text"]
    row = clusterer.conn.execute(
        "SELECT metadata FROM intent_embeddings WHERE module_path = ?",
        (res[0]["path"],),
    ).fetchone()
    meta = json.loads(row[0])
    assert meta.get("intent_text")
    assert set(meta.get("members", [])) == members
