import logging
import sys
from pathlib import Path

import intent_clusterer as ic


def _make_clusterer(tmp_path: Path) -> ic.IntentClusterer:
    db = ic.ModuleVectorDB(
        index_path=tmp_path / "idx.ann", metadata_path=tmp_path / "idx.json"
    )
    return ic.IntentClusterer(
        db=db,
        menace_id="t",
        local_db_path=tmp_path / "local.db",
        shared_db_path=tmp_path / "shared.db",
    )


def test_load_synergy_groups_logs_failure(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(ic, "governed_embed", lambda text: [0.1, 0.2])
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", None)
    clusterer = _make_clusterer(tmp_path)
    map_file = tmp_path / "sandbox_data" / "module_map.json"
    map_file.parent.mkdir()
    map_file.write_text("{bad json")
    caplog.set_level(logging.WARNING)
    groups = clusterer._load_synergy_groups(tmp_path)
    assert groups == {}
    assert "module_map.json" in caplog.text


def test_index_clusters_logs_failures(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(ic, "governed_embed", lambda text: [0.1, 0.2])
    clusterer = _make_clusterer(tmp_path)
    clusterer.vectors["a.py"] = [0.1, 0.2]  # path-ignore
    groups = {"1": ["a.py"]}  # path-ignore

    class BoomConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            raise RuntimeError("db boom")

    clusterer.conn = BoomConn()

    def fail_add_vector(*args, **kwargs):
        raise RuntimeError("retriever boom")

    clusterer.retriever.add_vector = fail_add_vector  # type: ignore[assignment]

    def fail_rebuild():
        raise RuntimeError("rebuild boom")

    clusterer.db._rebuild_index = fail_rebuild  # type: ignore[attr-defined]

    caplog.set_level(logging.WARNING)
    clusterer._index_clusters(groups)
    text = caplog.text
    assert "db boom" in text
    assert "retriever boom" in text
    assert "rebuild boom" in text


def test_post_init_logs_registration_failure(tmp_path, caplog, monkeypatch):
    monkeypatch.setattr(ic, "governed_embed", lambda text: [0.1, 0.2])
    clusterer = _make_clusterer(tmp_path)

    def fail_register(*args, **kwargs):
        raise RuntimeError("reg boom")

    clusterer.retriever.register_db = fail_register  # type: ignore[assignment]
    caplog.set_level(logging.WARNING)
    clusterer.__post_init__()
    assert "reg boom" in caplog.text


def test_index_modules_logs_retriever_failure(tmp_path, caplog, monkeypatch):
    module = tmp_path / "m.py"  # path-ignore
    module.write_text('"""doc"""\n')
    monkeypatch.setattr(ic, "governed_embed", lambda text: [0.1, 0.2])
    clusterer = _make_clusterer(tmp_path)

    def fail_add_vector(*args, **kwargs):
        raise RuntimeError("retriever boom")

    clusterer.retriever.add_vector = fail_add_vector  # type: ignore[assignment]
    monkeypatch.setattr(ic, "persist_embedding", lambda *a, **k: None)
    caplog.set_level(logging.WARNING)
    clusterer.index_modules([module])
    assert "retriever boom" in caplog.text
