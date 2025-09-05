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


def test_fallback_groups_by_dependency_graph(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", None)
    clusterer = _make_clusterer(tmp_path)
    a = tmp_path / "a.py"  # path-ignore
    a.write_text("import b\n")
    b = tmp_path / "b.py"  # path-ignore
    b.write_text("\n")
    solo = tmp_path / "solo.py"  # path-ignore
    solo.write_text("\n")
    groups = clusterer._load_synergy_groups(tmp_path)
    gsets = [set(map(Path, members)) for members in groups.values()]
    assert any(g == {a, b} for g in gsets)
    assert any(g == {solo} for g in gsets)
