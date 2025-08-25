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


def test_fallback_groups_by_prefix_and_import(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "module_synergy_grapher", None)
    clusterer = _make_clusterer(tmp_path)
    a1 = tmp_path / "alpha_one.py"
    a1.write_text("import os\n")
    a2 = tmp_path / "alpha_two.py"
    a2.write_text("import sys\n")
    b1 = tmp_path / "mod_a.py"
    b1.write_text("import json\n")
    b2 = tmp_path / "mod_b.py"
    b2.write_text("import json\n")
    solo = tmp_path / "solo.py"
    solo.write_text("import pickle\n")
    groups = clusterer._load_synergy_groups(tmp_path)
    gsets = [set(map(Path, members)) for members in groups.values()]
    assert any(g == {a1, a2} for g in gsets)
    assert any(g == {b1, b2} for g in gsets)
    assert any(g == {solo} for g in gsets)
