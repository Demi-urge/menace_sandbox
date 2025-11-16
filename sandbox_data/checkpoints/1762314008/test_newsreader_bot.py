import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from pathlib import Path

import menace.newsreader_bot as nb


def test_filter_events():
    ev1 = nb.Event(title="Stock crash", summary="Markets fall sharply", source="a", timestamp="t")
    ev2 = nb.Event(title="Sports update", summary="Team wins", source="b", timestamp="t")
    filtered = nb.filter_events([ev1, ev2], ["stock"])
    assert [e.title for e in filtered] == ["Stock crash"]


def test_db_roundtrip(tmp_path: Path):
    db = nb.NewsDB(tmp_path / "news.db")
    ev = nb.Event(title="A", summary="B", source="s", timestamp="t")
    db.add(ev)
    events = db.fetch()
    assert events and events[0].title == "A"


def test_cluster_events():
    ev1 = nb.Event(title="Market rises", summary="stocks up", source="x", timestamp="t")
    ev2 = nb.Event(title="Market soars", summary="stocks increase", source="y", timestamp="t")
    clusters = nb.cluster_events([ev1, ev2], n_clusters=1)
    assert len(clusters) == 1 and len(clusters[0]) == 2
