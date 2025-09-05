import importlib, sys
from pathlib import Path
import pytest

pytest.importorskip("networkx")

import menace.error_bot as eb
import menace.error_logger as elog
import menace.knowledge_graph as kg


def test_update_error_stats_records_causes(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = eb.ErrorDB(tmp_path / "e.db")
    logger = elog.ErrorLogger(db)

    mod = tmp_path / "m.py"  # path-ignore
    mod.write_text("def boom():\n    raise ValueError('bad')\n")
    sys.path.insert(0, str(tmp_path))
    m = importlib.import_module("m")

    for _ in range(2):
        try:
            m.boom()
        except Exception as exc:  # pragma: no cover - exception path
            logger.log(exc, "t", "b")

    graph = kg.KnowledgeGraph()
    graph.update_error_stats(db)

    assert graph.graph.get_edge_data("module:m", "cause:ValueError")["weight"] == 2
    assert graph.graph.nodes["cause:ValueError"]["weight"] == 2
