import sqlite3
import pytest

pytest.importorskip("networkx")

import knowledge_graph as kg


class DummyErrorDB:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute(
            "CREATE TABLE telemetry (bot_id TEXT, error_type TEXT, root_module TEXT, category TEXT, module TEXT, cause TEXT, frequency INTEGER)"
        )
        self.conn.execute(
            "INSERT INTO telemetry VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("b1", "E", "r", "cat", "mod", "cause", 2),
        )


def test_ingest_error_instance():
    db = DummyErrorDB()
    g = kg.KnowledgeGraph()
    g.ingest_error_db(db)
    assert g.graph.get_edge_data("error_category:cat", "module:mod")["weight"] == 2
    assert g.graph.get_edge_data("module:mod", "cause:cause")["weight"] == 2
    assert g.graph.nodes["error_category:cat"]["weight"] == 2
    assert g.graph.nodes["error_type:E"]["frequency"] == 2
    assert g.graph.get_edge_data("module:r", "error_type:E")["weight"] == 2
