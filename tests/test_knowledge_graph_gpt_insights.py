import sys
import types
import sqlite3
import pytest

sys.modules.setdefault(
    "menace.automated_reviewer", types.SimpleNamespace(AutomatedReviewer=object)
)

from menace.knowledge_graph import KnowledgeGraph
from menace.menace_memory_manager import MenaceMemoryManager
from menace.unified_event_bus import UnifiedEventBus


def test_ingest_gpt_insights_links_entities():
    kg = KnowledgeGraph()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memory (key TEXT, tags TEXT)")
    conn.execute(
        "INSERT INTO memory VALUES (?, ?)",
        ("idea1", "bot:alpha,code:module.py,error:ValueError"),  # path-ignore
    )
    manager = types.SimpleNamespace(conn=conn)
    kg.ingest_gpt_memory(manager)
    inode = "insight:idea1"
    assert inode in kg.graph
    assert ("insight:idea1", "bot:alpha") in kg.graph.edges
    assert ("insight:idea1", "code:module.py") in kg.graph.edges  # path-ignore
    assert ("insight:idea1", "error_category:ValueError") in kg.graph.edges


def test_bugfix_logging_creates_insight(tmp_path):
    pytest.importorskip("networkx")
    kg = KnowledgeGraph(tmp_path / "kg.gpickle")
    mgr = MenaceMemoryManager(path=tmp_path / "m.db")
    mgr.graph = kg
    mgr.store(
        "fix1",
        "data",
        tags="bugfix bot:alpha code:module.py error:ValueError",  # path-ignore
    )
    inode = "insight:fix1"
    assert inode in kg.graph
    assert ("insight:fix1", "bot:alpha") in kg.graph.edges
    kg2 = KnowledgeGraph(tmp_path / "kg.gpickle")
    assert inode in kg2.graph
    assert kg2.find_insights(bot="alpha") == ["fix1"]
    top = kg2.top_insights()
    assert top and top[0][0] == "fix1"


def test_memory_new_event_triggers_ingest():
    bus = UnifiedEventBus()
    mgr = MenaceMemoryManager(path=":memory:", event_bus=bus)
    kg = KnowledgeGraph()
    kg.listen_to_memory(bus, mgr)
    mgr.store(
        "idea1",
        "something",
        tags="bot:alpha,code:module.py,error:ValueError",  # path-ignore
    )
    inode = "insight:idea1"
    assert inode in kg.graph
    assert ("insight:idea1", "bot:alpha") in kg.graph.edges
    assert ("insight:idea1", "code:module.py") in kg.graph.edges  # path-ignore
    assert ("insight:idea1", "error_category:ValueError") in kg.graph.edges


def test_interactions_table_ingested_and_tags_mapped():
    kg = KnowledgeGraph()
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE interactions (prompt TEXT, response TEXT, tags TEXT, ts TEXT)"
    )
    conn.execute(
        "INSERT INTO interactions VALUES (?, ?, ?, ?)",
        (
            "p1",
            "r1",
            "feedback improvement_path error_fix insight",
            "ts",
        ),
    )
    manager = types.SimpleNamespace(conn=conn)
    kg.ingest_gpt_memory(manager)
    inode = "insight:p1"
    assert inode in kg.graph
    for tag in ["feedback", "improvement_path", "error_fix", "insight"]:
        node = f"tag:{tag}"
        assert (inode, node) in kg.graph.edges
        assert kg.graph[inode][node]["type"] == tag
