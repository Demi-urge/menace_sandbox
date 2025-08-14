import sys
import types

sys.modules.setdefault(
    "menace.automated_reviewer", types.SimpleNamespace(AutomatedReviewer=object)
)

from menace.knowledge_graph import KnowledgeGraph
from menace.menace_memory_manager import MenaceMemoryManager
from menace.unified_event_bus import UnifiedEventBus
import sqlite3


def test_ingest_gpt_insights_links_entities():
    kg = KnowledgeGraph()
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE memory (key TEXT, tags TEXT)")
    conn.execute(
        "INSERT INTO memory VALUES (?, ?)",
        ("idea1", "bot:alpha,code:module.py,error:ValueError"),
    )
    manager = types.SimpleNamespace(conn=conn)
    kg.ingest_gpt_memory(manager)
    inode = "insight:idea1"
    assert inode in kg.graph
    assert ("insight:idea1", "bot:alpha") in kg.graph.edges
    assert ("insight:idea1", "code:module.py") in kg.graph.edges
    assert ("insight:idea1", "error_category:ValueError") in kg.graph.edges


def test_memory_new_event_triggers_ingest():
    bus = UnifiedEventBus()
    mgr = MenaceMemoryManager(path=":memory:", event_bus=bus)
    kg = KnowledgeGraph()
    kg.listen_to_memory(bus, mgr)
    mgr.store(
        "idea1",
        "something",
        tags="bot:alpha,code:module.py,error:ValueError",
    )
    inode = "insight:idea1"
    assert inode in kg.graph
    assert ("insight:idea1", "bot:alpha") in kg.graph.edges
    assert ("insight:idea1", "code:module.py") in kg.graph.edges
    assert ("insight:idea1", "error_category:ValueError") in kg.graph.edges
