import importlib.util
import sys
from pathlib import Path
import types

spec = importlib.util.spec_from_file_location(
    "menace",
    Path(__file__).resolve().parents[1] / "__init__.py",  # path-ignore
    submodule_search_locations=[str(Path(__file__).resolve().parents[1])],
)
menace = importlib.util.module_from_spec(spec)
sys.modules["menace"] = menace
spec.loader.exec_module(menace)

bus_mod = types.ModuleType("menace.unified_event_bus")
class _Bus:
    def __init__(self, *a, **k):
        pass
    def publish(self, *a, **k):
        pass
bus_mod.UnifiedEventBus = _Bus
class _EventBus:
    pass
bus_mod.EventBus = _EventBus
sys.modules["menace.unified_event_bus"] = bus_mod

kg_mod = types.ModuleType("menace.knowledge_graph")
class _KG:
    def __init__(self, *a, **k):
        pass
    def add_memory_entry(self, *a, **k):
        pass
kg_mod.KnowledgeGraph = _KG
sys.modules["menace.knowledge_graph"] = kg_mod

from menace.menace_memory_manager import MenaceMemoryManager


def test_periodic_summary(tmp_path):
    mm = MenaceMemoryManager(tmp_path / "m.db", embedder=None, summary_interval=2)
    mm.store("k", "First.")
    mm.store("k", "Second.")  # triggers summarisation
    assert not mm.query("k", limit=5)
    summary_entries = mm.query("k:summary", limit=5)
    assert summary_entries
    assert "refs=1-2" in summary_entries[0].tags
