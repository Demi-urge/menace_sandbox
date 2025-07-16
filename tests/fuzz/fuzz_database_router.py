import importlib.util
import types
import sys
from pathlib import Path

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

# Load modules individually to avoid package side effects
ROOT = Path(__file__).resolve().parents[2]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
jinja_stub = types.ModuleType("jinja2")
jinja_stub.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_stub)

yaml_stub = types.ModuleType("yaml")
yaml_stub.safe_load = lambda *a, **k: {}
sys.modules.setdefault("yaml", yaml_stub)
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))

class StubInfoDB:
    def __init__(self, *a, **k):
        pass

    def search(self, term: str) -> list[dict]:
        return []


class StubMemoryMgr:
    def __init__(self, *a, **k):
        pass

    def search_by_tag(self, term: str) -> list[dict]:
        return []

rab_stub = types.ModuleType("menace.research_aggregator_bot")
rab_stub.InfoDB = StubInfoDB
rab_stub.__spec__ = importlib.util.spec_from_loader("menace.research_aggregator_bot", loader=None)
sys.modules.setdefault("menace.research_aggregator_bot", rab_stub)
setattr(pkg, "research_aggregator_bot", rab_stub)

mm_stub = types.ModuleType("menace.menace_memory_manager")
mm_stub.MenaceMemoryManager = StubMemoryMgr
mm_stub.__spec__ = importlib.util.spec_from_loader("menace.menace_memory_manager", loader=None)
sys.modules.setdefault("menace.menace_memory_manager", mm_stub)
setattr(pkg, "menace_memory_manager", mm_stub)


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


dr = _load("database_router")
bd = _load("bot_database")
cd = _load("code_database")
thb = _load("task_handoff_bot")
ueb = _load("unified_event_bus")


class _DummyInfoDB:
    def search(self, term: str) -> list[dict]:
        return []


class _DummyMemoryMgr:
    def search_by_tag(self, term: str) -> list[dict]:
        return []


def _make_router(tmp_path):
    bus = ueb.UnifiedEventBus()
    return dr.DatabaseRouter(
        code_db=cd.CodeDB(tmp_path / "c.db", event_bus=bus),
        bot_db=bd.BotDB(tmp_path / "b.db", event_bus=bus),
        info_db=_DummyInfoDB(),
        memory_mgr=_DummyMemoryMgr(),
        workflow_db=thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus),
        menace_db=None,
        event_bus=bus,
        auto_cross_link=False,
    )


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    names=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    term=st.text(min_size=0, max_size=10),
)
@pytest.mark.skipif("numpy" not in sys.modules and __import__('importlib').util.find_spec('numpy') is None,
                    reason="numpy not available")
def test_dbrouter_query_all_fuzz(tmp_path, names, term):
    """Fuzz DatabaseRouter.query_all with random terms."""
    router = _make_router(tmp_path)
    for n in names:
        router.insert_bot(bd.BotRecord(name=n))
    res = router.query_all(term)
    assert isinstance(res, dr.DBResult)


@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    names=st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=5),
    idx=st.integers(min_value=0, max_value=4),
)
@pytest.mark.skipif("numpy" not in sys.modules and __import__('importlib').util.find_spec('numpy') is None,
                    reason="numpy not available")
def test_dbrouter_execute_query_fuzz(tmp_path, names, idx):
    """Fuzz DatabaseRouter.execute_query with simple SELECT statements."""
    router = _make_router(tmp_path)
    for n in names:
        router.insert_bot(bd.BotRecord(name=n))
    name = names[idx % len(names)]
    rows = router.execute_query("bot", "SELECT name FROM bots WHERE name=?", [name])
    assert rows and rows[0]["name"] == name
