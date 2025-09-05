import pytest

pytest.importorskip("sqlalchemy")

import importlib.util
import types
import sys
from pathlib import Path

# Load modules individually to avoid package side effects
ROOT = Path(__file__).resolve().parents[1]
pkg = types.ModuleType("menace")
pkg.__path__ = [str(ROOT)]
sys.modules["menace"] = pkg
sys.modules.setdefault("pandas", types.ModuleType("pandas"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"menace.{name}", ROOT / f"{name}.py", submodule_search_locations=[str(ROOT)]  # path-ignore
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[f"menace.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

mn = _load("databases")
dr = _load("db_router")
bd = _load("bot_database")
cd = _load("code_database")
rab = _load("research_aggregator_bot")
mm_mod = _load("menace_memory_manager")
thb = _load("task_handoff_bot")
ueb = _load("unified_event_bus")


def _make_router(tmp_path, bus):
    return dr.DBRouter(
        code_db=cd.CodeDB(tmp_path / "c.db", event_bus=bus),
        bot_db=bd.BotDB(tmp_path / "b.db", event_bus=bus),
        info_db=rab.InfoDB(tmp_path / "i.db", event_bus=bus),
        memory_mgr=mm_mod.MenaceMemoryManager(tmp_path / "mem.db", event_bus=bus),
        workflow_db=thb.WorkflowDB(tmp_path / "wf.db", event_bus=bus),
        menace_db=mn.MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}"),
        event_bus=bus,
        auto_cross_link=False,
    )


def test_insert_bot_emits_cdc_event(tmp_path):
    bus = ueb.UnifiedEventBus()
    events = []
    bus.subscribe("cdc:bots", lambda t, e: events.append(e))
    router = _make_router(tmp_path, bus)

    bid = router.insert_bot(bd.BotRecord(name="bot"))

    assert events and events[0] == {"action": "insert", "bot_id": bid}


def test_update_bot_emits_cdc_event(tmp_path):
    bus = ueb.UnifiedEventBus()
    events = []
    bus.subscribe("cdc:bots", lambda t, e: events.append(e))
    router = _make_router(tmp_path, bus)

    bid = router.insert_bot(bd.BotRecord(name="bot"))
    events.clear()

    router.update_bot(bid, status="inactive")

    assert events and events[0] == {"action": "update", "bot_id": bid, "status": "inactive"}

