import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
from pathlib import Path
import json

import types
import menace.contrarian_model_bot as cmb
import menace.research_aggregator_bot as rab
import menace.task_handoff_bot as thb
import menace.contrarian_db as cdb
from vector_service.context_builder import ContextBuilder


def make_workflow_db(tmp_path: Path) -> cmb.WorkflowDB:
    data = [
        {
            "name": "wf1",
            "steps": [
                {"name": "scrape", "description": "gather data", "risk": 0.3},
                {"name": "analyse", "description": "process", "risk": 0.4},
            ],
            "tags": ["data"],
        },
        {
            "name": "wf2",
            "steps": [
                {"name": "publish", "description": "post", "risk": 0.8},
            ],
            "tags": ["media"],
        },
    ]
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(data))
    return cmb.WorkflowDB(path)


def test_load_workflows(tmp_path: Path):
    db = make_workflow_db(tmp_path)
    items = db.load()
    assert len(items) == 2
    assert items[0].steps[0].name == "scrape"


def test_merge_and_innovate(tmp_path: Path):
    db = make_workflow_db(tmp_path)
    innovations = cmb.InnovationsDB(tmp_path / "innov.json")
    info_db = rab.InfoDB(tmp_path / "info.db")
    wf_db = thb.WorkflowDB(tmp_path / "wf.db")
    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        workflows_db=wf_db,
        innovations_db=innovations,
        info_db=info_db,
        context_builder=builder,
        capital_manager=None,
        risk_tolerance=0.4,
        resources=1.0,
    )
    innov = bot.ideate()
    assert innov is not None
    stored = innovations.fetch()
    assert stored and stored[0].name == innov.name
    assert info_db.search(innov.name)
    assert any(rec.title == innov.name for rec in wf_db.fetch())


def test_ideate_triggers_research(tmp_path: Path, monkeypatch):
    db = make_workflow_db(tmp_path)
    innovations = cmb.InnovationsDB(tmp_path / "innov.json")
    called = {}

    class DummyAggregator:
        def __init__(self) -> None:
            self.requirements = []

        def process(self, topic: str, energy: int = 1) -> None:
            called["topic"] = topic

    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        innovations_db=innovations,
        aggregator=DummyAggregator(),
        context_builder=builder,
        capital_manager=None,
        risk_tolerance=0.4,
        resources=1.0,
    )
    innov = bot.ideate()
    assert innov is not None
    assert called.get("topic") == innov.name


def test_ideate_logs_contrarian(tmp_path: Path):
    db = make_workflow_db(tmp_path)
    innovations = cmb.InnovationsDB(tmp_path / "innov.json")
    contr_db = cdb.ContrarianDB(tmp_path / "c.db")
    info_db = rab.InfoDB(tmp_path / "info.db")
    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        innovations_db=innovations,
        contrarian_db=contr_db,
        model_ids=[1, 2],
        info_db=info_db,
        context_builder=builder,
        capital_manager=None,
        risk_tolerance=0.4,
        resources=1.0,
    )
    innov = bot.ideate()
    assert innov is not None
    items = contr_db.fetch()
    assert items and items[0].innovation_name == innov.name
    assert contr_db.models_for(items[0].contrarian_id) == [1, 2]
    info_items = info_db.search(innov.name)
    assert any(it.contrarian_id == items[0].contrarian_id for it in info_items)

def test_update_risk_tolerance_logs(tmp_path: Path, monkeypatch):
    class FailMgr:
        def energy_score(self, **_):
            raise RuntimeError("boom")

    db = make_workflow_db(tmp_path)
    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        capital_manager=FailMgr(),
        context_builder=builder,
    )
    calls = []
    monkeypatch.setattr(bot, "logger", types.SimpleNamespace(exception=calls.append))
    bot._update_risk_tolerance()
    assert calls and "risk tolerance" in calls[0]


def test_allocate_resources_logs(tmp_path: Path, monkeypatch):
    class FailAlloc:
        def allocate(self, _):
            raise RuntimeError("fail")
    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=make_workflow_db(tmp_path),
        allocator=FailAlloc(),
        context_builder=builder,
    )
    calls = []
    monkeypatch.setattr(bot, "logger", types.SimpleNamespace(exception=calls.append))
    bot.allocate_resources(1.0)
    assert calls and "resource allocation failed" in calls[0]


def test_ideate_logs_aggregator_error(tmp_path: Path, monkeypatch):
    class Agg:
        def __init__(self) -> None:
            self.info_db = rab.InfoDB(tmp_path / "info.db")
            self.requirements = []

        def process(self, *_):
            raise RuntimeError("nope")

    db = make_workflow_db(tmp_path)
    innovations = cmb.InnovationsDB(tmp_path / "innov.json")
    builder = ContextBuilder(
        bot_db="bots.db",
        code_db="code.db",
        error_db="errors.db",
        workflow_db="workflows.db",
    )
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        innovations_db=innovations,
        aggregator=Agg(),
        context_builder=builder,
        capital_manager=None,
        risk_tolerance=0.4,
        resources=1.0,
    )
    calls = []
    monkeypatch.setattr(bot, "logger", types.SimpleNamespace(exception=calls.append))
    innov = bot.ideate()
    assert innov is not None
    assert calls and "research aggregator failed" in calls[0]
