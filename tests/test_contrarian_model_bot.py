import json
from pathlib import Path
from dataclasses import dataclass, field
import types
import sys
import pytest


class DummyBuilder:
    def refresh_db_weights(self) -> None:  # pragma: no cover - simple stub
        pass


# ---------------------------------------------------------------------------
# Stub heavy dependencies so the module can be imported without optional packages
vector_service_stub = types.ModuleType("vector_service")
vector_service_stub.ContextBuilder = DummyBuilder
vector_service_stub.EmbeddableDBMixin = object
vector_service_stub.CognitionLayer = object
sys.modules.setdefault("vector_service", vector_service_stub)
sys.modules.setdefault(
    "vector_service.context_builder", types.SimpleNamespace(ContextBuilder=DummyBuilder)
)


class EnhancementDB:
    pass


@dataclass
class ResearchItem:
    topic: str
    content: str
    timestamp: float
    title: str = ""
    tags: list[str] = field(default_factory=list)
    category: str = ""
    type_: str = ""
    associated_bots: list[str] = field(default_factory=list)
    model_id: int | None = None
    contrarian_id: int = 0


class InfoDB:
    def __init__(self, path: Path | str | None = None) -> None:
        self.records: list[ResearchItem] = []

    def add(self, item: ResearchItem) -> None:
        self.records.append(item)

    def search(self, topic: str):
        return [r for r in self.records if r.topic == topic]

    def set_current_model(self, mid: int) -> None:
        pass

    def set_current_contrarian(self, cid: int) -> None:
        pass


class ResearchAggregatorBot:
    def __init__(
        self,
        requirements,
        info_db: InfoDB | None = None,
        enhancements_db: EnhancementDB | None = None,
        *,
        context_builder: DummyBuilder,
    ) -> None:
        self.requirements = list(requirements)
        self.info_db = info_db or InfoDB()
        self.enhancements_db = enhancements_db or EnhancementDB()
        self.context_builder = context_builder

    def process(self, topic: str, energy: int = 1) -> None:
        pass


rab_module = types.ModuleType("menace.research_aggregator_bot")
rab_module.ResearchAggregatorBot = ResearchAggregatorBot
rab_module.InfoDB = InfoDB
rab_module.ResearchItem = ResearchItem
sys.modules.setdefault("menace.research_aggregator_bot", rab_module)
rab = rab_module


class ResourceAllocationBot:
    def __init__(self, *, context_builder: DummyBuilder) -> None:
        self.context_builder = context_builder

    def allocate(self, metrics):
        pass


sys.modules.setdefault(
    "menace.resource_allocation_bot", types.SimpleNamespace(ResourceAllocationBot=ResourceAllocationBot)
)
sys.modules.setdefault(
    "menace.chatgpt_enhancement_bot", types.SimpleNamespace(EnhancementDB=EnhancementDB)
)
sys.modules.setdefault(
    "menace.automated_reviewer", types.SimpleNamespace(AutomatedReviewer=object)
)


@dataclass
class WorkflowRecord:
    workflow: list[str]
    title: str
    description: str
    task_sequence: list[str]
    tags: list[str]
    category: str
    type_: str


class WorkflowDB:
    def __init__(self, path: Path | str | None = None, *, event_bus=None) -> None:
        self.records: list[WorkflowRecord] = []

    def add(self, rec: WorkflowRecord) -> None:
        self.records.append(rec)

    def fetch(self):
        return self.records


thb_module = types.ModuleType("menace.task_handoff_bot")
thb_module.WorkflowDB = WorkflowDB
thb_module.WorkflowRecord = WorkflowRecord
sys.modules.setdefault("menace.task_handoff_bot", thb_module)
thb = thb_module


@dataclass
class ContrarianRecord:
    innovation_name: str
    innovation_type: str
    risk_score: float
    reward_score: float
    activation_trigger: str
    resource_allocation: dict
    contrarian_id: int = 0


class ContrarianDB:
    def __init__(self, path: Path | str | None = None) -> None:
        self.records: list[ContrarianRecord] = []
        self.links: dict[int, list[int]] = {}

    def add(self, rec: ContrarianRecord) -> int:
        cid = len(self.records) + 1
        rec.contrarian_id = cid
        self.records.append(rec)
        return cid

    def link_model(self, cid: int, mid: int) -> None:
        self.links.setdefault(cid, []).append(mid)

    def fetch(self):
        return self.records

    def models_for(self, cid: int):
        return self.links.get(cid, [])


cdb_module = types.ModuleType("menace.contrarian_db")
cdb_module.ContrarianDB = ContrarianDB
cdb_module.ContrarianRecord = ContrarianRecord
sys.modules.setdefault("menace.contrarian_db", cdb_module)
cdb = cdb_module


import menace.contrarian_model_bot as cmb


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
    builder = DummyBuilder()
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
        def __init__(self, *, context_builder: DummyBuilder) -> None:
            self.requirements = []
            self.context_builder = context_builder

        def process(self, topic: str, energy: int = 1) -> None:
            called["topic"] = topic

    builder = DummyBuilder()
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        innovations_db=innovations,
        aggregator=DummyAggregator(context_builder=builder),
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
    builder = DummyBuilder()
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
    builder = DummyBuilder()
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        capital_manager=FailMgr(),
        context_builder=builder,
        innovations_db=cmb.InnovationsDB(tmp_path / "innov.json"),
    )
    calls = []
    monkeypatch.setattr(bot, "logger", types.SimpleNamespace(exception=calls.append))
    bot._update_risk_tolerance()
    assert calls and "risk tolerance" in calls[0]


def test_allocate_resources_logs(tmp_path: Path, monkeypatch):
    class FailAlloc:
        def __init__(self, *, context_builder: DummyBuilder) -> None:
            self.context_builder = context_builder

        def allocate(self, _):
            raise RuntimeError("fail")

    builder = DummyBuilder()
    bot = cmb.ContrarianModelBot(
        workflow_db=make_workflow_db(tmp_path),
        allocator=FailAlloc(context_builder=builder),
        context_builder=builder,
        innovations_db=cmb.InnovationsDB(tmp_path / "innov.json"),
    )
    calls = []
    monkeypatch.setattr(bot, "logger", types.SimpleNamespace(exception=calls.append))
    bot.allocate_resources(1.0)
    assert calls and "resource allocation failed" in calls[0]


def test_ideate_logs_aggregator_error(tmp_path: Path, monkeypatch):
    class Agg:
        def __init__(self, *, context_builder: DummyBuilder) -> None:
            self.info_db = rab.InfoDB(tmp_path / "info.db")
            self.requirements = []
            self.context_builder = context_builder

        def process(self, *_):
            raise RuntimeError("nope")

    db = make_workflow_db(tmp_path)
    innovations = cmb.InnovationsDB(tmp_path / "innov.json")
    builder = DummyBuilder()
    bot = cmb.ContrarianModelBot(
        workflow_db=db,
        innovations_db=innovations,
        aggregator=Agg(context_builder=builder),
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


def test_requires_context_builder(tmp_path: Path):
    db = make_workflow_db(tmp_path)
    with pytest.raises(ValueError):
        cmb.ContrarianModelBot(workflow_db=db, context_builder=None)

