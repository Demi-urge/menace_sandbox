import os
import sys
import types
import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
ed_mod = sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed_mod.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed_mod.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
sys.modules["cryptography.hazmat.primitives"].serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("httpx", types.ModuleType("httpx"))
sys.modules.setdefault("requests", types.ModuleType("requests"))
jinja_mod = types.ModuleType("jinja2")
jinja_mod.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja_mod)
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
env_mod = types.ModuleType("env_config")
env_mod.DATABASE_URL = "sqlite:///"
sys.modules.setdefault("env_config", env_mod)

# Lightweight stubs for modules with heavy dependencies
rao_mod = types.ModuleType("menace.resource_allocation_optimizer")
rao_mod.ROIDB = type("ROIDB", (), {})
rao_mod.KPIRecord = type("KPIRecord", (), {})
sys.modules.setdefault("menace.resource_allocation_optimizer", rao_mod)

for name, attrs in {
    "menace.chatgpt_enhancement_bot": ["EnhancementDB", "ChatGPTEnhancementBot", "Enhancement"],
    "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
    "menace.text_research_bot": ["TextResearchBot"],
    "menace.video_research_bot": ["VideoResearchBot"],
}.items():
    mod = types.ModuleType(name)
    for attr in attrs:
        setattr(mod, attr, type(attr, (), {}))
    sys.modules.setdefault(name, mod)

pytest.importorskip("sqlalchemy")
pytest.importorskip("networkx")

from menace.bot_registry import BotRegistry
from menace.neuroplasticity import PathwayDB, PathwayRecord, Outcome
from menace.databases import MenaceDB
from menace import cross_query
import menace.resource_allocation_optimizer as rao
import menace.data_bot as db


def _setup_menace(tmp_path):
    db = MenaceDB(url=f"sqlite:///{tmp_path / 'm.db'}")
    with db.engine.begin() as conn:
        conn.execute(db.bots.insert().values(bot_id=1, bot_name='A'))
        conn.execute(db.bots.insert().values(bot_id=2, bot_name='B'))
        conn.execute(db.workflows.insert().values(workflow_id=1, workflow_name='wfA'))
        conn.execute(db.workflows.insert().values(workflow_id=2, workflow_name='wfB'))
        conn.execute(db.workflow_bots.insert().values(workflow_id=1, bot_id=1))
        conn.execute(db.workflow_bots.insert().values(workflow_id=2, bot_id=2))
        conn.execute(db.code.insert().values(code_id=1, template_type='foo', language='py', version='1', complexity_score=1.0, code_summary='foo code'))
        conn.execute(db.code.insert().values(code_id=2, template_type='bar', language='py', version='1', complexity_score=1.0, code_summary='bar code'))
        conn.execute(db.code_bots.insert().values(code_id=1, bot_id=1))
        conn.execute(db.code_bots.insert().values(code_id=2, bot_id=2))
    return db


def test_related_workflows(tmp_path):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction('A', 'B')
    pdb = PathwayDB(tmp_path / 'p.db')
    pdb.log(PathwayRecord(actions='wfB', inputs='', outputs='', exec_time=1.0, resources='', outcome=Outcome.SUCCESS, roi=1.0))

    results = cross_query.related_workflows('A', registry=reg, menace_db=db, pathway_db=pdb)
    assert 'wfA' in results
    assert 'wfB' in results


def test_similar_code_snippets(tmp_path):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction('A', 'B')
    pdb = PathwayDB(tmp_path / 'p2.db')
    pdb.log(PathwayRecord(actions='foo code', inputs='', outputs='', exec_time=1.0, resources='', outcome=Outcome.SUCCESS, roi=1.0))

    res = cross_query.similar_code_snippets('foo', menace_db=db, registry=reg, pathway_db=pdb)
    ids = [r['code_id'] for r in res]
    assert 1 in ids
    assert 2 in ids


def test_related_resources(tmp_path):
    import time
    import types
    import sys
    mods = {
        "menace.chatgpt_enhancement_bot": ["EnhancementDB", "ChatGPTEnhancementBot", "Enhancement"],
        "menace.chatgpt_prediction_bot": ["ChatGPTPredictionBot", "IdeaFeatures"],
        "menace.text_research_bot": ["TextResearchBot"],
        "menace.video_research_bot": ["VideoResearchBot"],
        "menace.chatgpt_research_bot": ["ChatGPTResearchBot", "Exchange", "summarise_text"],
        "menace.database_manager": ["get_connection", "DB_PATH"],
        "menace.capital_management_bot": ["CapitalManagementBot"],
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        for attr in attrs:
            if attr == "summarise_text":
                setattr(module, attr, lambda text, ratio=0.2: text[:10])
            elif attr == "get_connection":
                setattr(module, attr, lambda path: None)
            elif attr == "DB_PATH":
                setattr(module, attr, ":memory:")
            else:
                setattr(module, attr, type(attr, (), {}))
        sys.modules.setdefault(name, module)

    import menace.research_aggregator_bot as rab
    from menace.gpt_memory import GPTMemoryManager

    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction('A', 'B')

    info_db = rab.InfoDB(tmp_path / "info.db", menace_db=db)
    it1 = rab.ResearchItem(topic="t", content="c", timestamp=time.time(), title="A info", summary="sa", associated_bots=["A"])
    info_db.add(it1)
    it2 = rab.ResearchItem(topic="t2", content="c", timestamp=time.time(), title="B info", summary="sb", associated_bots=["B"])
    info_db.add(it2)

    mm = GPTMemoryManager(tmp_path / "mem.db")
    mm.store("ka", "d", tags=["A"])
    mm.store("kb", "d", tags=["B"])

    results = cross_query.related_resources(
        'A', registry=reg, menace_db=db, info_db=info_db, memory_mgr=mm
    )
    assert results['bots'] == ['A', 'B']
    assert 'wfA' in results['workflows']
    assert 'wfB' in results['workflows']
    assert 'A info' in results['information']
    assert 'B info' in results['information']
    assert 'ka' in results['memory']
    assert 'kb' in results['memory']


def test_entry_workflow_features(tmp_path):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction('A', 'B')
    pdb = PathwayDB(tmp_path / 'pf.db')

    feats = cross_query.entry_workflow_features(
        {"bot": "A"}, registry=reg, menace_db=db, pathway_db=pdb
    )
    assert 'wfA' in feats
    assert 'wfB' in feats


def test_workflow_roi_stats_and_ranking(tmp_path):
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")

    roi.add_action_roi("wfA", 10.0, 2.0, 1.0, 1.0)
    roi.add_action_roi("wfA", 20.0, 4.0, 2.0, 1.0)
    roi.add_action_roi("wfB", 5.0, 1.0, 1.0, 1.0)

    metrics.log_eval("wfA", "duration", 1.0)
    metrics.log_eval("wfA", "duration", 2.0)
    metrics.log_eval("wfA", "api_cost", 4.0)
    metrics.log_eval("wfB", "duration", 1.0)
    metrics.log_eval("wfB", "api_cost", 1.0)

    s = cross_query.workflow_roi_stats("wfA", roi, metrics)
    assert s["roi"] == 20.0
    assert s["cpu_seconds"] == 6.0
    assert s["api_cost"] == 10.0

    ranked = cross_query.rank_workflows(["wfA", "wfB"], roi, metrics)
    assert ranked[0][0] == "wfA"


def test_bot_roi_stats_and_ranking(tmp_path):
    roi = rao.ROIDB(tmp_path / "rb.db")
    metrics = db.MetricsDB(tmp_path / "mb.db")

    roi.add(rao.KPIRecord(bot="A", revenue=10.0, api_cost=2.0, cpu_seconds=1.0, success_rate=1.0))
    roi.add(rao.KPIRecord(bot="A", revenue=20.0, api_cost=4.0, cpu_seconds=2.0, success_rate=1.0))
    roi.add(rao.KPIRecord(bot="B", revenue=5.0, api_cost=1.0, cpu_seconds=1.0, success_rate=1.0))

    metrics.log_eval("A", "duration", 1.0)
    metrics.log_eval("A", "duration", 2.0)
    metrics.log_eval("A", "api_cost", 4.0)
    metrics.log_eval("B", "duration", 1.0)
    metrics.log_eval("B", "api_cost", 1.0)

    s = cross_query.bot_roi_stats("A", roi, metrics)
    assert s["roi"] == 20.0
    assert s["cpu_seconds"] == 6.0
    assert s["api_cost"] == 10.0

    ranked = cross_query.rank_bots(["A", "B"], roi, metrics)
    assert ranked[0][0] == "A"


def test_related_resources_failures(tmp_path, monkeypatch):
    import types
    import menace.research_aggregator_bot as rab
    from menace.gpt_memory import GPTMemoryManager

    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction("A", "B")

    info_db = rab.InfoDB(tmp_path / "info_fail.db", menace_db=db)
    it = rab.ResearchItem(topic="t", content="c", timestamp=0.0, title="A info", summary="s", associated_bots=["A"])
    info_db.add(it)

    mm = GPTMemoryManager(tmp_path / "mem_fail.db")
    mm.store("ka", "d", tags=["A"])

    def fail_search(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(info_db, "search", fail_search)
    monkeypatch.setattr(mm, "search_context", fail_search)

    res = cross_query.related_resources("A", registry=reg, menace_db=db, info_db=info_db, memory_mgr=mm)
    assert "wfA" in res["workflows"]
    assert res["information"] == []
    assert res["memory"] == []


def test_entry_workflow_features_snippet_failure(tmp_path, monkeypatch):
    db = _setup_menace(tmp_path)
    reg = BotRegistry()
    reg.register_interaction("A", "B")
    pdb = PathwayDB(tmp_path / "pf2.db")

    def fail_snippets(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(cross_query, "similar_code_snippets", fail_snippets)

    res = cross_query.entry_workflow_features({"bot": "A", "summary": "x"}, registry=reg, menace_db=db, pathway_db=pdb)
    assert "wfA" in res
    assert "wfB" in res


def test_workflow_roi_stats_history_failure(tmp_path, monkeypatch):
    roi = rao.ROIDB(tmp_path / "r2.db")
    metrics = db.MetricsDB(tmp_path / "m2.db")

    metrics.log_eval("wfA", "duration", 1.0)
    metrics.log_eval("wfA", "duration", 2.0)
    metrics.log_eval("wfA", "api_cost", 4.0)

    def fail_history(*args, **kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(roi, "history", fail_history)

    s = cross_query.workflow_roi_stats("wfA", roi, metrics)
    assert s["cpu_seconds"] == 3.0
    assert s["api_cost"] == 4.0
    assert s["roi"] == -4.0


def test_bot_roi_stats_metrics_failure(tmp_path, monkeypatch):
    roi = rao.ROIDB(tmp_path / "rb2.db")
    metrics = db.MetricsDB(tmp_path / "mb2.db")

    roi.add(rao.KPIRecord(bot="A", revenue=10.0, api_cost=2.0, cpu_seconds=1.0, success_rate=1.0))

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("oops")

    monkeypatch.setattr(metrics, "fetch_eval", fail_fetch)

    s = cross_query.bot_roi_stats("A", roi, metrics)
    assert s["cpu_seconds"] == 1.0
    assert s["api_cost"] == 2.0
    assert s["roi"] == 8.0
