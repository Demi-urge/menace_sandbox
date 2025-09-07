# flake8: noqa
import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import types
import menace.menace_orchestrator as mo
import menace.neuroplasticity as neu
import menace.discrepancy_detection_bot as ddb


def test_create_and_run_cycle(monkeypatch):
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    orch.create_oversight("bot1", "L1")
    # stub pipeline
    class StubPipeline:
        def run(self, model: str):
            return mo.AutomationResult(package=None, roi=None)
    orch.pipeline = StubPipeline()
    res = orch.run_cycle(["demo"])
    assert "demo" in res


def test_pathway_priority(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    hi = db.log(neu.PathwayRecord(actions="run_cycle:hi", inputs="", outputs="", exec_time=0, resources="", outcome=neu.Outcome.SUCCESS, roi=2))
    for _ in range(3):
        db._update_meta(hi, neu.PathwayRecord(actions="run_cycle:hi", inputs="", outputs="", exec_time=0, resources="", outcome=neu.Outcome.SUCCESS, roi=2))
    orch = mo.MenaceOrchestrator(pathway_db=db, context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    class StubPipeline:
        def __init__(self):
            self.calls = []
        def run(self, model: str):
            self.calls.append(model)
            return mo.AutomationResult(package=None, roi=None)
    orch.pipeline = StubPipeline()
    res = orch.run_cycle(["lo", "hi"])
    assert list(res.keys())[0] == "hi"
    cur = db.conn.execute("SELECT COUNT(*) FROM pathways WHERE actions LIKE 'run_cycle:hi%'").fetchone()
    assert cur[0] >= 2


def test_next_pathway_sequencing(tmp_path):
    db = neu.PathwayDB(tmp_path / "p.db")
    a = db.log(neu.PathwayRecord(actions="run_cycle:a", inputs="", outputs="", exec_time=0, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    b = db.log(neu.PathwayRecord(actions="run_cycle:b", inputs="", outputs="", exec_time=0, resources="", outcome=neu.Outcome.SUCCESS, roi=1))
    db.conn.execute("UPDATE metadata SET myelination_score=2 WHERE pathway_id=?", (b,))
    db.reinforce_link(a, b)
    db.conn.commit()
    orch = mo.MenaceOrchestrator(pathway_db=db, context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    class StubPipeline:
        def __init__(self):
            self.calls = []
        def run(self, model: str):
            self.calls.append(model)
            return mo.AutomationResult(package=None, roi=None)
    orch.pipeline = StubPipeline()
    res = orch.run_cycle(["a"])
    assert list(res.keys()) == ["a", "b"]
    assert orch.pipeline.calls == ["a", "b"]


def test_health_check_and_reroute(monkeypatch):
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    orch.create_oversight("A", "L1")
    orch.create_oversight("B", "L1")
    orch.create_oversight("C", "L1")

    class DummyKG:
        def root_causes(self, bot: str):
            return [f"error:{bot}"]

    orch.knowledge_graph = DummyKG()
    causes = orch.record_failure("B")
    assert causes == ["error:B"]
    bot = orch.reassign_task("A", ["B"], ["C"])
    assert bot == "C"


class DummyWatchdog:
    def __init__(self):
        self.hb = []
        self.healed = []
        self.healer = types.SimpleNamespace(heal=lambda b, pid=None: self.healed.append(b))

    def record_heartbeat(self, name):
        self.hb.append(name)

    def schedule(self, interval=60):
        pass


def test_planning_job_trigger(monkeypatch):
    monkeypatch.setenv("PLANNING_INTERVAL", "0")
    monkeypatch.setenv("WATCHDOG_INTERVAL", "0")
    monkeypatch.setattr(mo, "BackgroundScheduler", None)
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    orch.watchdog = DummyWatchdog()
    planner = types.SimpleNamespace(called=False)

    def _plan():
        planner.called = True
        return "plan"

    planner.plan_cycle = _plan
    orch.planner = planner
    monkeypatch.setattr(mo._SimpleScheduler, "add_job", lambda self, func, interval, id: self.tasks.setdefault(id, (interval, func, 0.0)))
    monkeypatch.setattr(mo._SimpleScheduler, "_run", lambda self: [info[1]() for info in self.tasks.values()] or (_ for _ in ()).throw(SystemExit))
    orch.start_scheduled_jobs()
    with pytest.raises(SystemExit):
        orch.scheduler._run()
    assert planner.called
    assert "planning" in orch.watchdog.hb


def test_planning_job_logs_error(monkeypatch, caplog):
    monkeypatch.setenv("PLANNING_INTERVAL", "0")
    monkeypatch.setenv("WATCHDOG_INTERVAL", "0")
    monkeypatch.setattr(mo, "BackgroundScheduler", None)
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    orch.watchdog = DummyWatchdog()
    monkeypatch.setattr(orch, "_trending_job", lambda: None)
    monkeypatch.setattr(orch, "_learning_job", lambda: None)
    planner = types.SimpleNamespace(plan_cycle=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    orch.planner = planner
    monkeypatch.setattr(mo._SimpleScheduler, "add_job", lambda self, func, interval, id: self.tasks.setdefault(id, (interval, func, 0.0)))
    monkeypatch.setattr(mo._SimpleScheduler, "_run", lambda self: [info[1]() for info in self.tasks.values()] or (_ for _ in ()).throw(SystemExit))
    caplog.set_level("ERROR")
    orch.start_scheduled_jobs()
    with pytest.raises(SystemExit):
        orch.scheduler._run()
    assert "planning job failed" in caplog.text


def test_seed_job_triggers(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNUP_URL", "http://example.com/signup")
    monkeypatch.setenv("VAULT_PATH", str(tmp_path / "v.db"))

    called = {}

    def fake_seed(url, vault):
        called["url"] = url
        called["vault"] = vault

    class DummyVault:
        def __init__(self, path):
            self.path = path
            self.low = True

        def count(self, domain):
            return 0 if self.low else 10

    dv = DummyVault(str(tmp_path / "v.db"))
    monkeypatch.setattr(mo, "SessionVault", lambda path: dv)
    monkeypatch.setattr(mo, "seed_identity", fake_seed)
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))
    monkeypatch.setattr(orch, "_heartbeat", lambda name: None)

    orch._seed_job()
    assert called["url"] == "http://example.com/signup"
    assert called["vault"] is dv

    called.clear()
    dv.low = False
    orch._seed_job()
    assert called == {}


def test_detection_hooks(monkeypatch):
    orch = mo.MenaceOrchestrator(context_builder=types.SimpleNamespace(refresh_db_weights=lambda: None))

    class StubPipeline:
        def __init__(self):
            self.roi_threshold = 0.0

        def run(self, model: str):
            return mo.AutomationResult(package=None, roi=None)

    orch.pipeline = StubPipeline()
    called = {"scan": False, "eff": False}

    def fake_scan():
        called["scan"] = True
        return [ddb.Detection("boom", 2.0, "wf")]

    def fake_eff():
        called["eff"] = True
        return {"predicted_bottleneck": 0.9}

    monkeypatch.setattr(orch.discrepancy_detector, "scan", fake_scan)
    monkeypatch.setattr(orch.bottleneck_detector, "assess_efficiency", fake_eff)

    orch.run_cycle(["demo"])

    assert called["scan"] and called["eff"]
    assert orch.pipeline.roi_threshold > 0.0
