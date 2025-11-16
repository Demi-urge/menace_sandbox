import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

# stub modules to avoid heavy dependencies
dummy_cap = types.ModuleType("menace_sandbox.capital_management_bot")
dummy_cap.CapitalManagementBot = object
sys.modules["menace_sandbox.capital_management_bot"] = dummy_cap

dummy_sys = types.ModuleType("menace_sandbox.system_evolution_manager")
dummy_sys.SystemEvolutionManager = object
sys.modules["menace_sandbox.system_evolution_manager"] = dummy_sys

dummy_evalhist = types.ModuleType("menace_sandbox.evaluation_history_db")
dummy_evalhist.EvaluationHistoryDB = object
sys.modules["menace_sandbox.evaluation_history_db"] = dummy_evalhist

dummy_trend = types.ModuleType("menace_sandbox.trend_predictor")
dummy_trend.TrendPredictor = object
sys.modules["menace_sandbox.trend_predictor"] = dummy_trend

dummy_scm = types.ModuleType("menace_sandbox.self_coding_manager")


class _HGE(Exception):
    pass


dummy_scm.HelperGenerationError = _HGE
sys.modules["menace_sandbox.self_coding_manager"] = dummy_scm

dummy_settings = types.ModuleType("menace_sandbox.sandbox_settings")


class _SS:
    pass


dummy_settings.SandboxSettings = _SS
sys.modules["menace_sandbox.sandbox_settings"] = dummy_settings

dummy_threshold = types.ModuleType("menace_sandbox.threshold_service")


class _TS:
    def get(self, *a, **k):
        class T:
            error_threshold = 0.0
            roi_drop = 0.0

        return T()

    def reload(self, *a, **k):
        return self.get()


th = _TS()
dummy_threshold.threshold_service = th
sys.modules["menace_sandbox.threshold_service"] = dummy_threshold

dummy_data_bot_mod = types.ModuleType("menace_sandbox.data_bot")


class _DataBot:
    pass


dummy_data_bot_mod.DataBot = _DataBot
sys.modules["menace_sandbox.data_bot"] = dummy_data_bot_mod

# stub self_improvement.baseline_tracker
di = types.ModuleType("menace_sandbox.self_improvement")
di.__path__ = []
bt = types.ModuleType("baseline_tracker")
bt.BaselineTracker = object
di.baseline_tracker = bt
sys.modules["menace_sandbox.self_improvement"] = di
sys.modules["menace_sandbox.self_improvement.baseline_tracker"] = bt

import menace_sandbox.evolution_orchestrator as eo  # noqa: E402
from menace_sandbox.evolution_orchestrator import (  # noqa: E402
    EvolutionOrchestrator,
    EvolutionTrigger,
)
from menace_sandbox.evolution_history_db import EvolutionHistoryDB  # noqa: E402
from menace_sandbox.bot_registry import BotRegistry  # noqa: E402


class DummyEventBus:
    def __init__(self):
        self.events = []

    def publish(self, topic, event):
        self.events.append((topic, event))

    def subscribe(self, topic, cb):
        pass


class DummyDataBot:
    def __init__(self):
        self.event_bus = DummyEventBus()
        self.db = types.SimpleNamespace(fetch=lambda limit: [])

    def subscribe_degradation(self, cb):
        pass

    def roi(self, bot):
        return 2.0

    def average_errors(self, bot):
        return 1.0

    def average_test_failures(self, bot):
        return 0.0


def test_register_patch_cycle_attempts_patch(monkeypatch, tmp_path):
    registry = BotRegistry()
    module_path = tmp_path / "mod.py"
    module_path.write_text("x=1\n")
    registry.graph.add_node("sample_bot", module=str(module_path))

    event_bus = DummyEventBus()
    data_bot = DummyDataBot()

    class SCM:
        def __init__(self):
            self.bot_name = "sample_bot"
            self.bot_registry = registry
            self.event_bus = event_bus
            self._last_patch_id = 1
            self._last_commit_hash = "abc"
            self.reg_calls = []
            self.gen_calls = []

        def register_patch_cycle(self, desc, ctx, *, provenance_token: str):
            self.reg_calls.append((desc, ctx, provenance_token))

        def generate_and_patch(
            self,
            path,
            desc,
            *,
            context_meta=None,
            context_builder=None,
            provenance_token: str,
        ):
            self.gen_calls.append(
                (path, desc, context_meta, context_builder, provenance_token)
            )
            return None, self._last_commit_hash

    scm = SCM()
    hist_db = EvolutionHistoryDB(path=tmp_path / "hist.db")

    class DummyBuilder:
        def refresh_db_weights(self):
            pass
    monkeypatch.setattr(eo, "create_context_builder", lambda: DummyBuilder())
    monkeypatch.setattr(eo, "ensure_fresh_weights", lambda b: None)

    trig = EvolutionTrigger()
    orch = EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        history_db=hist_db,
        triggers=trig,
        selfcoding_manager=scm,
        event_bus=event_bus,
        dataset_path=tmp_path / "ds.csv",
    )

    event = {"bot": "sample_bot", "roi_baseline": 1.0, "errors_baseline": 1.0}
    orch.register_patch_cycle(event)

    assert scm.gen_calls, "generate_and_patch not called"
    assert any(t == "self_coding:patch_attempt" for t, _ in event_bus.events)
    rows = hist_db.conn.execute("SELECT action FROM evolution_history").fetchall()
    assert rows and rows[0][0] in {"patch", "patch_failed"}
