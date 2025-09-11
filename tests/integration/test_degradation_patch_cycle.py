import importlib
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("menace")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", package)

dpr = types.ModuleType("dynamic_path_router")
dpr.resolve_path = lambda p: Path(p)
dpr.repo_root = lambda: ROOT
dpr.resolve_dir = lambda p: Path(p)
dpr.path_for_prompt = lambda p: str(p)
sys.modules.setdefault("dynamic_path_router", dpr)

sr_pkg = types.ModuleType("sandbox_runner")
sr_pkg.__path__ = []
sys.modules.setdefault("sandbox_runner", sr_pkg)
th = types.ModuleType("sandbox_runner.test_harness")
th.run_tests = lambda *a, **k: types.SimpleNamespace(success=True, stdout="", duration=0)
th.TestHarnessResult = types.SimpleNamespace
sys.modules.setdefault("sandbox_runner.test_harness", th)

scm_mod = types.ModuleType("menace.self_coding_manager")

class HelperGenerationError(RuntimeError):
    pass


class SelfCodingManager:
    def __init__(self, **kwargs):
        self.quick_fix = kwargs.get("quick_fix")
        self.bot_name = kwargs.get("bot_name", "")
        self.bot_registry = kwargs.get("bot_registry")
        self.called = False

    def generate_and_patch(
        self, path: Path, description: str, *, context_meta=None, context_builder=None
    ):
        self.called = True
        passed, patch_id = self.quick_fix.apply_validated_patch(
            str(path), description, context_meta or {}
        )
        self.bot_registry.update_bot(
            self.bot_name, str(path), patch_id=patch_id, commit="deadbeef"
        )


scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = HelperGenerationError
sys.modules.setdefault("menace.self_coding_manager", scm_mod)

db_mod = importlib.import_module("menace.data_bot")
DataBot = db_mod.DataBot
MetricsDB = db_mod.MetricsDB
eo_mod = importlib.import_module("menace.evolution_orchestrator")
EvolutionOrchestrator = eo_mod.EvolutionOrchestrator
BotRegistry = importlib.import_module("menace.bot_registry").BotRegistry


class DummyBus:
    def __init__(self):
        self.subs = {}

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummyContextBuilder:
    pass


class DummyEngine:
    def __init__(self):
        self.cognition_layer = types.SimpleNamespace(context_builder=None)

    def generate_helper(self, description, path, metadata, strategy, target_region):
        return None


class DummyQuickFix:
    def __init__(self):
        self.calls = []

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc))
        return True, 123


class DummyPipeline:
    pass


class DummyCapital:
    trend_predictor = None

    def energy_score(self, load, success_rate, deploy_eff, failure_rate):
        return 1.0


class DummyImprovement:
    pass


class DummyEvolution:
    pass


class DummyHistoryDB:
    def add(self, *a, **k):
        pass


def test_degradation_triggers_patch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")

    import importlib

    importlib.invalidate_caches()
    __import__("dummy_module")

    bus = DummyBus()
    monkeypatch.setattr(eo_mod, "ContextBuilder", DummyContextBuilder)

    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)

    quick_fix = DummyQuickFix()

    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_module",
        bot_registry=registry,
    )

    orch = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
    )

    data_bot.check_degradation("dummy_module", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_module", roi=0.0, errors=2.0)

    assert manager.called
    assert quick_fix.calls
    node = registry.graph.nodes["dummy_module"]
    assert node["module"] == str(mod_path)
    assert node["patch_id"] == 123
