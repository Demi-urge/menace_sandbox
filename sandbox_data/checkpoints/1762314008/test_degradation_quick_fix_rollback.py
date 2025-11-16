from __future__ import annotations

from pathlib import Path
import sys
import types
import importlib

from tests.fixtures.metric_generators import StaticMetricGenerator

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

cbi = types.ModuleType("menace.coding_bot_interface")
cbi.self_coding_managed = lambda f: f
sys.modules.setdefault("menace.coding_bot_interface", cbi)


class SelfCodingManager:
    def __init__(self, **kwargs):
        self.quick_fix = kwargs.get("quick_fix")
        self.bot_name = kwargs.get("bot_name", "")
        self.bot_registry = kwargs.get("bot_registry")
        self.event_bus = kwargs.get("event_bus")
        self._last_patch_id = None
        self._last_commit_hash = None
        if self.bot_registry:
            self.bot_registry.register_bot(self.bot_name)

    def should_refactor(self) -> bool:
        return True

    def register_patch_cycle(self, description, context_meta=None):
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered",
                {"bot": self.bot_name, "description": description},
            )
        self._last_patch_id = 100
        self._last_commit_hash = "deadbeef"
        return 100, "deadbeef"

    def run_patch(self, path: Path, description: str, *, context_meta=None, context_builder=None):
        passed, pid, _ = self.quick_fix.apply_validated_patch(
            str(path), description, context_meta or {}
        )
        if not passed:
            self._last_patch_id = None
            self._last_commit_hash = None
            raise RuntimeError("quick fix validation failed")
        self._last_patch_id = pid
        self._last_commit_hash = "deadbeef"
        if self.event_bus:
            roi = context_meta.get("roi", 0.0)
            self.event_bus.publish(
                "self_coding:patch_applied",
                {
                    "bot": self.bot_name,
                    "patch_id": pid,
                    "roi_before": roi,
                    "roi_after": roi,
                    "roi_delta": 0.0,
                },
            )
        return None


scm_mod = types.ModuleType("menace.self_coding_manager")
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = RuntimeError
sys.modules.setdefault("menace.self_coding_manager", scm_mod)

db_mod = importlib.import_module("menace.data_bot")
DataBot = db_mod.DataBot
MetricsDB = db_mod.MetricsDB
sem = types.ModuleType("menace.system_evolution_manager")
sem.SystemEvolutionManager = object
sys.modules.setdefault("menace.system_evolution_manager", sem)
sebot = types.ModuleType("menace.structural_evolution_bot")
sys.modules.setdefault("menace.structural_evolution_bot", sebot)
eo_mod = importlib.import_module("menace.evolution_orchestrator")
EvolutionOrchestrator = eo_mod.EvolutionOrchestrator
BotRegistry = importlib.import_module("menace.bot_registry").BotRegistry


class DummyBus:
    def __init__(self) -> None:
        self.subs: dict[str, list] = {}
        self.events: list[tuple[str, dict]] = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.events.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummyContextBuilder:
    def refresh_db_weights(self):
        pass


class DummyCapital:
    def energy_score(self, *a, **k):
        return 1.0


class DummyImprovement:
    pass


class DummyEvolution:
    pass


class DummyHistoryDB:
    def add(self, *a, **k):
        pass


class SuccessfulQuickFix:
    def __init__(self, roi_state):
        self.calls: list = []
        self.roi_state = roi_state

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc))
        self.roi_state["value"] = 5.0
        return True, 321, []


class FailingQuickFix:
    def __init__(self, rollback_calls):
        self.calls: list = []
        self.rollback_calls = rollback_calls

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc))
        self.rollback_calls.append("100")
        return False, 100, []


def _setup(mod_path, monkeypatch, quick_fix):
    bus = DummyBus()
    monkeypatch.setattr(eo_mod, "ContextBuilder", DummyContextBuilder)
    data_bot = DataBot(MetricsDB(mod_path.parent / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)
    registry.graph.add_node("dummy_module", module=str(mod_path))
    def _upd(name, module_path, patch_id=None, commit=None):
        registry.graph.add_node(name)
        registry.graph.nodes[name]["module"] = module_path
        registry.graph.nodes[name]["patch_id"] = patch_id
        registry.graph.nodes[name]["commit"] = commit
    monkeypatch.setattr(registry, "update_bot", _upd)
    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_module",
        bot_registry=registry,
        event_bus=bus,
    )
    EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
    )
    return data_bot, registry, bus


def test_degradation_applies_patch_and_improves_roi(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")
    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    __import__("dummy_module")

    roi_state = {"value": 0.0}
    quick_fix = SuccessfulQuickFix(roi_state)
    data_bot, registry, bus = _setup(mod_path, monkeypatch, quick_fix)
    monkeypatch.setattr(data_bot, "roi", lambda _name: roi_state["value"])

    metrics1 = StaticMetricGenerator(roi=1.0).generate()
    metrics2 = StaticMetricGenerator(roi=0.0, errors=2.0).generate()
    data_bot.check_degradation("dummy_module", **metrics1)
    data_bot.check_degradation("dummy_module", **metrics2)

    node = registry.graph.nodes["dummy_module"]
    assert node["patch_id"] == 321
    assert node["commit"] == "deadbeef"
    patched_event = next(p for t, p in bus.events if t == "bot:patched")
    assert patched_event["roi_delta"] > 0
    assert any(t == "self_coding:patch_applied" for t, _ in bus.events)
    assert any(t == "bot:patch_applied" for t, _ in bus.events)


def test_failed_validation_triggers_rollback(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")
    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    __import__("dummy_module")

    rollback_calls: list[str] = []
    quick_fix = FailingQuickFix(rollback_calls)
    data_bot, registry, bus = _setup(mod_path, monkeypatch, quick_fix)

    metrics1 = StaticMetricGenerator(roi=1.0).generate()
    metrics2 = StaticMetricGenerator(roi=0.0, errors=2.0).generate()
    data_bot.check_degradation("dummy_module", **metrics1)
    data_bot.check_degradation("dummy_module", **metrics2)

    assert rollback_calls == ["100"]
    topics = [t for t, _ in bus.events]
    assert "bot:patch_failed" in topics
    assert "bot:patched" not in topics
    assert "bot:patch_applied" not in topics
