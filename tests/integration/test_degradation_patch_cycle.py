import importlib
import sys
import types
from pathlib import Path

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

scm_mod = types.ModuleType("menace.self_coding_manager")


class HelperGenerationError(RuntimeError):
    pass


class SelfCodingManager:
    def __init__(self, **kwargs):
        self.quick_fix = kwargs.get("quick_fix")
        self.bot_name = kwargs.get("bot_name", "")
        self.bot_registry = kwargs.get("bot_registry")
        self.called = False
        self.event_bus = kwargs.get("event_bus")
        self._last_patch_id = None
        if self.bot_registry:
            self.bot_registry.register_bot(self.bot_name)

    def generate_and_patch(
        self, path: Path, description: str, *, context_meta=None, context_builder=None
    ):
        self.called = True
        passed, patch_id = self.quick_fix.apply_validated_patch(
            str(path), description, context_meta or {}
        )
        self._last_patch_id = patch_id
        return None, "deadbeef"

    def register_patch_cycle(self, description, context_meta=None):  # noqa: D401,D403
        """Publish registration event for assertions."""
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered",
                {"bot": self.bot_name, "description": description},
            )

    def should_refactor(self) -> bool:  # noqa: D401 - simple always-true stub
        return True


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
        self.events = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.events.append((topic, payload))
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
    sys.modules.pop("dummy_module", None)
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

    data_bot.check_degradation("dummy_module", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_module", roi=0.0, errors=2.0)

    assert manager.called
    assert quick_fix.calls
    node = registry.graph.nodes["dummy_module"]
    assert node["module"] == str(mod_path)
    assert node["patch_id"] == 123
    assert node["commit"] == "deadbeef"

    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics
    assert "bot:hot_swapped" in topics


def test_bot_degraded_event_triggers_patch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")

    import importlib

    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    __import__("dummy_module")

    bus = DummyBus()
    monkeypatch.setattr(eo_mod, "ContextBuilder", DummyContextBuilder)

    registry = BotRegistry(event_bus=bus)
    registry.graph.add_node("dummy_module", module=str(mod_path))

    quick_fix = DummyQuickFix()

    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_module",
        bot_registry=registry,
        event_bus=bus,
    )

    calls: list[str] = []

    def fake_update_bot(name, module_path, patch_id=None, commit=None):
        calls.append(module_path)
        registry.graph.add_node(name)
        registry.graph.nodes[name]["module"] = module_path

    monkeypatch.setattr(registry, "update_bot", fake_update_bot)

    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
    )

    bus.publish("bot:degraded", {"bot": "dummy_module"})

    assert manager.called
    assert calls and calls[0] == str(mod_path)

    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics
    assert "bot:hot_swapped" in topics


def test_decorated_bot_triggers_degradation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text(
        "from menace.coding_bot_interface import self_coding_managed\n"
        "@self_coding_managed\n"
        "class DummyBot:\n"
        "    name = 'dummy_module'\n"
        "    def __init__(self, manager=None, evolution_orchestrator=None,\n"
        "                 bot_registry=None, data_bot=None):\n"
        "        pass\n"
    )

    import importlib

    def _decorator(cls):
        orig_init = cls.__init__

        def _wrapped(self, *a, **kw):
            orch = kw.get("evolution_orchestrator")
            orig_init(self, *a, **kw)
            if orch:
                orch.register_bot(self.name)

        cls.__init__ = _wrapped
        return cls

    monkeypatch.setattr(cbi, "self_coding_managed", _decorator)

    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    dummy_module = importlib.import_module("dummy_module")

    bus = DummyBus()
    monkeypatch.setattr(eo_mod, "ContextBuilder", DummyContextBuilder)

    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)
    quick_fix = DummyQuickFix()
    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_module",
        bot_registry=registry,
        event_bus=bus,
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

    dummy_module.DummyBot(
        manager=manager,
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orch,
    )

    data_bot.check_degradation("dummy_module", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_module", roi=0.0, errors=2.0)

    assert manager.called
    assert quick_fix.calls
    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics
    assert "bot:hot_swapped" in topics
