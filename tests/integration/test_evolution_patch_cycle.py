import sys
import types
import importlib
from pathlib import Path
from menace.coding_bot_interface import manager_generate_helper


def test_evolution_orchestrator_patch_cycle(tmp_path, monkeypatch):
    ROOT = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(tmp_path)
    sys.path.insert(0, str(ROOT))

    # stub modules to avoid heavy imports
    dpr = types.ModuleType("dynamic_path_router")
    dpr.resolve_path = lambda p: Path(p)
    dpr.repo_root = lambda: ROOT
    dpr.resolve_dir = lambda p: Path(p)
    dpr.path_for_prompt = lambda p: str(p)
    sys.modules["dynamic_path_router"] = dpr

    alert_mod = types.ModuleType("alert_dispatcher")
    alert_mod.send_discord_alert = lambda *a, **k: None
    alert_mod.CONFIG = {}
    sys.modules["alert_dispatcher"] = alert_mod

    ueb = types.ModuleType("unified_event_bus")

    class UnifiedEventBus:
        def __init__(self):
            self.subs = {}

        def subscribe(self, topic, fn):
            self.subs.setdefault(topic, []).append(fn)

        def publish(self, topic, payload):
            for fn in self.subs.get(topic, []):
                fn(topic, payload)

    ueb.UnifiedEventBus = UnifiedEventBus
    sys.modules["unified_event_bus"] = ueb

    cbi = types.ModuleType("menace.coding_bot_interface")
    cbi.self_coding_managed = lambda f: f
    sys.modules["menace.coding_bot_interface"] = cbi

    metrics_exporter = types.ModuleType("metrics_exporter")
    metrics_exporter.update_relevancy_metrics = lambda *a, **k: None
    metrics_exporter.Gauge = type("_G", (), {"__call__": lambda self, *a, **k: None})
    sys.modules["metrics_exporter"] = metrics_exporter
    rmdb = types.ModuleType("relevancy_metrics_db")
    rmdb.RelevancyMetricsDB = object
    sys.modules["relevancy_metrics_db"] = rmdb

    scm_stub = types.ModuleType("menace.self_coding_manager")

    class HelperGenerationError(RuntimeError):
        pass

    scm_stub.HelperGenerationError = HelperGenerationError
    sys.modules["menace.self_coding_manager"] = scm_stub

    # create dummy bot module
    mod_path = tmp_path / "dummy_bot.py"
    mod_path.write_text("def foo():\n    return 1\n")
    importlib.invalidate_caches()
    sys.modules.pop("dummy_bot", None)
    __import__("dummy_bot")

    eo_mod = importlib.import_module("menace.evolution_orchestrator")
    EvolutionOrchestrator = eo_mod.EvolutionOrchestrator
    EvolutionTrigger = eo_mod.EvolutionTrigger

    created_builders = []

    class DummyContextBuilder:
        def __init__(self):
            self.built = False
            created_builders.append(self)

        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            self.built = True
            return "", "", []

    monkeypatch.setattr(eo_mod, "create_context_builder", lambda: DummyContextBuilder())
    monkeypatch.setattr(eo_mod, "ensure_fresh_weights", lambda b: b.refresh_db_weights())

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
    bus = DummyBus()

    class DummyDataBot:
        def __init__(self, event_bus):
            self.event_bus = event_bus
            self.cb = None

        def subscribe_degradation(self, cb):
            self.cb = cb

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            if errors > 1.0:
                event = {"bot": bot, "roi": roi, "errors": errors}
                if self.event_bus:
                    self.event_bus.publish("bot:degraded", event)
                elif self.cb:
                    self.cb(event)
                return True
            return False

        def roi(self, bot):
            return 0.0

        def average_errors(self, bot):
            return 0.0

    class DummyGraph(dict):
        @property
        def nodes(self):
            return self

    class DummyBotRegistry:
        def __init__(self, event_bus=None):
            self.event_bus = event_bus
            self.graph = DummyGraph()
            self.updated = None

        def register_bot(self, name, module=None):
            self.graph[name] = {"module": module}
            if self.event_bus:
                self.event_bus.publish("bot:registered", {"bot": name})

        def update_bot(self, name, module, patch_id=None, commit=None):
            self.updated = (name, module, patch_id, commit)
            self.graph[name] = {"module": module, "patch_id": patch_id, "commit": commit}

    class DummyQuickFix:
        def __init__(self):
            self.validated = []

        def validate_patch(self, module_path, code):
            self.validated.append((module_path, code))
            return True

        def apply_patch(self, module_path, code):
            return 123

    class DummyEngine:
        def __init__(self):
            self.cognition_layer = types.SimpleNamespace(context_builder=None)
            self.called_builder = None

        def generate_helper(self, desc, **kwargs):
            self.called_builder = self.cognition_layer.context_builder
            return "helper"

    class SelfCodingManager:
        def __init__(self, engine, quick_fix, bot_registry, data_bot, bot_name, event_bus):
            self.engine = engine
            self.quick_fix = quick_fix
            self.bot_registry = bot_registry
            self.data_bot = data_bot
            self.bot_name = bot_name
            self.event_bus = event_bus
            self.cycle_registered = False
            self._last_patch_id = None
            self.bot_registry.register_bot(bot_name, module=str(mod_path))
            self.run_patch_called = False

        def register_patch_cycle(self, description, context_meta=None):
            self.cycle_registered = True
            if self.event_bus:
                self.event_bus.publish(
                    "self_coding:cycle_registered",
                    {"bot": self.bot_name, "description": description},
                )

        def should_refactor(self):
            return True

        def run_patch(self, path, description, *, context_meta=None, context_builder=None):
            self.run_patch_called = True
            return None, "deadbeef"

        def generate_and_patch(self, path, description, *, context_meta=None, context_builder=None):
            context_builder.build(description)
            self.engine.cognition_layer.context_builder = context_builder
            code = manager_generate_helper(self, description, path=str(path))
            self.quick_fix.validate_patch(str(path), code)
            patch_id = self.quick_fix.apply_patch(str(path), code)
            self._last_patch_id = patch_id
            self.bot_registry.update_bot(
                self.bot_name, str(path), patch_id=patch_id, commit="deadbeef"
            )
            return self.run_patch(
                path,
                description,
                context_meta=context_meta,
                context_builder=context_builder,
            )

    class DummyCapital:
        pass

    class DummyImprovement:
        pass

    class DummyEvolution:
        pass

    class DummyHistoryDB:
        def add(self, *a, **k):
            pass

    data_bot = DummyDataBot(bus)
    registry = DummyBotRegistry(event_bus=bus)
    engine = DummyEngine()
    quick_fix = DummyQuickFix()
    manager = SelfCodingManager(engine, quick_fix, registry, data_bot, "dummy_bot", bus)

    _ = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
        triggers=EvolutionTrigger(error_rate=0.1, roi_drop=-0.1),
    )

    data_bot.check_degradation("dummy_bot", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_bot", roi=0.5, errors=2.0)

    assert manager.cycle_registered
    assert engine.called_builder is created_builders[0]
    assert quick_fix.validated
    assert registry.updated == ("dummy_bot", str(mod_path), 123, "deadbeef")
    assert len(created_builders) == 1 and created_builders[0].built
    assert manager.run_patch_called
