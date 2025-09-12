import importlib
import sys
import types
import contextvars
from pathlib import Path


def test_instantiated_bot_triggers_patch(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(tmp_path)
    ROOT = Path(__file__).resolve().parents[2]
    package = types.ModuleType("menace")
    package.__path__ = [str(ROOT)]
    sys.modules["menace"] = package

    dpr = types.ModuleType("dynamic_path_router")
    dpr.resolve_path = lambda p: Path(p)
    dpr.repo_root = lambda: ROOT
    dpr.resolve_dir = lambda p: Path(p)
    dpr.path_for_prompt = lambda p: str(p)
    sys.modules["dynamic_path_router"] = dpr

    sr_pkg = types.ModuleType("sandbox_runner")
    sr_pkg.__path__ = []
    sys.modules["sandbox_runner"] = sr_pkg
    th = types.ModuleType("sandbox_runner.test_harness")
    th.run_tests = lambda *a, **k: types.SimpleNamespace(success=True, stdout="", duration=0)
    th.TestHarnessResult = types.SimpleNamespace
    sys.modules["sandbox_runner.test_harness"] = th

    scm_mod = types.ModuleType("menace.self_coding_manager")

    class HelperGenerationError(RuntimeError):
        pass

    class SelfCodingManager:
        def __init__(self, **kwargs):
            self.quick_fix = kwargs.get("quick_fix")
            self.bot_name = kwargs.get("bot_name", "")
            self.bot_registry = kwargs.get("bot_registry")
            self.called = False
            if self.bot_registry:
                self.bot_registry.register_bot(self.bot_name)

        def run_patch(self, path: Path, description: str, *, context_meta=None, context_builder=None):
            self.called = True
            self.quick_fix.apply_validated_patch(str(path), description, context_meta or {})
            self.bot_registry.update_bot(self.bot_name, str(path), patch_id=123, commit="deadbeef")

        def register_patch_cycle(self, *_, **__):
            pass

        def should_refactor(self) -> bool:
            return True

    scm_mod.SelfCodingManager = SelfCodingManager
    scm_mod.HelperGenerationError = HelperGenerationError
    sys.modules["menace.self_coding_manager"] = scm_mod

    sce_mod = types.ModuleType("menace.self_coding_engine")
    sce_mod.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
    sys.modules["menace.self_coding_engine"] = sce_mod

    sys.modules.pop("menace.coding_bot_interface", None)
    cbi = types.ModuleType("menace.coding_bot_interface")

    def self_coding_managed(cls):
        orig_init = cls.__init__

        def wrapped(self, *a, **kw):
            orch = kw.get("evolution_orchestrator")
            orig_init(self, *a, **kw)
            if orch:
                orch.register_bot(self.name)

        cls.__init__ = wrapped
        return cls

    cbi.self_coding_managed = self_coding_managed
    sys.modules["menace.coding_bot_interface"] = cbi
    from menace.coding_bot_interface import self_coding_managed

    class DummyBus:
        def __init__(self):
            self.subs = {}

        def subscribe(self, topic, fn):
            self.subs.setdefault(topic, []).append(fn)

        def publish(self, topic, payload):
            for fn in self.subs.get(topic, []):
                fn(topic, payload)

    class DummyQuickFix:
        def __init__(self):
            self.calls = []

        def apply_validated_patch(self, module_path, desc, ctx_meta):
            self.calls.append((module_path, desc))
            return True, 123

    class DummyDataBot:
        def __init__(self, event_bus=None):
            self.event_bus = event_bus
            self.db = types.SimpleNamespace(log_eval=lambda *a, **k: None)
            self._callbacks = []

        def roi(self, _name):  # pragma: no cover - simple
            return 1.0

        def subscribe_degradation(self, cb):  # pragma: no cover - simple
            self._callbacks.append(cb)

        def check_degradation(self, bot, roi, errors, test_failures=0.0):
            event = {
                "bot": bot,
                "delta_roi": roi - 1.0,
                "delta_errors": float(errors),
                "roi_threshold": -0.1,
                "error_threshold": 1.0,
                "test_failures": test_failures,
                "test_failure_threshold": 0.0,
            }
            degraded = roi < 1.0 or errors > 0.0
            if degraded:
                for cb in self._callbacks:
                    cb(event)
            return degraded

    class DummyRegistry:
        def __init__(self):
            self.graph = {}

        def register_bot(self, name):
            self.graph[name] = {"module": ""}

        def update_bot(self, name, module, **extra):
            self.graph[name].update({"module": module, **extra})

    class EvolutionOrchestrator:
        def __init__(self, data_bot, selfcoding_manager):
            self.data_bot = data_bot
            self.selfcoding_manager = selfcoding_manager
            self._registered_bots = set()

        def register_bot(self, bot):
            if bot in self._registered_bots:
                return
            self._registered_bots.add(bot)
            self.data_bot.subscribe_degradation(self._on_bot_degraded)
            self.data_bot.check_degradation(bot, roi=0.0, errors=0.0)

        def _on_bot_degraded(self, event):
            path = Path(__file__)
            desc = f"auto_patch_due_to_degradation:{event.get('bot')}"
            self.selfcoding_manager.run_patch(path, desc, context_meta=event)

    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text(
        "from menace.coding_bot_interface import self_coding_managed\n"
        "@self_coding_managed\n"
        "class DummyBot:\n"
        "    name = 'dummy_module'\n"
        "    def __init__(self, manager=None, evolution_orchestrator=None, **kwargs):\n"
        "        if manager is not None:\n"
        "            self.manager = manager\n"
        "        if evolution_orchestrator is not None:\n"
        "            self.evolution_orchestrator = evolution_orchestrator\n"
    )

    importlib.invalidate_caches()
    dummy_module = importlib.import_module("dummy_module")

    bus = DummyBus()
    data_bot = DummyDataBot()
    registry = DummyRegistry()
    quick_fix = DummyQuickFix()
    manager = SelfCodingManager(quick_fix=quick_fix, bot_name="dummy_module", bot_registry=registry)
    orch = EvolutionOrchestrator(data_bot, manager)

    dummy_module.DummyBot(
        manager=manager,
        bot_registry=registry,
        data_bot=data_bot,
        evolution_orchestrator=orch,
    )

    assert "dummy_module" in orch._registered_bots

    data_bot.check_degradation("dummy_module", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_module", roi=0.0, errors=2.0)

    assert manager.called
    assert quick_fix.calls
