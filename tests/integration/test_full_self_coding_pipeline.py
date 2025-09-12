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


class SelfCodingManager:
    def __init__(self, **kwargs):
        self.quick_fix = kwargs.get("quick_fix")
        self.bot_name = kwargs.get("bot_name", "")
        self.bot_registry = kwargs.get("bot_registry")
        self.data_bot = kwargs.get("data_bot")
        self.engine = kwargs.get("engine")
        self.event_bus = kwargs.get("event_bus")
        self._last_patch_id = None
        if self.bot_registry:
            self.bot_registry.register_bot(self.bot_name)

    def register_patch_cycle(self, description, context_meta=None):
        roi = 0.0
        errors = 0.0
        patch_db = getattr(self.engine, "patch_db", None)
        if patch_db:
            self._last_patch_id = patch_db.add(
                filename=f"{self.bot_name}.cycle",
                description=description,
                roi_before=roi,
                roi_after=roi,
                errors_before=int(errors),
                errors_after=int(errors),
            )
        if self.event_bus:
            payload = {
                "bot": self.bot_name,
                "patch_id": self._last_patch_id,
                "roi_before": roi,
                "errors_before": errors,
            }
            if context_meta:
                payload.update(context_meta)
            self.event_bus.publish("self_coding:cycle_registered", payload)

    def generate_and_patch(
        self, path: Path, description: str, *, context_meta=None, context_builder=None
    ):
        if context_builder:
            try:
                context_builder.build_helper(self.bot_name)
            except Exception:
                pass
        passed, patch_id = self.quick_fix.apply_validated_patch(
            str(path), description, context_meta or {}
        )
        self._last_patch_id = patch_id
        commit = "deadbeef"
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:patch_applied",
                {
                    "bot": self.bot_name,
                    "patch_id": patch_id,
                    "path": str(path),
                    "description": description,
                    "roi_before": 0.0,
                    "roi_after": 0.0,
                    "roi_delta": 0.0,
                },
            )
        return None, commit


def should_refactor(self) -> bool:
    return True


SelfCodingManager.should_refactor = should_refactor
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.HelperGenerationError = RuntimeError
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
        self.published = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.published.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummyContextBuilder:
    last = None

    def __init__(self):
        self.calls = []
        DummyContextBuilder.last = self

    def build_helper(self, bot_name):
        self.calls.append(bot_name)
        return Path("helper.py")

    def refresh_db_weights(self):
        pass


ea_mod_ContextBuilder = DummyContextBuilder


class QuickFixEngine:
    def __init__(self):
        self.calls = []

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc, ctx_meta))
        return True, 42


class DummyPatchDB:
    def __init__(self):
        self.records = []
        self.counter = 0

    def add(self, **rec):
        self.counter += 1
        rec["id"] = self.counter
        self.records.append(rec)
        return self.counter


class DummyCapital:
    def energy_score(self, *a, **k):
        return 1.0


class DummyImprovement:
    pass


class DummyEvolution:
    pass


class DummyHistoryDB:
    def __init__(self):
        self.events = []

    def add(self, event):
        self.events.append(event)


def test_full_self_coding_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(eo_mod, "ContextBuilder", DummyContextBuilder)
    monkeypatch.syspath_prepend(tmp_path)
    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text(
        "from menace.coding_bot_interface import self_coding_managed\n\n"
        "@self_coding_managed\n"
        "def foo():\n    return 1\n"
    )

    importlib.invalidate_caches()
    sys.modules.pop("dummy_module", None)
    __import__("dummy_module")

    bus = DummyBus()
    patch_db = DummyPatchDB()
    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)
    registry.hot_swap_bot = lambda *_a, **_k: None
    quick_fix = QuickFixEngine()
    manager = SelfCodingManager(
        quick_fix=quick_fix,
        bot_name="dummy_module",
        bot_registry=registry,
        data_bot=data_bot,
        engine=types.SimpleNamespace(patch_db=patch_db),
        event_bus=bus,
    )
    history = DummyHistoryDB()
    monkeypatch.setattr(eo_mod, "create_context_builder", lambda: DummyContextBuilder())
    EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        DummyImprovement(),
        DummyEvolution(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=history,
    )

    data_bot.check_degradation("dummy_module", roi=1.0, errors=0.0)
    data_bot.check_degradation("dummy_module", roi=0.0, errors=2.0)

    assert quick_fix.calls
    path_called, desc, _ = quick_fix.calls[0]
    assert Path(path_called) == mod_path
    assert desc.startswith("auto_patch_due_to_degradation")

    assert DummyContextBuilder.last and DummyContextBuilder.last.calls == ["dummy_module"]

    cycle_event = next(p for t, p in bus.published if t == "self_coding:cycle_registered")
    assert cycle_event["bot"] == "dummy_module"

    patch_event = next(p for t, p in bus.published if t == "self_coding:patch_applied")
    assert patch_event["patch_id"] == 42

    patched = next(p for t, p in bus.published if t == "bot:patched")
    assert patched["bot"] == "dummy_module"
    assert patched["commit"] == "deadbeef"
    assert registry.graph.nodes["dummy_module"]["commit"] == "deadbeef"
