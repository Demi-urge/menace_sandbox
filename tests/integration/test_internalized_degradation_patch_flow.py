import sys
import types
import importlib
from pathlib import Path

# Prepare menace package root for dynamic imports
ROOT = Path(__file__).resolve().parents[2]
package = types.ModuleType("menace")
package.__path__ = [str(ROOT)]
sys.modules.setdefault("menace", package)

# Minimal dynamic_path_router to avoid filesystem interactions
_dpr = types.ModuleType("dynamic_path_router")
_dpr.resolve_path = lambda p: Path(p)
_dpr.repo_root = lambda: ROOT
_dpr.resolve_dir = lambda p: Path(p)
_dpr.path_for_prompt = lambda p: str(p)
sys.modules.setdefault("dynamic_path_router", _dpr)

# Stub sandbox runner harness
sr_pkg = types.ModuleType("sandbox_runner")
sr_pkg.__path__ = []
sys.modules.setdefault("sandbox_runner", sr_pkg)
sys.modules.setdefault("menace.sandbox_runner", sr_pkg)
th = types.ModuleType("sandbox_runner.test_harness")
th.run_tests = lambda *a, **k: types.SimpleNamespace(success=True, stdout="", duration=0)
th.TestHarnessResult = types.SimpleNamespace
sys.modules.setdefault("sandbox_runner.test_harness", th)
sys.modules.setdefault("menace.sandbox_runner.test_harness", th)

# Stub coding bot interface before importing helper
cbi_stub = types.ModuleType("menace.coding_bot_interface")
def manager_generate_helper(manager, description, path=None):
    return manager.engine.generate_helper(description)
cbi_stub.manager_generate_helper = manager_generate_helper
cbi_stub.self_coding_managed = lambda cls: cls
sys.modules.setdefault("menace.coding_bot_interface", cbi_stub)
from menace.coding_bot_interface import manager_generate_helper

# Dummy event bus to capture interactions
class DummyBus:
    def __init__(self):
        self.subs: dict[str, list] = {}
        self.events: list[tuple[str, object]] = []

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        self.events.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)

# Provide unified event bus stub before importing data_bot
bus_mod = types.ModuleType("menace.unified_event_bus")
bus_mod.UnifiedEventBus = DummyBus
sys.modules.setdefault("menace.unified_event_bus", bus_mod)

# Import core modules after stubbing environment
DataBot = importlib.import_module("menace.data_bot").DataBot
MetricsDB = importlib.import_module("menace.data_bot").MetricsDB
BotRegistry = importlib.import_module("menace.bot_registry").BotRegistry

# Dummy context builder and util hooks
class DummyContextBuilder:
    def build(self, description):
        return None

    def refresh_db_weights(self):
        pass

context_builder_util = importlib.import_module("context_builder_util")
context_builder_util.create_context_builder = lambda: DummyContextBuilder()
context_builder_util.ensure_fresh_weights = lambda *a, **k: None

# Stubs for engine and quick fix components
class DummyEngine:
    def __init__(self):
        self.calls: list[str] = []

    def generate_helper(self, desc, **kwargs):
        self.calls.append(desc)
        return "code"

class DummyQuickFix:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def apply_validated_patch(self, module_path, desc, ctx_meta):
        self.calls.append((module_path, desc))
        return True, 123, []

# Self-coding manager stub implementing generate_and_patch
class SelfCodingManager:
    def __init__(
        self,
        engine,
        pipeline,
        *,
        bot_name,
        data_bot,
        bot_registry,
        event_bus=None,
        quick_fix=None,
    ):
        self.engine = engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.bot_registry = bot_registry
        self.event_bus = event_bus
        self.quick_fix = quick_fix
        self.gen_calls: list[tuple[Path, str]] = []
        self._last_patch_id = None
        self._last_commit_hash = None

    def register_patch_cycle(
        self, description, context_meta=None, provenance_token=None, **kwargs
    ):
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered",
                {"bot": self.bot_name, "description": description},
            )
        self._last_patch_id = 123
        self._last_commit_hash = "commit-hash"
        return 123, "commit-hash"

    def generate_and_patch(
        self,
        path: Path,
        description: str,
        *,
        context_meta=None,
        context_builder=None,
        provenance_token=None,
        **kwargs,
    ):
        self.gen_calls.append((path, description))
        manager_generate_helper(self, description, path=str(path))
        if self.quick_fix:
            self.quick_fix.apply_validated_patch(str(path), description, context_meta or {})
        self._last_patch_id = 123
        self._last_commit_hash = "commit-hash"
        self.bot_registry.update_bot(
            self.bot_name, str(path), patch_id=123, commit="commit-hash"
        )
        if self.event_bus:
            self.event_bus.publish(
                "bot:patch_applied", {"bot": self.bot_name, "commit": "commit-hash"}
            )
        return None, "commit-hash"

# internalize_coding_bot helper wiring manager and orchestrator

def internalize_coding_bot(
    bot_name,
    engine,
    pipeline,
    *,
    data_bot,
    bot_registry,
    evolution_orchestrator=None,
    quick_fix=None,
    roi_threshold=None,
    error_threshold=None,
):
    bus = getattr(evolution_orchestrator, "event_bus", None)
    manager = SelfCodingManager(
        engine,
        pipeline,
        bot_name=bot_name,
        data_bot=data_bot,
        bot_registry=bot_registry,
        event_bus=bus,
        quick_fix=quick_fix,
    )
    bot_registry.register_bot(bot_name, manager=manager, data_bot=data_bot, is_coding_bot=True)
    data_bot._thresholds[bot_name] = (roi_threshold or 0.1, error_threshold or 0.1)
    if evolution_orchestrator is not None:
        evolution_orchestrator.selfcoding_manager = manager
        try:
            evolution_orchestrator.register_bot(bot_name)
        except Exception:
            pass
        if bus:
            bus.subscribe(
                "degradation:detected",
                lambda _t, e: evolution_orchestrator.register_patch_cycle(e),
            )
    return manager

# Expose stub manager module
scm_mod = types.ModuleType("menace.self_coding_manager")
scm_mod.SelfCodingManager = SelfCodingManager
scm_mod.internalize_coding_bot = internalize_coding_bot
sys.modules.setdefault("menace.self_coding_manager", scm_mod)

# Minimal EvolutionOrchestrator triggering generate_and_patch
class EvolutionOrchestrator:
    def __init__(self, *, event_bus, module_path: Path):
        self.event_bus = event_bus
        self.module_path = module_path
        self.selfcoding_manager = None

    def register_bot(self, bot: str) -> None:  # pragma: no cover - simple stub
        pass

    def register_patch_cycle(self, event: dict) -> None:
        desc = f"auto_patch_due_to_degradation:{event.get('bot', '')}"
        if self.selfcoding_manager:
            self.selfcoding_manager.generate_and_patch(
                self.module_path,
                desc,
                context_meta=event,
                context_builder=object(),
                provenance_token="prov",
            )

# ---------------------------------------------------------------------------
# Test begins


def test_internalized_degradation_patch_flow(tmp_path, monkeypatch):
    bus = DummyBus()
    data_bot = DataBot(MetricsDB(tmp_path / "metrics.db"), event_bus=bus)
    registry = BotRegistry(event_bus=bus)

    update_calls: list[tuple[str, str, int | None, str | None]] = []

    def fake_update_bot(name, module_path, patch_id=None, commit=None):
        update_calls.append((name, module_path, patch_id, commit))

    monkeypatch.setattr(registry, "update_bot", fake_update_bot)

    mod_path = tmp_path / "dummy_module.py"
    mod_path.write_text("def foo():\n    return 1\n")

    engine = DummyEngine()
    quick_fix = DummyQuickFix()
    pipeline = object()
    orch = EvolutionOrchestrator(event_bus=bus, module_path=mod_path)

    internalize_coding_bot(
        "dummy_bot",
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        evolution_orchestrator=orch,
        quick_fix=quick_fix,
    )

    data_bot.check_degradation("dummy_bot", roi=0.0, errors=2.0)

    assert orch.selfcoding_manager.gen_calls, "generate_and_patch not invoked"
    assert quick_fix.calls, "quick fix engine not used"
    assert update_calls and update_calls[-1][3] == "commit-hash", "update_bot not called"
    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics, "patch cycle not registered"
    assert "bot:patch_applied" in topics, "patch not applied"
