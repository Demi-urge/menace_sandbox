import sys
import types
import importlib
from pathlib import Path


def test_unique_builder_each_cycle(monkeypatch, tmp_path):
    ROOT = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    sys.path.insert(0, str(ROOT))

    dpr = types.ModuleType("dynamic_path_router")
    dpr.resolve_path = lambda p: Path(p)
    dpr.repo_root = lambda: ROOT
    dpr.resolve_dir = lambda p: Path(p)
    dpr.path_for_prompt = lambda p: str(p)
    monkeypatch.setitem(sys.modules, "dynamic_path_router", dpr)

    mod_path = tmp_path / "dummy.py"
    mod_path.write_text("def foo():\n    return 1\n")
    importlib.invalidate_caches()

    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = [str(ROOT)]
    sys.modules.setdefault("menace", menace_pkg)
    eo_mod = importlib.import_module("menace.evolution_orchestrator")

    created = []

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, description, session_id=None, include_vectors=False):
            return "", "", []

    def fake_create_context_builder():
        builder = DummyBuilder()
        created.append(builder)
        return builder

    monkeypatch.setattr(eo_mod, "create_context_builder", fake_create_context_builder)
    monkeypatch.setattr(eo_mod, "ensure_fresh_weights", lambda b: b.refresh_db_weights())

    class DummySelfCodingManager:
        def __init__(self):
            self.bot_name = "dummy"
            self.bot_registry = types.SimpleNamespace(graph={"dummy": {"module": str(mod_path)}})
            self.builder_ids = []
            self.event_bus = None

        def register_patch_cycle(self, desc, ctx=None):
            pass

        def should_refactor(self):
            return True

        def generate_and_patch(self, path, description, *, context_meta=None, context_builder=None):
            self.builder_ids.append(id(context_builder))
            return None, "deadbeef"

    manager = DummySelfCodingManager()

    class DummyDataBot:
        def __init__(self):
            self.event_bus = None
            self.settings = types.SimpleNamespace(sandbox_data_dir=".")

        def roi(self, bot):
            return 0.0

        def average_errors(self, bot):
            return 0.0

        def average_test_failures(self, bot):
            return 0.0

        def log_evolution_cycle(self, *a, **k):
            pass

    data_bot = DummyDataBot()
    capital = types.SimpleNamespace(trend_predictor=object())
    hist = types.SimpleNamespace(add=lambda *a, **k: None)

    orch = eo_mod.EvolutionOrchestrator(
        data_bot,
        capital,
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        history_db=hist,
        triggers=eo_mod.EvolutionTrigger(error_rate=0.1, roi_drop=-0.1),
    )

    event = {"bot": "dummy"}
    orch._on_bot_degraded(event)
    orch._on_bot_degraded(event)

    assert len(manager.builder_ids) == 2
    assert manager.builder_ids[0] != manager.builder_ids[1]
