import importlib
import sys
import types
from pathlib import Path


def test_auto_registration_patch_hot_swap(tmp_path, monkeypatch):
    registry_mod = importlib.import_module("menace.bot_registry")
    BotRegistry = registry_mod.BotRegistry
    # Avoid writing threshold files during test
    monkeypatch.setattr(registry_mod, "persist_sc_thresholds", lambda *a, **k: None)

    registry = BotRegistry()
    registry.hot_swap_bot = lambda *a, **k: None
    registry.health_check_bot = lambda *a, **k: None

    class DummyDataBot:
        def __init__(self):
            self._callbacks = []

        def reload_thresholds(self, _name):
            return types.SimpleNamespace(
                roi_drop=-0.1, error_threshold=1.0, test_failure_threshold=0.0
            )

        def subscribe_degradation(self, cb):
            self._callbacks.append(cb)

        def emit_degradation(self, bot):
            event = {"bot": bot}
            for cb in list(self._callbacks):
                cb(event)

    data_bot = DummyDataBot()

    class DummySelfCodingManager:
        def __init__(self, **kwargs):
            self.bot_name = kwargs.get("bot_name", "")
            self.bot_registry = kwargs.get("bot_registry")
            self.data_bot = kwargs.get("data_bot")
            self.quick_fix = object()
            self.register_calls: list[tuple[str, dict | None]] = []
            self.patch_calls: list[tuple[Path, str]] = []
            self._last_patch_id = None
            self._last_commit_hash = None

        def register_patch_cycle(self, desc: str, ctx: dict | None = None):
            self.register_calls.append((desc, ctx))
            self._last_patch_id = 1
            self._last_commit_hash = "deadbeef"

        def run_patch(self, path: Path, desc: str, *, context_meta=None, context_builder=None):
            self.patch_calls.append((path, desc))
            if self.bot_registry:
                self.bot_registry.update_bot(
                    self.bot_name, str(path), patch_id=1, commit="deadbeef"
                )

    scm_mod = types.ModuleType("menace.self_coding_manager")
    scm_mod.SelfCodingManager = DummySelfCodingManager
    sys.modules["menace.self_coding_manager"] = scm_mod

    sce_mod = types.ModuleType("menace.self_coding_engine")
    import contextvars
    sce_mod.MANAGER_CONTEXT = contextvars.ContextVar("MANAGER_CONTEXT")
    sys.modules["menace.self_coding_engine"] = sce_mod

    qf_mod = types.ModuleType("menace.quick_fix_engine")
    qf_mod.QuickFixEngine = object
    sys.modules["menace.quick_fix_engine"] = qf_mod

    err_mod = types.ModuleType("menace.error_bot")
    err_mod.ErrorDB = object
    sys.modules["menace.error_bot"] = err_mod

    reg_holder = types.ModuleType("reg_holder")
    reg_holder.registry = registry
    sys.modules["reg_holder"] = reg_holder

    db_holder = types.ModuleType("db_holder")
    db_holder.data_bot = data_bot
    sys.modules["db_holder"] = db_holder

    mod_path = tmp_path / "dummy_mod.py"
    mod_path.write_text(
        "from menace.coding_bot_interface import self_coding_managed\n"
        "from reg_holder import registry\n"
        "from db_holder import data_bot\n"
        "@self_coding_managed(bot_registry=registry, data_bot=data_bot)\n"
        "class DummyBot:\n"
        "    name = 'dummy'\n"
        "    def __init__(self, manager=None, evolution_orchestrator=None):\n"
        "        if manager is not None:\n"
        "            self.manager = manager\n"
        "        if evolution_orchestrator is not None:\n"
        "            self.evolution_orchestrator = evolution_orchestrator\n"
    )

    sys.path.insert(0, str(tmp_path))
    dummy_mod = importlib.import_module("dummy_mod")

    manager = DummySelfCodingManager(
        bot_name="dummy", bot_registry=registry, data_bot=data_bot
    )
    dummy_mod.DummyBot(
        manager=manager,
        evolution_orchestrator=types.SimpleNamespace(register_bot=lambda *_: None),
    )
    registry.graph.nodes["dummy"]["module"] = str(mod_path)

    assert "dummy" in registry.graph

    data_bot.emit_degradation("dummy")

    assert manager.register_calls
    assert registry.graph.nodes["dummy"]["patch_history"]

