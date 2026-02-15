import importlib
import logging
import sys
import types

import pytest


@pytest.mark.parametrize("package_alias", ["menace", "menace_sandbox"])
def test_initialize_deferred_components_error_bot_import_is_local(monkeypatch, package_alias):
    """Deferred initialization should not fail to resolve ErrorBot for either alias."""

    try:
        scm = importlib.import_module(f"{package_alias}.self_coding_manager")
    except (ModuleNotFoundError, ImportError) as exc:
        pytest.skip(f"{package_alias} alias is not available in this environment: {exc}")

    error_bot_module = types.ModuleType("menace.error_bot")

    class ErrorDB:
        pass

    class ErrorBot:
        def __init__(self, error_db, metrics_db, context_builder=None):
            self.error_db = error_db
            self.metrics_db = metrics_db
            self.context_builder = context_builder

    error_bot_module.ErrorBot = ErrorBot
    error_bot_module.ErrorDB = ErrorDB
    monkeypatch.setitem(sys.modules, "menace.error_bot", error_bot_module)
    monkeypatch.setitem(sys.modules, "menace_sandbox.error_bot", error_bot_module)

    menace_error_bot = importlib.import_module("menace.error_bot")
    menace_sandbox_error_bot = importlib.import_module("menace_sandbox.error_bot")
    assert menace_error_bot is menace_sandbox_error_bot

    data_bot_module = types.ModuleType(f"{package_alias}.data_bot")

    class MetricsDB:
        pass

    data_bot_module.MetricsDB = MetricsDB
    monkeypatch.setitem(sys.modules, f"{package_alias}.data_bot", data_bot_module)

    cap_module = types.ModuleType(f"{package_alias}.capital_management_bot")
    cap_module.CapitalManagementBot = lambda *a, **k: object()
    monkeypatch.setitem(sys.modules, f"{package_alias}.capital_management_bot", cap_module)

    sem_module = types.ModuleType(f"{package_alias}.system_evolution_manager")
    sem_module.SystemEvolutionManager = lambda *_a, **_k: object()
    monkeypatch.setitem(sys.modules, f"{package_alias}.system_evolution_manager", sem_module)

    evo_module = types.ModuleType(f"{package_alias}.evolution_orchestrator")
    evo_module.EvolutionOrchestrator = lambda *_a, **_k: object()
    monkeypatch.setitem(sys.modules, f"{package_alias}.evolution_orchestrator", evo_module)

    self_improvement_pkg = types.ModuleType(f"{package_alias}.self_improvement")
    self_improvement_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, f"{package_alias}.self_improvement", self_improvement_pkg)

    engine_module = types.ModuleType(f"{package_alias}.self_improvement.engine")

    class StubSelfImprovementEngine:
        def __init__(self, context_builder, *_args, **_kwargs):
            from ..error_bot import ErrorBot, ErrorDB
            from ..data_bot import MetricsDB

            self.error_bot = ErrorBot(
                ErrorDB(),
                MetricsDB(),
                context_builder=context_builder,
            )

    engine_module.SelfImprovementEngine = StubSelfImprovementEngine
    monkeypatch.setitem(sys.modules, f"{package_alias}.self_improvement.engine", engine_module)

    manager = object.__new__(scm.SelfCodingManager)
    manager.logger = logging.getLogger(f"test-scm-{package_alias}")
    manager.bot_name = f"{package_alias}-bot"
    manager.data_bot = object()
    manager.pipeline = types.SimpleNamespace(_pipeline_promoter=None)
    manager._context_builder = object()
    manager.evolution_orchestrator = None
    manager.bot_registry = types.SimpleNamespace(graph={})
    manager._bootstrap_owner_token = None
    manager._bootstrap_provenance_token = None
    manager._mark_construction_phase = lambda *_a, **_k: None
    manager._ensure_broker_owner_ready = lambda **_k: None

    manager.initialize_deferred_components()
