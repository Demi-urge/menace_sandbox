from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import coding_bot_interface as cbi


def _reset_cbi_state() -> None:
    broker = cbi._bootstrap_dependency_broker()
    broker.clear()
    cbi._PREPARE_PIPELINE_WATCHDOG.clear()
    for attr in (
        "depth",
        "sentinel_manager",
        "pipeline",
        "owner_depths",
        "active_bootstrap_guard",
    ):
        if hasattr(cbi._BOOTSTRAP_STATE, attr):
            delattr(cbi._BOOTSTRAP_STATE, attr)


def test_research_aggregator_reuses_broker_pipeline(monkeypatch):
    _reset_cbi_state()
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]
    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)

    monkeypatch.setattr(cbi, "prepare_pipeline_for_bootstrap", mock.Mock())

    module = importlib.import_module("menace_sandbox.research_aggregator_bot")
    monkeypatch.setattr(module, "registry", None)
    monkeypatch.setattr(module, "data_bot", None)
    monkeypatch.setattr(module, "_context_builder", None)
    monkeypatch.setattr(module, "engine", None)
    monkeypatch.setattr(module, "_PipelineCls", None)
    monkeypatch.setattr(module, "pipeline", None)
    monkeypatch.setattr(module, "evolution_orchestrator", None)
    monkeypatch.setattr(module, "manager", None)
    monkeypatch.setattr(module, "_runtime_state", None)
    monkeypatch.setattr(module, "_runtime_placeholder", None)
    monkeypatch.setattr(module, "_runtime_initializing", False)
    monkeypatch.setattr(module, "_self_coding_configured", False)

    class _Registry:
        pass

    class _DataBot:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return None

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    promotions: list[object] = []

    def _promote(manager: object | None) -> None:
        promotions.append(manager)

    monkeypatch.setattr(module, "BotRegistry", _Registry)
    monkeypatch.setattr(module, "DataBot", _DataBot)
    monkeypatch.setattr(module, "ContextBuilder", _ContextBuilder)
    monkeypatch.setattr(module, "create_context_builder", mock.Mock(return_value=_ContextBuilder()))
    monkeypatch.setattr(module, "SelfCodingEngine", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "CodeDB", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "GPTMemoryManager", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "get_orchestrator", mock.Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(module, "get_thresholds", mock.Mock(return_value=SimpleNamespace(
        roi_drop=1.0, error_increase=2.0, test_failure_increase=3.0
    )))
    monkeypatch.setattr(module, "persist_sc_thresholds", mock.Mock())
    monkeypatch.setattr(module, "self_coding_managed", lambda **_: (lambda cls: cls))

    manager_instance = SimpleNamespace(pipeline=pipeline_placeholder)
    monkeypatch.setattr(
        module,
        "internalize_coding_bot",
        mock.Mock(return_value=manager_instance),
    )
    monkeypatch.setattr(
        module,
        "ThresholdService",
        mock.Mock(return_value=SimpleNamespace()),
    )
    monkeypatch.setattr(module, "_resolve_pipeline_cls", mock.Mock(return_value=type("_Pipeline", (), {})))

    state = module._ensure_runtime_dependencies(
        pipeline_override=pipeline_placeholder,
        manager_override=sentinel_placeholder,
        promote_pipeline=_promote,
    )

    assert state.pipeline is pipeline_placeholder
    assert getattr(state.manager, "bootstrap_placeholder", False)
    assert promotions == [state.manager]
    module.internalize_coding_bot.assert_not_called()
    cbi.prepare_pipeline_for_bootstrap.assert_not_called()


def test_service_supervisor_reuses_dependency_broker_pipeline(monkeypatch, tmp_path, caplog):
    _reset_cbi_state()
    caplog.set_level("INFO")
    sentinel_placeholder = SimpleNamespace(bootstrap_placeholder=True)
    pipeline_placeholder = SimpleNamespace(
        manager=sentinel_placeholder,
        initial_manager=sentinel_placeholder,
        bootstrap_placeholder=True,
    )
    cbi._mark_bootstrap_placeholder(sentinel_placeholder)
    cbi._mark_bootstrap_placeholder(pipeline_placeholder)
    broker = cbi._bootstrap_dependency_broker()
    broker.advertise(pipeline=pipeline_placeholder, sentinel=sentinel_placeholder)
    cbi._BOOTSTRAP_STATE.depth = 1  # type: ignore[attr-defined]
    cbi._BOOTSTRAP_STATE.owner_depths = {object(): 1}  # type: ignore[attr-defined]

    code_db_module = ModuleType("code_database")
    code_db_module.CodeDB = type("_CodeDB", (), {})
    code_db_module.PatchRecord = type("_PatchRecord", (), {})
    monkeypatch.setitem(sys.modules, "code_database", code_db_module)

    master_module = ModuleType("menace_sandbox.menace_master")
    master_module._init_unused_bots = lambda: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.menace_master", master_module)

    sys.modules.pop("menace_sandbox.service_supervisor", None)
    sys.modules.pop("service_supervisor", None)

    supervisor_module = ModuleType("menace_sandbox.service_supervisor")
    supervisor_module.prepare_pipeline_for_bootstrap = mock.Mock()

    class _ServiceSupervisor:
        def __init__(
            self,
            *,
            context_builder: object,
            dependency_broker: object | None = None,
            pipeline: object | None = None,
            pipeline_promoter: Callable[[object], None] | None = None,
            **_: object,
        ) -> None:
            self.context_builder = context_builder
            self._bootstrap_dependency_broker = (
                dependency_broker if dependency_broker is not None else cbi._bootstrap_dependency_broker()
            )
            broker_pipeline, _sentinel = self._bootstrap_dependency_broker.resolve()
            self.pipeline = pipeline or broker_pipeline
            if pipeline_promoter is None:
                self._pipeline_promoter = lambda manager: setattr(self, "promoted", manager)
            else:
                self._pipeline_promoter = pipeline_promoter
            if self.pipeline is None:
                raise RuntimeError("ServiceSupervisor requires a pipeline")

        def _resolve_bootstrap_handles(self) -> tuple[object | None, Callable[[object], None] | None]:
            return self.pipeline, self._pipeline_promoter

    supervisor_module.ServiceSupervisor = _ServiceSupervisor
    monkeypatch.setitem(sys.modules, "menace_sandbox.service_supervisor", supervisor_module)

    import menace_sandbox.service_supervisor as ss

    class _ContextBuilder:
        def refresh_db_weights(self) -> None:
            return None

    supervisor = ss.ServiceSupervisor(
        context_builder=_ContextBuilder(),
        log_path=str(tmp_path / "supervisor.log"),
        restart_log=str(tmp_path / "restart.log"),
        dependency_broker=broker,
        pipeline=pipeline_placeholder,
    )

    assert supervisor._pipeline_promoter is not None
    assert supervisor._resolve_bootstrap_handles()[0] is pipeline_placeholder
    ss.prepare_pipeline_for_bootstrap.assert_not_called()
    assert not any(
        "prepare_pipeline_for_bootstrap" in record.getMessage()
        for record in caplog.records
    )
