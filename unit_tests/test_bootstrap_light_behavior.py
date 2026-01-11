import importlib
import sys
import types

import pytest

import menace_sandbox.system_evolution_manager as system_evolution_manager
import menace_sandbox.trend_predictor as trend_predictor


def _install_shared_orchestrator_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    module_name = "menace_sandbox.shared_evolution_orchestrator"
    stub = types.ModuleType(module_name)

    def _get_orchestrator(*_args: object, **_kwargs: object) -> None:
        return None

    stub.get_orchestrator = _get_orchestrator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module_name, stub)


def _import_structural_evolution_bot(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_shared_orchestrator_stub(monkeypatch)
    module_name = "menace_sandbox.structural_evolution_bot"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _reset_structural_state(structural_evolution_bot) -> None:
    structural_evolution_bot._data_bot = None
    structural_evolution_bot._engine = None


def test_structural_evolution_bootstrap_light_defers_bots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENACE_BOOTSTRAP_LIGHT", "1")
    structural_evolution_bot = _import_structural_evolution_bot(monkeypatch)
    _reset_structural_state(structural_evolution_bot)

    def _fail_init(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Heavy constructor invoked during bootstrap light")

    monkeypatch.setattr(structural_evolution_bot.DataBot, "__init__", _fail_init, raising=True)
    monkeypatch.setattr(structural_evolution_bot.SelfCodingEngine, "__init__", _fail_init, raising=True)

    data_bot = structural_evolution_bot._get_data_bot()
    engine = structural_evolution_bot._get_engine()

    assert isinstance(data_bot, structural_evolution_bot._BootstrapDataBotStub)
    assert isinstance(engine, structural_evolution_bot._BootstrapEngineStub)
    _reset_structural_state(structural_evolution_bot)


def test_trend_predictor_bootstrap_light_defers_db_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MENACE_BOOTSTRAP_LIGHT", "1")

    def _fail_init(*_args: object, **_kwargs: object) -> None:
        pytest.fail("DB initialized during bootstrap light")

    monkeypatch.setattr(trend_predictor, "EvolutionHistoryDB", type("History", (), {"__init__": _fail_init}))
    monkeypatch.setattr(trend_predictor, "MetricsDB", type("Metrics", (), {"__init__": _fail_init}))

    predictor = trend_predictor.TrendPredictor()

    assert predictor._lazy_history_db is True
    assert predictor._lazy_metrics_db is True


def test_trend_predictor_normal_runtime_eager_init(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MENACE_BOOTSTRAP_LIGHT", raising=False)
    calls: dict[str, int] = {"history": 0, "metrics": 0}

    class HistoryStub:
        def __init__(self) -> None:
            calls["history"] += 1

    class MetricsStub:
        def __init__(self) -> None:
            calls["metrics"] += 1

    monkeypatch.setattr(trend_predictor, "EvolutionHistoryDB", HistoryStub)
    monkeypatch.setattr(trend_predictor, "MetricsDB", MetricsStub)

    trend_predictor.TrendPredictor()

    assert calls == {"history": 1, "metrics": 1}


def test_system_evolution_manager_defers_struct_bot(monkeypatch: pytest.MonkeyPatch) -> None:
    structural_evolution_bot = _import_structural_evolution_bot(monkeypatch)

    def _fail_init(*_args: object, **_kwargs: object) -> None:
        pytest.fail("StructuralEvolutionBot initialized during manager init")

    monkeypatch.setattr(
        structural_evolution_bot.StructuralEvolutionBot, "__init__", _fail_init, raising=True
    )

    metrics_stub = type("MetricsStub", (), {"log_eval": lambda *args, **kwargs: None})()
    radar_stub = type("RadarStub", (), {"start": lambda self: None})()

    manager = system_evolution_manager.SystemEvolutionManager(
        bots=["bot-a"],
        metrics_db=metrics_stub,
        radar_service=radar_stub,
    )

    assert manager._struct_bot is None


def test_system_evolution_manager_run_cycle_instantiates_struct_bot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structural_evolution_bot = _import_structural_evolution_bot(monkeypatch)
    created = {"count": 0}

    class StructBotStub:
        def __init__(self) -> None:
            created["count"] += 1

        def take_snapshot(self):
            return object()

        def predict_changes(self, _snap):
            return []

    monkeypatch.setattr(structural_evolution_bot, "StructuralEvolutionBot", StructBotStub)

    class GAStub:
        def run_cycle(self, _bot: str):
            class Rec:
                roi = 0.0

            return Rec()

    metrics_stub = type("MetricsStub", (), {"log_eval": lambda *args, **kwargs: None})()
    radar_stub = type("RadarStub", (), {"start": lambda self: None})()

    manager = system_evolution_manager.SystemEvolutionManager(
        bots=["bot-a"],
        metrics_db=metrics_stub,
        radar_service=radar_stub,
    )
    manager.ga_manager = GAStub()

    result = manager.run_cycle()

    assert created["count"] == 1
    assert result.ga_results == {"bot-a": 0.0}
