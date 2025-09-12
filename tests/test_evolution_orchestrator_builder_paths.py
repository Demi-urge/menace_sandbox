import sys
import types


# Minimal stubs to satisfy imports in evolution_orchestrator
data_bot = types.ModuleType("menace_sandbox.data_bot")


class DataBot:  # pragma: no cover - simple stub
    pass


data_bot.DataBot = DataBot
sys.modules["menace_sandbox.data_bot"] = data_bot

capital_mod = types.ModuleType("menace_sandbox.capital_management_bot")


class CapitalManagementBot:  # pragma: no cover - simple stub
    trend_predictor = None


capital_mod.CapitalManagementBot = CapitalManagementBot
sys.modules["menace_sandbox.capital_management_bot"] = capital_mod

system_mod = types.ModuleType("menace_sandbox.system_evolution_manager")


class SystemEvolutionManager:  # pragma: no cover - simple stub
    pass


system_mod.SystemEvolutionManager = SystemEvolutionManager
sys.modules["menace_sandbox.system_evolution_manager"] = system_mod

history_mod = types.ModuleType("menace_sandbox.evolution_history_db")


class EvolutionHistoryDB:  # pragma: no cover - simple stub
    pass


class EvolutionEvent:  # pragma: no cover - simple stub
    pass


history_mod.EvolutionHistoryDB = EvolutionHistoryDB
history_mod.EvolutionEvent = EvolutionEvent
sys.modules["menace_sandbox.evolution_history_db"] = history_mod

eval_mod = types.ModuleType("menace_sandbox.evaluation_history_db")


class EvaluationHistoryDB:  # pragma: no cover - simple stub
    pass


eval_mod.EvaluationHistoryDB = EvaluationHistoryDB
sys.modules["menace_sandbox.evaluation_history_db"] = eval_mod

trend_mod = types.ModuleType("menace_sandbox.trend_predictor")


class TrendPredictor:  # pragma: no cover - simple stub
    pass


trend_mod.TrendPredictor = TrendPredictor
sys.modules["menace_sandbox.trend_predictor"] = trend_mod

threshold_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")


class _T:
    error_increase = 0.1
    roi_drop = -0.1


def get_thresholds(_bot=None):  # pragma: no cover - simple stub
    return _T()


threshold_mod.get_thresholds = get_thresholds
sys.modules["menace_sandbox.self_coding_thresholds"] = threshold_mod

scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")


class HelperGenerationError(Exception):  # pragma: no cover - simple stub
    pass


scm_mod.HelperGenerationError = HelperGenerationError
sys.modules["menace_sandbox.self_coding_manager"] = scm_mod

ml_mod = types.ModuleType("menace_sandbox.mutation_logger")
sys.modules["menace_sandbox.mutation_logger"] = ml_mod

vec_mod = types.ModuleType("vector_service.context_builder")


class ContextBuilder:  # pragma: no cover - simple stub
    pass


vec_mod.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = vec_mod

import menace_sandbox.evolution_orchestrator as eo


def test_builder_uses_sandbox_settings(tmp_path, monkeypatch):
    for var in [
        "BOT_DB_PATH",
        "CODE_DB_PATH",
        "ERROR_DB_PATH",
        "WORKFLOW_DB_PATH",
    ]:
        monkeypatch.delenv(var, raising=False)

    recorded: dict[str, tuple] = {}

    class RecordingBuilder:
        def __init__(self, *paths):
            recorded["paths"] = paths

    monkeypatch.setattr(eo, "ContextBuilder", RecordingBuilder)

    data_bot = types.SimpleNamespace(
        db=types.SimpleNamespace(fetch=lambda limit=50: []),
        subscribe_degradation=lambda cb: None,
        settings=types.SimpleNamespace(sandbox_data_dir=str(tmp_path)),
    )

    manager = types.SimpleNamespace(
        bot_name="dummy",
        event_bus=None,
        should_refactor=lambda: True,
        register_patch_cycle=lambda *a, **k: None,
        generate_and_patch=lambda *a, **k: None,
    )

    orch = eo.EvolutionOrchestrator(
        data_bot,
        types.SimpleNamespace(energy_score=lambda **k: 1.0),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
    )

    mod_path = tmp_path / "dummy.py"
    mod_path.write_text("pass\n")
    mod = types.ModuleType("dummy")
    mod.__file__ = str(mod_path)
    sys.modules["dummy"] = mod

    orch._on_bot_degraded({"bot": "dummy"})

    assert recorded["paths"] == (
        str(tmp_path / "bots.db"),
        str(tmp_path / "code.db"),
        str(tmp_path / "errors.db"),
        str(tmp_path / "workflows.db"),
    )

