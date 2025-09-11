import sys
import types
from pathlib import Path
import logging

import pytest
from unittest.mock import MagicMock

# create a lightweight package stub for menace_sandbox
package = types.ModuleType("menace_sandbox")
package.__path__ = [str(Path(__file__).resolve().parent.parent)]
sys.modules.setdefault("menace_sandbox", package)

# stub out heavy dependencies before importing EvolutionOrchestrator

data_bot = types.ModuleType("menace_sandbox.data_bot")


class DataBot:  # minimal stub
    pass


data_bot.DataBot = DataBot
sys.modules["menace_sandbox.data_bot"] = data_bot

capital_mod = types.ModuleType("menace_sandbox.capital_management_bot")


class CapitalManagementBot:  # minimal stub
    trend_predictor = None


capital_mod.CapitalManagementBot = CapitalManagementBot
sys.modules["menace_sandbox.capital_management_bot"] = capital_mod

system_mod = types.ModuleType("menace_sandbox.system_evolution_manager")


class SystemEvolutionManager:  # minimal stub
    pass


system_mod.SystemEvolutionManager = SystemEvolutionManager
sys.modules["menace_sandbox.system_evolution_manager"] = system_mod

history_mod = types.ModuleType("menace_sandbox.evolution_history_db")


class EvolutionHistoryDB:  # minimal stub
    pass


class EvolutionEvent:  # minimal stub
    pass


history_mod.EvolutionHistoryDB = EvolutionHistoryDB
history_mod.EvolutionEvent = EvolutionEvent
sys.modules["menace_sandbox.evolution_history_db"] = history_mod

_eval_mod = types.ModuleType("menace_sandbox.evaluation_history_db")


class EvaluationHistoryDB:  # minimal stub
    pass


_eval_mod.EvaluationHistoryDB = EvaluationHistoryDB
sys.modules["menace_sandbox.evaluation_history_db"] = _eval_mod

trend_mod = types.ModuleType("menace_sandbox.trend_predictor")


class TrendPredictor:  # minimal stub
    pass


trend_mod.TrendPredictor = TrendPredictor
sys.modules["menace_sandbox.trend_predictor"] = trend_mod

threshold_mod = types.ModuleType("menace_sandbox.self_coding_thresholds")


class _T:
    error_increase = 0.1
    roi_drop = -0.1


def get_thresholds(_bot=None):
    return _T()


threshold_mod.get_thresholds = get_thresholds
sys.modules["menace_sandbox.self_coding_thresholds"] = threshold_mod

scm_mod = types.ModuleType("menace_sandbox.self_coding_manager")


class HelperGenerationError(Exception):
    pass


scm_mod.HelperGenerationError = HelperGenerationError
sys.modules["menace_sandbox.self_coding_manager"] = scm_mod

ml_mod = types.ModuleType("menace_sandbox.mutation_logger")
sys.modules["menace_sandbox.mutation_logger"] = ml_mod

vec_mod = types.ModuleType("vector_service.context_builder")


class ContextBuilder:  # minimal stub
    pass


vec_mod.ContextBuilder = ContextBuilder
sys.modules["vector_service.context_builder"] = vec_mod

from menace_sandbox.evolution_orchestrator import EvolutionOrchestrator  # noqa: E402


class DummyDB:
    def fetch(self, limit=50):
        return []


class DummyDataBot:
    def __init__(self):
        self.db = DummyDB()


class DummyImprovementEngine:
    bot_name = "engine"


class DummyEvolutionManager:
    pass


def test_unwritable_dataset_path(tmp_path):
    bad_path = tmp_path / "missing_dir" / "dataset.csv"
    with pytest.raises(FileNotFoundError):
        EvolutionOrchestrator(
            data_bot=DummyDataBot(),
            capital_bot=CapitalManagementBot(),
            improvement_engine=DummyImprovementEngine(),
            evolution_manager=DummyEvolutionManager(),
            history_db=MagicMock(),
            dataset_path=bad_path,
        )


def test_latest_eval_score_error_fallback(monkeypatch, caplog):
    eng = DummyImprovementEngine()
    orchestrator = EvolutionOrchestrator(
        data_bot=DummyDataBot(),
        capital_bot=CapitalManagementBot(),
        improvement_engine=eng,
        evolution_manager=DummyEvolutionManager(),
        history_db=MagicMock(),
    )
    orchestrator._cached_eval_score = 1.23

    class BadEvalDB:
        def history(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "menace_sandbox.evolution_orchestrator.EvaluationHistoryDB",
        lambda: BadEvalDB(),
    )

    with caplog.at_level(logging.ERROR):
        score = orchestrator._latest_eval_score()

    assert score == 1.23
    assert any("failed to fetch latest eval score" in r.getMessage() for r in caplog.records)
