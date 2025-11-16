import types
import pytest
import importlib
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent))
scm = importlib.import_module("test_self_coding_manager_thresholds").scm


class DataBotStub(scm.DataBot):
    def __init__(self, roi: float, errors: float, failures: float, forecasts: dict) -> None:
        self._roi = roi
        self._errors = errors
        self._failures = failures
        self._forecasts = forecasts
        self.updated: dict | None = None
        self._thresholds = types.SimpleNamespace(
            roi_drop=-0.5, error_threshold=2.0, test_failure_threshold=1.0
        )

    def roi(self, bot: str) -> float:
        return self._roi

    def average_errors(self, bot: str) -> float:
        return self._errors

    def average_test_failures(self, bot: str) -> float:
        return self._failures

    def get_thresholds(self, bot: str):
        return self._thresholds

    def reload_thresholds(self, bot: str):
        return self._thresholds

    def forecast_metrics(self, bot: str, *, roi: float, errors: float, tests_failed: float):
        return self._forecasts

    def update_thresholds(
        self,
        bot: str,
        *,
        roi_drop,
        error_threshold,
        test_failure_threshold,
        forecast,
    ):
        self.updated = {
            "roi_drop": roi_drop,
            "error_threshold": error_threshold,
            "test_failure_threshold": test_failure_threshold,
            "forecast": forecast,
        }

    def check_degradation(self, bot: str, roi: float, errors: float, failures: float) -> bool:
        t = self.updated
        assert t is not None
        return (
            roi - t["forecast"]["roi"] <= t["roi_drop"]
            or errors - t["forecast"]["errors"] >= t["error_threshold"]
            or failures - t["forecast"]["tests_failed"] >= t["test_failure_threshold"]
        )


class Engine:
    def __init__(self):
        builder = types.SimpleNamespace(session_id="", refresh_db_weights=lambda: None)
        self.cognition_layer = types.SimpleNamespace(context_builder=builder)
        self.patch_suggestion_db = None


def make_manager(data_bot: DataBotStub) -> scm.SelfCodingManager:
    return scm.SelfCodingManager(
        Engine(),
        scm.ModelAutomationPipeline(),
        bot_name="alpha",
        data_bot=data_bot,
        bot_registry=scm.BotRegistry(),
        quick_fix=types.SimpleNamespace(context_builder=None),
    )


def test_forecast_triggers_refactor():
    forecasts = {
        "roi": (0.4, 0.3, 0.5),
        "errors": (0.0, 0.0, 0.1),
        "tests_failed": (0.0, 0.0, 0.1),
    }
    data_bot = DataBotStub(roi=-1.0, errors=0.0, failures=0.0, forecasts=forecasts)
    mgr = make_manager(data_bot)
    assert mgr.should_refactor()
    assert data_bot.updated["forecast"] == {
        "roi": 0.4,
        "errors": 0.0,
        "tests_failed": 0.0,
    }
    assert pytest.approx(data_bot.updated["roi_drop"], rel=1e-6) == -0.5 + (0.3 - 0.4)
    assert pytest.approx(data_bot.updated["error_threshold"], rel=1e-6) == 2.0 + (0.1 - 0.0)
    assert pytest.approx(data_bot.updated["test_failure_threshold"], rel=1e-6) == 1.0 + (0.1 - 0.0)


def test_forecast_prevents_refactor():
    forecasts = {
        "roi": (1.5, 1.4, 1.6),
        "errors": (0.0, 0.0, 0.1),
        "tests_failed": (0.0, 0.0, 0.1),
    }
    data_bot = DataBotStub(roi=1.0, errors=0.5, failures=0.0, forecasts=forecasts)
    mgr = make_manager(data_bot)
    assert not mgr.should_refactor()
    assert data_bot.updated["forecast"] == {
        "roi": 1.5,
        "errors": 0.0,
        "tests_failed": 0.0,
    }
    assert pytest.approx(data_bot.updated["roi_drop"], rel=1e-6) == -0.5 + (1.4 - 1.5)
    assert pytest.approx(data_bot.updated["error_threshold"], rel=1e-6) == 2.0 + (0.1 - 0.0)
    assert pytest.approx(data_bot.updated["test_failure_threshold"], rel=1e-6) == 1.0 + (0.1 - 0.0)
