import types
import pytest
import importlib

scm = importlib.import_module("unit_tests.test_self_coding_manager_thresholds").scm


class TrendStub:
    def __init__(self, roi: float, errors: float) -> None:
        self.roi = roi
        self.errors = errors


class PredictorStub:
    def __init__(self, roi: float, errors: float) -> None:
        self.roi = roi
        self.errors = errors
        self.trained = False

    def train(self) -> None:
        self.trained = True

    def predict_future_metrics(self, cycles: int = 1) -> TrendStub:
        return TrendStub(self.roi, self.errors)


class DataBotStub(scm.DataBot):
    def __init__(self, roi: float, errors: float, failures: float, predictor: PredictorStub) -> None:
        self._roi = roi
        self._errors = errors
        self._failures = failures
        self.trend_predictor = predictor
        self.updated = None
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

    def update_thresholds(self, bot: str, *, roi_drop, error_threshold, test_failure_threshold, forecast):
        self.updated = {
            "roi_drop": roi_drop,
            "error_threshold": error_threshold,
            "test_failure_threshold": test_failure_threshold,
            "forecast": forecast,
        }

    def check_degradation(self, bot: str, roi: float, errors: float, failures: float) -> bool:
        return False


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


def test_downward_trend_adjusts_thresholds():
    predictor = PredictorStub(roi=0.4, errors=2.0)
    data_bot = DataBotStub(roi=1.0, errors=0.0, failures=0.0, predictor=predictor)
    mgr = make_manager(data_bot)
    mgr.should_refactor()
    assert predictor.trained
    assert data_bot.updated["forecast"] == {
        "roi": 0.4,
        "errors": 2.0,
        "tests_failed": 0.0,
    }
    assert pytest.approx(data_bot.updated["roi_drop"], rel=1e-6) == -0.5 + (1.0 - 0.4)
    assert pytest.approx(data_bot.updated["error_threshold"], rel=1e-6) == 2.0 + (0.0 - 2.0)


def test_upward_trend_adjusts_thresholds():
    predictor = PredictorStub(roi=1.5, errors=0.1)
    data_bot = DataBotStub(roi=1.0, errors=1.0, failures=0.0, predictor=predictor)
    mgr = make_manager(data_bot)
    mgr.should_refactor()
    assert predictor.trained
    assert data_bot.updated["forecast"] == {
        "roi": 1.5,
        "errors": 0.1,
        "tests_failed": 0.0,
    }
    assert pytest.approx(data_bot.updated["roi_drop"], rel=1e-6) == -0.5 + (1.0 - 1.5)
    assert pytest.approx(data_bot.updated["error_threshold"], rel=1e-6) == 2.0 + (1.0 - 0.1)
