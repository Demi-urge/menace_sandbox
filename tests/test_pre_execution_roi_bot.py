import types
import sys
import importlib.util
from pathlib import Path
import importlib.machinery
import pytest
from shared.cooperative_init import cooperative_init_call

ROOT = Path(__file__).resolve().parents[1]


def load_mod():
    if "menace" not in sys.modules:
        pkg = types.ModuleType("menace")
        pkg.__path__ = [str(ROOT)]
        pkg.__spec__ = importlib.machinery.ModuleSpec("menace", loader=None, is_package=True)
        sys.modules["menace"] = pkg
    # Provide lightweight stubs for required submodules
    if "menace.data_bot" not in sys.modules:
        data_mod = types.ModuleType("menace.data_bot")
        class DataBot:
            def __init__(self, *a, **k):
                self.db = types.SimpleNamespace(fetch=lambda n: [])
            def complexity_score(self, df):
                return 0.0
        data_mod.DataBot = DataBot
        sys.modules["menace.data_bot"] = data_mod
    if "menace.task_handoff_bot" not in sys.modules:
        thb = types.ModuleType("menace.task_handoff_bot")
        class TaskInfo: ...
        class TaskPackage: ...
        class TaskHandoffBot:
            def __init__(self, *a, **k):
                pass
            def compile(self, infos):
                return TaskPackage()
            def store_plan(self, *a, **k):
                pass
            def send_package(self, *a, **k):
                pass
        class WorkflowDB:
            def __init__(self, *a, **k):
                pass
            def fetch(self):
                return []
        thb.TaskInfo = TaskInfo
        thb.TaskPackage = TaskPackage
        thb.TaskHandoffBot = TaskHandoffBot
        thb.WorkflowDB = WorkflowDB
        sys.modules["menace.task_handoff_bot"] = thb
    if "menace.implementation_optimiser_bot" not in sys.modules:
        mod = types.ModuleType("menace.implementation_optimiser_bot")
        class ImplementationOptimiserBot:
            pass
        mod.ImplementationOptimiserBot = ImplementationOptimiserBot
        sys.modules["menace.implementation_optimiser_bot"] = mod
    if "menace.chatgpt_enhancement_bot" not in sys.modules:
        mod = types.ModuleType("menace.chatgpt_enhancement_bot")
        class EnhancementDB:
            def __init__(self, *a, **k):
                pass
            def fetch(self):
                return []
        mod.EnhancementDB = EnhancementDB
        sys.modules["menace.chatgpt_enhancement_bot"] = mod
    if "menace.database_manager" not in sys.modules:
        mod = types.ModuleType("menace.database_manager")
        mod.DB_PATH = "db"
        mod.update_model = lambda *a, **k: None
        mod.init_db = lambda conn: None
        sys.modules["menace.database_manager"] = mod
    if "menace.prediction_manager_bot" not in sys.modules:
        mod = types.ModuleType("menace.prediction_manager_bot")
        class PredictionManager:
            def __init__(self, *a, **k):
                self.registry = {}
            def assign_prediction_bots(self, _):
                return []
        mod.PredictionManager = PredictionManager
        sys.modules["menace.prediction_manager_bot"] = mod
    if "menace.unified_event_bus" not in sys.modules:
        mod = types.ModuleType("menace.unified_event_bus")
        class UnifiedEventBus:
            pass
        mod.UnifiedEventBus = UnifiedEventBus
        sys.modules["menace.unified_event_bus"] = mod
    path = ROOT / "pre_execution_roi_bot.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace.pre_execution_roi_bot", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["menace.pre_execution_roi_bot"] = mod
    spec.loader.exec_module(mod)
    return mod


prb = load_mod()


def _tasks():
    return [
        prb.BuildTask(
            name="a",
            complexity=1.0,
            frequency=1.0,
            expected_income=10.0,
            resources={"compute": 1.0, "storage": 1.0},
        ),
        prb.BuildTask(
            name="b",
            complexity=2.0,
            frequency=2.0,
            expected_income=20.0,
            resources={"compute": 2.0, "storage": 2.0},
        ),
    ]


def test_estimate_cost(tmp_path):
    db = prb.ROIHistoryDB(tmp_path / "hist.csv")
    db.add(1.0, 1.0, 0.1, 0.5, 5.0, 1.0)
    bot = prb.PreExecutionROIBot(db)
    cost = bot.estimate_cost(_tasks())
    assert cost > 0


def test_forecast_roi(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    result = bot.forecast_roi(_tasks(), projected_income=30.0, discount_rate=0.1)
    assert abs(result.roi - (result.income - result.cost)) < 1e-5
    assert result.roi_pct == pytest.approx((result.roi / result.cost) * 100)
    no_disc = bot.forecast_roi(_tasks(), projected_income=30.0)
    assert result.income < no_disc.income


def test_run_scenario(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    base = bot.forecast_roi(_tasks(), 30.0)

    def remove_a(tasks):
        return [t for t in tasks if t.name != "a"]

    scenario = bot.run_scenario(_tasks(), remove_a)
    assert scenario.cost < base.cost


def test_diminishing_returns(tmp_path):
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"))
    res = bot.forecast_roi_diminishing(_tasks(), 1000.0)
    assert res.income <= bot.forecast_roi(_tasks(), 1000.0).income


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, feats):
        self.called = True
        return feats[0] + 1.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _):
        return ["p"]


def test_predict_model_roi_with_prediction(tmp_path):
    manager = StubManager(DummyPred())
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"), prediction_manager=manager)
    res = bot.predict_model_roi("m", _tasks())
    assert manager.registry["p"].bot.called
    assert res.roi != bot.forecast_roi(_tasks(), projected_income=30.0).roi


class DummyMetricPred:
    def __init__(self):
        self.args = None

    def predict_metric(self, name, feats):
        self.args = (name, feats)
        return 42.0


def test_predict_metric_helper(tmp_path):
    bot_obj = DummyMetricPred()
    manager = StubManager(bot_obj)
    bot = prb.PreExecutionROIBot(prb.ROIHistoryDB(tmp_path / "hist.csv"), prediction_manager=manager)
    val = bot.predict_metric("lucrativity", [1.0, 2.0])
    assert val == 42.0
    assert bot_obj.args[0] == "lucrativity"


def test_scrape_bonus_no_requests(monkeypatch):
    monkeypatch.setattr(prb, "requests", None)
    bot = prb.PreExecutionROIBot()
    assert bot._scrape_bonus() == 0.0


def test_scrape_bonus_with_sentiment(monkeypatch):
    class DummyBlob:
        def __init__(self, text):
            self.sentiment = types.SimpleNamespace(polarity=0.5)

    tb = types.ModuleType("textblob")
    tb.TextBlob = DummyBlob
    monkeypatch.setitem(sys.modules, "textblob", tb)

    monkeypatch.setattr(prb, "requests", types.SimpleNamespace(get=lambda *a, **k: None))

    import asyncio
    monkeypatch.setattr(asyncio, "run", lambda coro: ["good news"] * 3)

    bot = prb.PreExecutionROIBot()
    val = bot._scrape_bonus()
    assert val > 0


def test_cooperative_init_accepts_manager_keyword():
    manager = object()
    bot = prb.PreExecutionROIBot.__new__(prb.PreExecutionROIBot)

    state = cooperative_init_call(
        prb.PreExecutionROIBot.__init__,
        bot,
        manager=manager,
    )

    assert state["dropped"] == ()
    assert bot.manager is manager
