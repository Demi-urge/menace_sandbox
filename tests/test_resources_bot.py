import menace.resource_prediction_bot as rpb
import menace.resource_allocation_bot as rab
import menace.resources_bot as resb
from menace.vector_service.context_builder import ContextBuilder


class _DummyBuilder(ContextBuilder):
    def __init__(self):
        pass

    def build(self, *_: object, **__: object) -> str:  # type: ignore[override]
        return "ctx"

    def refresh_db_weights(self):  # type: ignore[override]
        pass

def test_redistribute_records(tmp_path):
    db = resb.ROIHistoryDB(tmp_path / "roi.db")
    alloc_db = rab.AllocationDB(tmp_path / "a.db")
    builder = _DummyBuilder()
    alloc_bot = rab.ResourceAllocationBot(
        alloc_db, rpb.TemplateDB(tmp_path / "t.csv"), context_builder=builder
    )
    bot = resb.ResourcesBot(db, alloc_bot, context_builder=builder)
    metrics = {
        "b1": rpb.ResourceMetrics(cpu=1.0, memory=50.0, disk=1.0, time=1.0),
        "b2": rpb.ResourceMetrics(cpu=5.0, memory=100.0, disk=2.0, time=2.0),
    }
    actions = bot.redistribute(metrics, {"market": 1.0})
    hist = db.history()
    assert len(hist) == 2
    assert actions[0][1] is True


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, vec):
        self.called = True
        return 0.5


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _bot):
        return ["p"]


class DummyStrategy:
    def __init__(self):
        self.calls = []

    def receive_resource_usage(self, metrics):
        self.calls.append(metrics)


def test_strategy_and_persistence(tmp_path):
    manager = StubManager(DummyPred())
    strategy = DummyStrategy()
    db = resb.ROIHistoryDB(tmp_path / "r.db")
    alloc_db = rab.AllocationDB(tmp_path / "a.db")
    builder = _DummyBuilder()
    alloc_bot = rab.ResourceAllocationBot(
        alloc_db, rpb.TemplateDB(tmp_path / "t.csv"), context_builder=builder
    )
    bot = resb.ResourcesBot(
        db,
        alloc_bot,
        context_builder=builder,
        prediction_manager=manager,
        strategy_bot=strategy,
    )
    metrics = {"b": rpb.ResourceMetrics(cpu=1.0, memory=1.0, disk=1.0, time=1.0)}
    bot.redistribute(metrics)
    hist = bot.current_allocations()
    assert len(hist) == 1
    assert strategy.calls and "b" in strategy.calls[0]
    assert bot.assigned_prediction_bots == ["p"]
