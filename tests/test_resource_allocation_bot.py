import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.resource_allocation_bot as rab
import menace.resource_prediction_bot as rpb


class _DummyBuilder:
    def build(self, *_: object, **__: object) -> str:
        return "ctx"

    def refresh_db_weights(self):
        pass


def test_allocate_and_history(tmp_path):
    db = rab.AllocationDB(tmp_path / "a.db")
    bot = rab.ResourceAllocationBot(
        db, rpb.TemplateDB(tmp_path / "t.csv"), context_builder=_DummyBuilder()
    )
    metrics = {
        "x": rpb.ResourceMetrics(cpu=1.0, memory=50.0, disk=1.0, time=1.0),
        "y": rpb.ResourceMetrics(cpu=10.0, memory=90.0, disk=5.0, time=2.0),
    }
    actions = bot.allocate(metrics)
    hist = db.history()
    assert len(hist) == 2
    assert actions[0][1] is True
    assert actions[1][1] is False


def test_refresh_db_weights_failure(tmp_path):
    class BadBuilder(_DummyBuilder):
        def refresh_db_weights(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        rab.ResourceAllocationBot(rab.AllocationDB(tmp_path / "a.db"), context_builder=BadBuilder())


def test_genetic_step():
    bot = rab.ResourceAllocationBot(rab.AllocationDB(":memory:"), context_builder=_DummyBuilder())
    strategies = [{"name": "a", "roi": 0.1}, {"name": "b", "roi": 0.5}]
    best = bot.genetic_step(strategies)
    assert best["name"] == "b"


class DummyPred:
    def __init__(self):
        self.called = False

    def predict(self, vec):
        self.called = True
        return 1.0


class StubManager:
    def __init__(self, bot):
        self.registry = {"p": type("E", (), {"bot": bot})()}

    def assign_prediction_bots(self, _):
        return ["p"]


def test_allocate_with_prediction(tmp_path):
    manager = StubManager(DummyPred())
    db = rab.AllocationDB(tmp_path / "a.db")
    bot = rab.ResourceAllocationBot(
        db,
        rpb.TemplateDB(tmp_path / "t.csv"),
        prediction_manager=manager,
        context_builder=_DummyBuilder(),
    )
    metrics = {"x": rpb.ResourceMetrics(cpu=1.0, memory=1.0, disk=1.0, time=1.0)}
    scores = bot.evaluate(metrics)
    assert manager.registry["p"].bot.called
    assert scores["x"] > 1.0
