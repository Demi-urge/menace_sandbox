import menace.cross_model_scheduler as cms

class DummyComparator:
    def __init__(self) -> None:
        self.calls = 0

    def rank_and_deploy(self) -> None:
        self.calls += 1


def test_scheduler_adds_job(monkeypatch):
    cmp = DummyComparator()
    service = cms.ModelRankingService(comparator=cmp)
    monkeypatch.setattr(cms, "BackgroundScheduler", None)
    recorded = {}

    def fake_add_job(self, func, interval, id):
        recorded["func"] = func
        recorded["interval"] = interval
        recorded["id"] = id

    monkeypatch.setattr(cms._SimpleScheduler, "add_job", fake_add_job)
    service.run_continuous(interval=123)
    assert recorded["interval"] == 123
    assert recorded["id"] == "model_ranking"
    recorded["func"]()
    assert cmp.calls == 1
