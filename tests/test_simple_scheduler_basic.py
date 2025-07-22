import pytest
import menace.cross_model_scheduler as cms

class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
    def start(self):
        pass
    def join(self, timeout=0):
        pass

def test_job_runs_and_shutdown(monkeypatch):
    monkeypatch.setattr(cms.threading, "Thread", DummyThread)
    monkeypatch.setattr(cms.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    sched = cms._SimpleScheduler()
    called = []
    def job():
        called.append(True)
    sched.add_job(job, interval=1, id="j")
    sched._next_runs["j"] = cms.time.time()
    with pytest.raises(SystemExit):
        sched._run()
    assert called == [True]
    sched.shutdown()


def test_remove_job(monkeypatch):
    monkeypatch.setattr(cms.threading, "Thread", DummyThread)
    sched = cms._SimpleScheduler()
    sched.add_job(lambda: None, interval=10, id="j")
    sched.remove_job("j")
    assert sched.list_jobs() == []
    sched.shutdown()
