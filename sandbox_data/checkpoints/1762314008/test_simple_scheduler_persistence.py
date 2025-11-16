import pytest
import menace.cross_model_scheduler as cms

class DummyThread:
    def __init__(self, target=None, daemon=None):
        self.target = target
    def start(self):
        pass
    def join(self, timeout=0):
        pass


def dummy_job():
    pass


def failing_job():
    raise RuntimeError


def test_state_persistence(monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setattr(cms._SimpleScheduler, "STATE_FILE", state)
    monkeypatch.setattr(cms.threading, "Thread", DummyThread)
    sched = cms._SimpleScheduler()
    sched.add_job(dummy_job, interval=10, id="j1", retry_delay=2.0, max_retries=5)
    sched.shutdown()

    sched2 = cms._SimpleScheduler()
    assert sched2.list_jobs() == ["j1"]
    assert sched2._retry_delays["j1"] == 2.0
    assert sched2._max_retries["j1"] == 5


def test_retry_persistence(monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setattr(cms._SimpleScheduler, "STATE_FILE", state)
    monkeypatch.setattr(cms.threading, "Thread", DummyThread)
    monkeypatch.setattr(cms.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(cms.time, "time", lambda: 0.0)

    sched = cms._SimpleScheduler()
    sched.add_job(failing_job, interval=10, id="j")
    sched._next_runs["j"] = 0.0
    with pytest.raises(SystemExit):
        sched._run()
    sched.shutdown()

    sched2 = cms._SimpleScheduler()
    assert sched2._retry_counts["j"] == 1
    assert pytest.approx(sched2._next_runs["j"], 0.0001) == sched._next_runs["j"]


def test_job_misfire(monkeypatch, tmp_path):
    state = tmp_path / "s.json"
    monkeypatch.setattr(cms._SimpleScheduler, "STATE_FILE", state)
    monkeypatch.setattr(cms.threading, "Thread", DummyThread)
    monkeypatch.setattr(cms.time, "sleep", lambda s: (_ for _ in ()).throw(SystemExit))
    monkeypatch.setattr(cms.time, "time", lambda: 100.0)
    called = []
    def job():
        called.append(True)
    sched = cms._SimpleScheduler()
    sched.add_job(job, interval=10, id="j", misfire_grace_time=1.0)
    sched._next_runs["j"] = 50.0
    with pytest.raises(SystemExit):
        sched._run()
    assert not called
    assert sched._next_runs["j"] == 110.0
