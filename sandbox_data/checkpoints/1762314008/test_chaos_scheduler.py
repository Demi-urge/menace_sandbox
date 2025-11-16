import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import sys  # noqa: E402
import subprocess  # noqa: E402
import menace.chaos_scheduler as sched_mod  # noqa: E402
from menace.chaos_scheduler import ChaosScheduler  # noqa: E402
from menace.chaos_tester import ChaosTester  # noqa: E402
import menace.watchdog as wd  # noqa: E402
import menace.error_bot as eb  # noqa: E402
from vector_service.context_builder import ContextBuilder  # noqa: E402
import menace.resource_allocation_optimizer as rao  # noqa: E402
import menace.data_bot as db  # noqa: E402


def _setup_dbs(tmp_path):
    err = eb.ErrorDB(tmp_path / "e.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    return err, roi, metrics


class DummyBot:
    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self.checked = False

    def check(self) -> None:
        self.checked = True


def _stop_after_first(sched: ChaosScheduler):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_scheduler_records_fault(monkeypatch, tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    bot = DummyBot(proc)
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    builder = ContextBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    watch = wd.Watchdog(err_db, roi_db, metrics_db, context_builder=builder)
    sched = ChaosScheduler(processes=[proc], bots=[bot], interval=0, watchdog=watch)

    monkeypatch.setattr(
        ChaosTester, "chaos_monkey", lambda self, processes=None, threads=None: "killed_process"
    )
    monkeypatch.setattr(
        sched_mod.ChaosTester,
        "validate_recovery",
        staticmethod(lambda b, action: (action(), True)[1]),
    )
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))

    sched.running = True
    with pytest.raises(SystemExit):
        sched._loop()

    assert watch.synthetic_faults[0]["recovered"]
    assert bot.checked
