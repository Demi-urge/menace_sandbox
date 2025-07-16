import sys
import subprocess
import types
import pytest

jinja2 = types.ModuleType("jinja2")
jinja2.Template = object
sys.modules.setdefault("jinja2", jinja2)

yaml = types.ModuleType("yaml")
sys.modules.setdefault("yaml", yaml)

watchdog_stub = types.ModuleType("menace.watchdog")
class DummyWatchdog:
    def record_fault(self, *args, **kwargs):
        pass

watchdog_stub.Watchdog = DummyWatchdog
sys.modules.setdefault("menace.watchdog", watchdog_stub)

import menace.chaos_scheduler as sched_mod
from menace.chaos_scheduler import ChaosScheduler
from menace.chaos_tester import ChaosTester


def _stop_after_first(sched: ChaosScheduler):
    def inner(_: float) -> None:
        sched.running = False
        raise SystemExit

    return inner


def test_scheduler_invokes_disk_and_network(monkeypatch, tmp_path):
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1)"])
    disk = tmp_path / "d"
    disk.write_text("x")
    called = {"disk": 0, "net": 0}

    monkeypatch.setattr(
        ChaosTester, "chaos_monkey", lambda self, processes=None, threads=None: None
    )

    def fake_corrupt(path: str) -> None:
        called["disk"] += 1

    def fake_partition(hosts):
        called["net"] += 1
        return hosts[:1]

    monkeypatch.setattr(ChaosTester, "corrupt_disk", staticmethod(fake_corrupt))
    monkeypatch.setattr(
        ChaosTester, "partition_network", staticmethod(fake_partition)
    )
    sched = ChaosScheduler(
        processes=[proc], disk_paths=[str(disk)], hosts=["a", "b"], interval=0
    )
    monkeypatch.setattr(sched_mod.time, "sleep", _stop_after_first(sched))

    sched.running = True
    with pytest.raises(SystemExit):
        sched._loop()

    proc.terminate()
    assert called["disk"] == 1
    assert called["net"] == 1
