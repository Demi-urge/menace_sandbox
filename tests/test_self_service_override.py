import os
import time
import threading

import menace.self_service_override as so
import menace.resource_allocation_optimizer as rao
import menace.data_bot as db


def test_run_continuous_invokes_adjust(tmp_path, monkeypatch):
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    svc = so.SelfServiceOverride(roi, metrics)

    calls = []
    monkeypatch.setattr(svc, "adjust", lambda: calls.append(True))
    stop = threading.Event()
    thread = svc.run_continuous(interval=0.01, stop_event=stop)
    time.sleep(0.03)
    stop.set()
    thread.join(timeout=0.1)
    assert calls


def test_adjust_auto_fix(tmp_path, monkeypatch):
    class DummyDF(list):
        @property
        def iloc(self):
            class ILoc:
                def __init__(self, rows):
                    self._rows = rows

                def __getitem__(self, idx):
                    return self._rows[idx]

            return ILoc(self)

    class FakeROI:
        def history(self, limit=2):
            return DummyDF(
                [
                    {"revenue": 50.0, "api_cost": 0.0},
                    {"revenue": 100.0, "api_cost": 0.0},
                ]
            )

    roi = FakeROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(db.MetricRecord(bot="b", cpu=0.0, memory=0.0, response_time=0.0, disk_io=0.0, net_io=0.0, errors=10))

    calls = {"fix": 0}

    monkeypatch.setattr(so.SecurityAuditor, "audit", lambda self: True)

    def fake_fix(aud):
        calls["fix"] += 1
        return True

    monkeypatch.setattr(so, "fix_until_safe", fake_fix)
    svc = so.SelfServiceOverride(roi, metrics)
    svc.adjust()
    assert calls["fix"] == 1
    assert os.environ.get("MENACE_SAFE") is None
