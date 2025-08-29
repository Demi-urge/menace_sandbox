# flake8: noqa
import os
import sys
import types
import time
import subprocess

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
pkg = types.ModuleType("menace")
pkg.__path__ = [os.getcwd()]
sys.modules.setdefault("menace", pkg)

import menace.self_service_override as so
import menace.resource_allocation_optimizer as rao
import menace.data_bot as db


class DummyDF(list):
    @property
    def iloc(self):
        class ILoc:
            def __init__(self, rows):
                self._rows = rows

            def __getitem__(self, idx):
                return self._rows[idx]

        return ILoc(self)


class FlatROI:
    def history(self, limit=2):
        return DummyDF(
            [{"revenue": 100.0, "api_cost": 0.0}, {"revenue": 100.0, "api_cost": 0.0}]
        )


class DroppingROI:
    def history(self, limit=2):
        return DummyDF(
            [{"revenue": 50.0, "api_cost": 0.0}, {"revenue": 100.0, "api_cost": 0.0}]
        )


def test_run_continuous_invokes_adjust_and_stop(tmp_path, monkeypatch):
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    svc = so.SelfServiceOverride(roi, metrics)

    calls = []
    monkeypatch.setattr(svc, "adjust", lambda: calls.append(True))
    thread = svc.run_continuous(interval=0.01)
    time.sleep(0.03)
    svc.stop()
    thread.join(timeout=0.1)
    assert calls


def test_adjust_enables_safe_on_roi_drop(tmp_path, monkeypatch):
    monkeypatch.delenv("MENACE_SAFE", raising=False)
    roi = DroppingROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(
        db.MetricRecord(
            bot="b",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=0,
        )
    )
    svc = so.SelfServiceOverride(roi, metrics)
    svc.adjust()
    assert os.environ.get("MENACE_SAFE") == "1"


def test_adjust_enables_safe_on_error_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("MENACE_SAFE", raising=False)
    roi = FlatROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(
        db.MetricRecord(
            bot="b",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=10,
        )
    )
    svc = so.SelfServiceOverride(roi, metrics)
    svc.adjust()
    assert os.environ.get("MENACE_SAFE") == "1"


def test_auto_rollback_service_triggers_revert(tmp_path, monkeypatch):
    monkeypatch.delenv("MENACE_SAFE", raising=False)

    roi = DroppingROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    metrics.add(
        db.MetricRecord(
            bot="b",
            cpu=0.0,
            memory=0.0,
            response_time=0.0,
            disk_io=0.0,
            net_io=0.0,
            errors=10,
        )
    )
    metrics.log_eval("system", "avg_energy_score", 0.2)

    called = []

    def fake_revert(cmd, check=True, stdout=None, stderr=None):
        called.append(cmd)
        return None

    monkeypatch.setattr(subprocess, "run", fake_revert)

    svc = so.AutoRollbackService(roi, metrics)
    svc.adjust()
    assert called and called[0][0] == "git"
    assert os.environ.get("MENACE_SAFE") == "1"
