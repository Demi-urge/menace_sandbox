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
pkg.RAISE_ERRORS = False

# stub lightweight dependencies to avoid heavy imports
rao_mod = types.ModuleType("menace.resource_allocation_optimizer")
class _ROIDB:
    def __init__(self, *a, **k):
        pass
    def history(self, limit=2):  # pragma: no cover - not used in tests
        return []
rao_mod.ROIDB = _ROIDB
sys.modules.setdefault("menace.resource_allocation_optimizer", rao_mod)

db_mod = types.ModuleType("menace.data_bot")
class _MetricsDB:
    def __init__(self, *a, **k):
        pass
    def fetch(self, limit=10):  # pragma: no cover - not used in tests
        return []
    def fetch_eval(self, *a, **k):  # pragma: no cover - not used
        return []
db_mod.MetricsDB = _MetricsDB
db_mod.MetricRecord = object
db_mod.DataBot = object
sys.modules.setdefault("menace.data_bot", db_mod)

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
    roi = FlatROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    tracker = so.BaselineTracker(window=3, deviation_multipliers={"roi": 0.5, "error": 0.5})
    svc = so.SelfServiceOverride(roi, metrics, tracker=tracker)
    roi_values = iter([0.0, 0.05, 0.0, 0.4, 0.5, 0.6])
    monkeypatch.setattr(svc, "_calc_roi_drop", lambda: next(roi_values))
    monkeypatch.setattr(svc, "_error_rate", lambda: 0.0)
    for _ in range(6):
        svc.adjust()
    assert os.environ.get("MENACE_SAFE") == "1"


def test_adjust_enables_safe_on_error_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("MENACE_SAFE", raising=False)
    roi = FlatROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    tracker = so.BaselineTracker(window=3, deviation_multipliers={"roi": 0.5, "error": 0.5})
    svc = so.SelfServiceOverride(roi, metrics, tracker=tracker)
    err_values = iter([0.0, 0.1, 0.0, 0.4, 0.5, 0.6])
    monkeypatch.setattr(svc, "_calc_roi_drop", lambda: 0.0)
    monkeypatch.setattr(svc, "_error_rate", lambda: next(err_values))
    for _ in range(6):
        svc.adjust()
    assert os.environ.get("MENACE_SAFE") == "1"


def test_auto_rollback_service_triggers_revert(tmp_path, monkeypatch):
    monkeypatch.delenv("MENACE_SAFE", raising=False)
    roi = FlatROI()
    metrics = db.MetricsDB(tmp_path / "m.db")
    tracker = so.BaselineTracker(
        window=5, deviation_multipliers={"roi": 0.5, "error": 0.5, "energy": 0.5}
    )
    svc = so.AutoRollbackService(roi, metrics, tracker=tracker)
    monkeypatch.setattr(svc, "_calc_roi_drop", lambda: 0.0)
    monkeypatch.setattr(svc, "_error_rate", lambda: 0.0)
    energy_values = iter([1.0, 0.9, 1.1, 1.0, 1.05, 0.2, 0.2, 0.2])
    monkeypatch.setattr(svc, "_energy_score", lambda: next(energy_values))

    called = []

    def fake_revert(cmd, check=True, stdout=None, stderr=None):
        called.append(cmd)
        return None

    monkeypatch.setattr(subprocess, "run", fake_revert)

    for _ in range(8):
        svc.adjust()
    assert called and called[-1][0] == "git"
    assert os.environ.get("MENACE_SAFE") == "1"
