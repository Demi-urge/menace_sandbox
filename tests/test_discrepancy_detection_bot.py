import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import importlib.metadata as metadata
import menace.discrepancy_detection_bot as ddb
import menace.failure_learning_system as fls
import menace.resource_allocation_optimizer as rao


def test_detection_triggers_yank(monkeypatch, tmp_path):
    db = fls.DiscrepancyDB(tmp_path / "d.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    opt = rao.ResourceAllocationOptimizer(roi)
    called = []
    monkeypatch.setattr(opt, "scale_down", lambda wf, amount=1: called.append(wf))

    def rule(models=None, workflows=None, storage=None):
        return [("corrupt_csv", 1.0, "wf1")]

    class EP:
        name = "rule"

        def load(self):
            return rule

    class EPS:
        def select(self, group):
            return [EP()] if group == "menace.discrepancy_rules" else []

    monkeypatch.setattr(metadata, "entry_points", lambda: EPS())
    bot = ddb.DiscrepancyDetectionBot(db=db, optimizer=opt)
    bot.scan()
    df = db.fetch_detections()
    assert not df.empty
    assert called and called[0] == "wf1"


def test_low_severity(monkeypatch, tmp_path):
    db = fls.DiscrepancyDB(tmp_path / "d.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    opt = rao.ResourceAllocationOptimizer(roi)
    called = []
    monkeypatch.setattr(opt, "scale_down", lambda *a, **k: called.append(True))

    def rule(models=None, workflows=None, storage=None):
        return [("minor_issue", 0.2, "wf1")]

    class EP:
        name = "rule"

        def load(self):
            return rule

    class EPS:
        def select(self, group):
            return [EP()] if group == "menace.discrepancy_rules" else []

    monkeypatch.setattr(metadata, "entry_points", lambda: EPS())
    bot = ddb.DiscrepancyDetectionBot(db=db, optimizer=opt)
    bot.scan()
    df = db.fetch_detections()
    assert len(df) == 1
    assert not called
