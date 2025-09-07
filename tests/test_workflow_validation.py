import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import menace.watchdog as wd  # noqa: E402
import menace.error_bot as eb  # noqa: E402
import menace.resource_allocation_optimizer as rao  # noqa: E402
import menace.data_bot as db  # noqa: E402
from menace.recovery_validator import ReplayValidator  # noqa: E402
from menace.menace_orchestrator import MenaceOrchestrator  # noqa: E402


def _setup_dbs(tmp_path):
    err = eb.ErrorDB(tmp_path / "e.db")
    roi = rao.ROIDB(tmp_path / "r.db")
    metrics = db.MetricsDB(tmp_path / "m.db")
    return err, roi, metrics


def test_replay_updates_confidence(tmp_path):
    err_db, roi_db, metrics_db = _setup_dbs(tmp_path)
    builder = wd.get_default_context_builder()
    watch = wd.Watchdog(err_db, roi_db, metrics_db, context_builder=builder)
    orch = MenaceOrchestrator(context_builder=builder)
    watch.record_fault("fail", workflow="wf1")
    validator = ReplayValidator(lambda wf: None)
    res = watch.validate_workflows(validator, orch)
    assert res == {"wf1": True}
    assert orch.workflow_confidence["wf1"] == 1.0

    watch.record_fault("fail", workflow="wf1")
    validator = ReplayValidator(lambda wf: (_ for _ in ()).throw(Exception()))
    res = watch.validate_workflows(validator, orch)
    assert res == {"wf1": False}
    assert 0.4 <= orch.workflow_confidence["wf1"] <= 0.6
