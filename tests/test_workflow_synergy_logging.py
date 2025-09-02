from pathlib import Path
from unittest.mock import patch

from workflow_synergy_comparator import WorkflowSynergyComparator


def test_update_best_practices_logs_write_failure(tmp_path, monkeypatch):
    path = tmp_path / "best.json"
    WorkflowSynergyComparator.best_practices_file = path

    def failing_write(self, *args, **kwargs):
        raise IOError("disk full")

    monkeypatch.setattr(Path, "write_text", failing_write)

    with patch("workflow_synergy_comparator.logger.warning") as warn:
        WorkflowSynergyComparator._update_best_practices(["m1", "m2"])

    warn.assert_called()
    # Ensure the path of the failing file is included in the log call
    assert str(path) in str(warn.call_args[0][1]) or str(path) in str(warn.call_args[0][0])
