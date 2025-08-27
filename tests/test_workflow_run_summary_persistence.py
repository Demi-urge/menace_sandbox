import json
import importlib
from pathlib import Path


def test_roi_history_persistence_and_delta(tmp_path, monkeypatch):
    history_file = tmp_path / "roi_history.json"
    monkeypatch.setenv("WORKFLOW_ROI_HISTORY_PATH", str(history_file))
    monkeypatch.setenv("WORKFLOW_SUMMARY_STORE", str(tmp_path))

    import menace.workflow_run_summary as wrs
    wrs = importlib.reload(wrs)
    wrs.reset_history()

    parent_spec = {"metadata": {"workflow_id": "parent"}}
    (tmp_path / "parent.workflow.json").write_text(json.dumps(parent_spec))
    wrs.record_run("parent", 1.0)
    wrs.save_summary("parent", tmp_path)

    child_spec = {"metadata": {"workflow_id": "child", "parent_id": "parent"}}
    (tmp_path / "child.workflow.json").write_text(json.dumps(child_spec))
    wrs.record_run("child", 2.0)

    # simulate restart
    wrs = importlib.reload(wrs)

    assert wrs._WORKFLOW_ROI_HISTORY["parent"] == [1.0]
    assert wrs._WORKFLOW_ROI_HISTORY["child"] == [2.0]

    summary_path = wrs.save_summary("child", tmp_path)
    data = json.loads(Path(summary_path).read_text())
    assert data["roi_delta"] == 1.0
    assert data["avg_roi_delta"] == 1.0

    monkeypatch.delenv("WORKFLOW_ROI_HISTORY_PATH", raising=False)
    monkeypatch.delenv("WORKFLOW_SUMMARY_STORE", raising=False)
    importlib.reload(wrs)
