import json
import importlib
from pathlib import Path
from pytest import approx


def test_load_and_merge_existing_summaries(tmp_path, monkeypatch):
    summary_store = tmp_path / "workflows"
    summary_store.mkdir()
    (summary_store / "foo.summary.json").write_text(
        json.dumps(
            {
                "workflow_id": "foo",
                "cumulative_roi": 10.0,
                "num_runs": 2,
                "average_roi": 5.0,
            }
        )
    )

    monkeypatch.setenv("WORKFLOW_SUMMARY_STORE", str(summary_store))
    monkeypatch.setenv("WORKFLOW_ROI_HISTORY_PATH", str(tmp_path / "roi_history.json"))

    import menace.workflow_run_summary as wrs
    wrs = importlib.reload(wrs)

    assert wrs._WORKFLOW_ROI_HISTORY["foo"] == [5.0, 5.0]

    wrs.reset_history()
    wrs.record_run("foo", 7.0)
    path = wrs.save_summary("foo", summary_store)
    data = json.loads(Path(path).read_text())
    assert data["cumulative_roi"] == approx(17.0)
    assert data["num_runs"] == 3
    assert data["average_roi"] == approx(17.0 / 3)
