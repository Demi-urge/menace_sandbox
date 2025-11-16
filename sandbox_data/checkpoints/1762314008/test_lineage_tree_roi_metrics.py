import json
import importlib


def test_lineage_tree_contains_roi_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_ROI_HISTORY_PATH", str(tmp_path / "roi_history.json"))
    summary_store = tmp_path / "summaries"
    monkeypatch.setenv("WORKFLOW_SUMMARY_STORE", str(summary_store))

    import menace.workflow_run_summary as wrs
    wrs = importlib.reload(wrs)
    wrs.reset_history()

    # create parent workflow summary
    parent_spec = {"metadata": {"workflow_id": "1", "mutation_description": "root"}}
    (tmp_path / "1.workflow.json").write_text(json.dumps(parent_spec))
    wrs.record_run("1", 1.0)
    wrs.save_summary("1", tmp_path)

    # create child workflow summary referencing parent
    child_spec = {
        "metadata": {
            "workflow_id": "2",
            "parent_id": "1",
            "mutation_description": "child",
        }
    }
    (tmp_path / "2.workflow.json").write_text(json.dumps(child_spec))
    wrs.record_run("2", 2.0)
    wrs.save_summary("2", tmp_path)

    import menace.evolution_history_db as eh
    import menace.lineage_tracker as lt
    lt = importlib.reload(lt)

    edb = eh.EvolutionHistoryDB(tmp_path / "e.db")
    edb.add(
        eh.EvolutionEvent(
            action="root",
            before_metric=0.0,
            after_metric=1.0,
            roi=1.0,
            workflow_id=1,
        )
    )
    edb.add(
        eh.EvolutionEvent(
            action="child",
            before_metric=1.0,
            after_metric=2.0,
            roi=1.0,
            workflow_id=2,
        )
    )

    tracker = lt.LineageTracker(edb)
    tree = tracker.build_tree()
    nodes = {n["workflow_id"]: n for n in tree}

    assert nodes[1]["average_roi"] == 1.0
    assert nodes[1]["mutation_description"] == "root"
    assert nodes[1]["roi_delta"] is None

    assert nodes[2]["average_roi"] == 2.0
    assert nodes[2]["roi_delta"] == 1.0
    assert nodes[2]["mutation_description"] == "child"
