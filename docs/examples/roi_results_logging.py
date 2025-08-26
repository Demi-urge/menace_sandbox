"""Example showing how to log workflow metrics to ``roi_results.db``.

Run with ``python docs/examples/roi_results_logging.py``.
"""

from menace_sandbox.roi_results_db import ROIResultsDB

if __name__ == "__main__":
    db = ROIResultsDB("roi_results.db")
    db.log_result(
        workflow_id="wf_demo",
        run_id="run_001",
        runtime=0.5,
        success_rate=1.0,
        roi_gain=1.2,
        workflow_synergy_score=0.9,
        bottleneck_index=0.1,
        patchability_score=0.05,
        module_deltas={
            "step_a": {
                "runtime": 0.5,
                "roi_delta": 1.2,
                "success_rate": 1.0,
                "bottleneck_contribution": 0.1,
            }
        },
    )
    rows = db.conn.execute(
        "SELECT workflow_id, run_id, roi_gain FROM workflow_results"
    ).fetchall()
    print(rows)
