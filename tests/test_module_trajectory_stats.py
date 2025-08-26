import os
from statistics import fmean, pvariance

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

from menace_sandbox.roi_results_db import ROIResultsDB  # noqa: E402


def test_module_trajectories_expose_moving_stats(tmp_path):
    db_path = tmp_path / "roi.db"
    db = ROIResultsDB(db_path)

    deltas = [0.2, 0.4, -0.1]
    for i, delta in enumerate(deltas, 1):
        db.log_result(
            workflow_id="wf",
            run_id=f"r{i}",
            runtime=0.0,
            success_rate=1.0,
            roi_gain=0.0,
            workflow_synergy_score=0.0,
            bottleneck_index=0.0,
            patchability_score=0.0,
            module_deltas={"alpha": {"roi_delta": delta, "success_rate": 1.0}},
        )

    trajectories = db.fetch_module_trajectories("wf", "alpha")["alpha"]

    expected_ma = []
    expected_var = []
    for i in range(len(deltas)):
        subset = deltas[: i + 1]
        expected_ma.append(fmean(subset))
        expected_var.append(pvariance(subset) if len(subset) > 1 else 0.0)

    assert [t["moving_avg"] for t in trajectories] == pytest.approx(expected_ma)
    assert [t["variance"] for t in trajectories] == pytest.approx(expected_var)

    latest = db.fetch_module_volatility("wf", "alpha")
    assert latest["moving_avg"] == pytest.approx(expected_ma[-1])
    assert latest["variance"] == pytest.approx(expected_var[-1])
