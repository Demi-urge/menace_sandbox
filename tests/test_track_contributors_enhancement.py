import types
import sys

import pytest

db_stub = types.SimpleNamespace(
    DBRouter=object, GLOBAL_ROUTER=object(), init_db_router=lambda *a, **k: object()
)
sys.modules.setdefault("db_router", db_stub)

from vector_service.patch_logger import PatchLogger
from enhancement_score import EnhancementMetrics, compute_enhancement_score


class DummyPatchDB:
    def __init__(self):
        self.kwargs = None

    def record_vector_metrics(
        self,
        session_id,
        pairs,
        *,
        patch_id,
        contribution,
        roi_delta=None,
        win,
        regret,
        lines_changed,
        tests_passed,
        tests_failed_after,
        context_tokens,
        patch_difficulty,
        effort_estimate,
        enhancement_name,
        start_time,
        time_to_completion,
        timestamp,
        errors,
        error_trace_count,
        roi_tag,
        diff,
        summary,
        outcome,
        roi_deltas,
        enhancement_score,
    ):
        self.kwargs = {
            "roi_tag": roi_tag,
            "enhancement_score": enhancement_score,
        }

    # The following methods are invoked by PatchLogger but are not
    # relevant for this test; they simply satisfy the interface.
    def record_provenance(self, *a, **k):  # pragma: no cover - interface stub
        pass

    def log_ancestry(self, *a, **k):  # pragma: no cover - interface stub
        pass

    def log_contributors(self, *a, **k):  # pragma: no cover - interface stub
        pass

    def get(self, patch_id):  # pragma: no cover - not used
        return None


class DummyVectorMetrics:
    def __init__(self):
        self.kwargs = None

    def record_patch_summary(self, *args, **kwargs):
        self.kwargs = kwargs

    def update_outcome(self, *a, **k):  # pragma: no cover - interface stub
        pass

    def record_patch_ancestry(self, *a, **k):  # pragma: no cover - interface stub
        pass


def test_track_contributors_enhancement_score_and_roi_tag():
    pdb = DummyPatchDB()
    vm = DummyVectorMetrics()
    pl = PatchLogger(
        patch_db=pdb,
        vector_metrics=vm,
        weight_adjuster=types.SimpleNamespace(adjust=lambda *a, **k: None),
    )
    res = pl.track_contributors(
        ["db:v1"],
        True,
        patch_id="1",
        session_id="s",
        lines_changed=10,
        tests_passed=True,
        start_time=0.0,
        end_time=30.0,
        roi_tag="low-ROI",
        effort_estimate=5.0,
    )
    metrics = EnhancementMetrics(
        lines_changed=10,
        context_tokens=0,
        time_to_completion=30.0,
        tests_passed=1,
        tests_failed=0,
        error_traces=0,
        effort_estimate=5.0,
    )
    expected = compute_enhancement_score(metrics)
    assert res.enhancement_score == pytest.approx(expected)
    assert pdb.kwargs["roi_tag"] == "low-ROI"
    assert vm.kwargs["roi_tag"] == "low-ROI"
    assert pdb.kwargs["enhancement_score"] == pytest.approx(expected)
    assert vm.kwargs["enhancement_score"] == pytest.approx(expected)


def test_recompute_enhancement_score_updates_metrics():
    vm = DummyVectorMetrics()
    pl = PatchLogger(patch_db=None, vector_metrics=vm)
    metrics = EnhancementMetrics(
        lines_changed=5,
        context_tokens=3,
        time_to_completion=2.0,
        tests_passed=1,
        tests_failed=0,
        error_traces=0,
        effort_estimate=1.0,
    )
    score = pl.recompute_enhancement_score("42", metrics, roi_tag="high-ROI")
    expected = compute_enhancement_score(metrics)
    assert score == pytest.approx(expected)
    assert vm.kwargs["enhancement_score"] == pytest.approx(expected)
    assert vm.kwargs["roi_tag"] == "high-ROI"
