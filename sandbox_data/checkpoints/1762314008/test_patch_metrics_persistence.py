
import sys
from pathlib import Path
import types
import sqlite3

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

db_stub = types.SimpleNamespace(
    DBRouter=object, GLOBAL_ROUTER=object(), init_db_router=lambda *a, **k: object()
)
sys.modules.setdefault("db_router", db_stub)

from vector_service.patch_logger import PatchLogger
from vector_metrics_db import VectorMetricsDB
from enhancement_score import EnhancementMetrics, compute_enhancement_score


class SimplePatchDB:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE patch_history(id INTEGER PRIMARY KEY, patch_difficulty INTEGER, time_to_completion REAL, error_trace_count INTEGER, effort_estimate REAL, enhancement_score REAL, tests_failed_after INTEGER)"
        )

    def add(self):
        cur = self.conn.execute("INSERT INTO patch_history DEFAULT VALUES")
        self.conn.commit()
        return cur.lastrowid

    def record_provenance(self, *a, **k):
        pass

    def log_ancestry(self, *a, **k):
        pass

    def log_contributors(self, *a, **k):
        pass

    def get(self, *a, **k):
        return None

    def record_vector_metrics(
        self,
        session_id,
        vectors,
        *,
        patch_id,
        contribution,
        roi_delta=None,
        win,
        regret,
        lines_changed=None,
        tests_passed=None,
        tests_failed_after=None,
        context_tokens=None,
        patch_difficulty=None,
        effort_estimate=None,
        enhancement_name=None,
        timestamp=None,
        start_time=None,
        time_to_completion=None,
        roi_deltas=None,
        diff=None,
        summary=None,
        outcome=None,
        errors=None,
        error_trace_count=None,
        roi_tag=None,
        enhancement_score=None,
    ):
        self.conn.execute(
            "UPDATE patch_history SET patch_difficulty=?, time_to_completion=?, error_trace_count=?, effort_estimate=?, enhancement_score=?, tests_failed_after=? WHERE id=?",
            (
                patch_difficulty,
                time_to_completion,
                error_trace_count,
                effort_estimate,
                enhancement_score,
                tests_failed_after,
                patch_id,
            ),
        )
        self.conn.commit()


def _dummy_patch_safety():
    return types.SimpleNamespace(
        evaluate=lambda *a, **k: (True, 0.0, {}),
        record_failure=lambda *a, **k: None,
        threshold=0.0,
        max_alert_severity=1.0,
        max_alerts=5,
        license_denylist=set(),
    )


def test_patch_logger_persists_metrics(tmp_path):
    vm_path = tmp_path / "vm.db"
    ph_path = tmp_path / "ph.db"
    vmdb = VectorMetricsDB(vm_path)
    patch_db = SimplePatchDB(ph_path)
    patch_id = patch_db.add()

    pl = PatchLogger(patch_db=patch_db, vector_metrics=vmdb)
    pl.patch_safety = _dummy_patch_safety()

    meta = {"patch:1": {"prompt_tokens": 3}}
    pl.track_contributors(
        ["patch:1"],
        True,
        patch_id=str(patch_id),
        session_id="s",
        retrieval_metadata=meta,
        lines_changed=2,
        tests_passed=True,
        start_time=1.0,
        end_time=4.0,
        error_summary="err",
        effort_estimate=2.0,
    )
    metrics = EnhancementMetrics(
        lines_changed=2,
        context_tokens=3,
        time_to_completion=3.0,
        tests_passed=1,
        tests_failed=0,
        error_traces=1,
        effort_estimate=2.0,
    )
    expected_score = compute_enhancement_score(metrics)

    vm_row = vmdb.conn.execute(
        "SELECT patch_difficulty, time_to_completion, error_trace_count, effort_estimate, enhancement_score FROM patch_metrics WHERE patch_id=?",
        (str(patch_id),),
    ).fetchone()
    assert vm_row == (5, 3.0, 1, 2.0, pytest.approx(expected_score))

    ph_row = patch_db.conn.execute(
        "SELECT patch_difficulty, time_to_completion, error_trace_count, effort_estimate, enhancement_score FROM patch_history WHERE id=?",
        (patch_id,),
    ).fetchone()
    assert ph_row == (5, 3.0, 1, 2.0, pytest.approx(expected_score))
