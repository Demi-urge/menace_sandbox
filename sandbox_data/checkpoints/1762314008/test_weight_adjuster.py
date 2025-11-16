import types
import pytest

from vector_metrics_db import VectorMetricsDB
from vector_service.weight_adjuster import WeightAdjuster, RoiTag
from vector_service.patch_logger import PatchLogger


def test_weight_adjuster_updates_weights(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    vectors = [("patch", "v1", 1.0, RoiTag.SUCCESS)]

    adj.adjust(vectors, tests_passed=True)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.2)

    adj.adjust([("patch", "v1", 1.0, RoiTag.LOW_ROI)], tests_passed=True)
    assert db.get_vector_weight("patch:v1") == pytest.approx(0.1)


def test_weight_adjuster_scales_by_quality(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm2.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    vectors = [("patch", "v1", 1.0, RoiTag.SUCCESS)]

    adj.adjust(vectors, tests_passed=True)
    base = db.get_vector_weight("patch:v1")

    adj.adjust(
        vectors,
        tests_passed=False,
        error_trace_count=2,
    )
    assert db.get_vector_weight("patch:v1") == pytest.approx(base - (0.1 / 3))


def test_track_contributors_updates_vector_weights(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm3.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    ps = types.SimpleNamespace(
        evaluate=lambda *a, **k: (True, 0.0, {}),
        record_failure=lambda *a, **k: None,
        threshold=0.0,
        max_alert_severity=1.0,
        max_alerts=5,
        license_denylist=set(),
    )
    pl = PatchLogger(vector_metrics=db, patch_safety=ps, weight_adjuster=adj)

    pl.track_contributors(["db:v1", "db:v2"], True, lines_changed=1)

    assert db.get_vector_weight("db:v1") == pytest.approx(0.2)
    assert db.get_vector_weight("db:v2") == pytest.approx(0.2)


def test_track_contributors_negative_roi_decreases_weight(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm4.db")
    adj = WeightAdjuster(
        vector_metrics=db,
        db_success_delta=0.2,
        db_failure_delta=0.1,
        vector_success_delta=0.2,
        vector_failure_delta=0.1,
    )
    ps = types.SimpleNamespace(
        evaluate=lambda *a, **k: (True, 0.0, {}),
        record_failure=lambda *a, **k: None,
        threshold=0.0,
        max_alert_severity=1.0,
        max_alerts=5,
        license_denylist=set(),
    )
    pl = PatchLogger(vector_metrics=db, patch_safety=ps, weight_adjuster=adj)

    pl.track_contributors(["db:v1"], True, lines_changed=1)
    pl.track_contributors(["db:v1"], True, lines_changed=1, roi_tag=RoiTag.LOW_ROI)

    assert db.get_vector_weight("db:v1") == pytest.approx(0.1)
