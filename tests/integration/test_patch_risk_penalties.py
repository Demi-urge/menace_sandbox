import sys
import types

import pytest

sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))

from patch_safety import PatchSafety, _VIOLATIONS
from vector_metrics_db import VectorMetricsDB
from vector_service.patch_logger import PatchLogger
from vector_service.cognition_layer import CognitionLayer


class DummyBuilder:
    def __init__(self):
        self.db_weights = {}

    def refresh_db_weights(self, weights):
        self.db_weights.update(weights)


class DummyBus:
    def publish(self, *_, **__):
        pass


def test_failing_patch_increases_risk_and_penalises_ranker(tmp_path):
    ps = PatchSafety(threshold=1.1, max_alerts=1, storage_path=str(tmp_path / "failures.jsonl"), failure_db_path=None)
    vm = VectorMetricsDB(tmp_path / "vec.db")
    bus = DummyBus()
    pl = PatchLogger(patch_safety=ps, vector_metrics=vm, max_alerts=1, event_bus=bus)
    layer = CognitionLayer(
        retriever=None,
        context_builder=DummyBuilder(),
        patch_logger=pl,
        vector_metrics=vm,
        event_bus=bus,
    )

    meta_fail = {
        "code:safe": {"category": "fail", "module": "m", "license": "mit"},
        "blocked:deny": {"license": "GPL-3.0"},
        "warn:alert": {"semantic_alerts": ["x", "y"]},
    }
    start_license = _VIOLATIONS.labels("license")._value.get()
    start_alerts = _VIOLATIONS.labels("alerts")._value.get()
    res1 = pl.track_contributors(list(meta_fail), False, retrieval_metadata=meta_fail)
    assert _VIOLATIONS.labels("license")._value.get() == start_license + 1
    assert _VIOLATIONS.labels("alerts")._value.get() == start_alerts + 1
    assert "blocked" not in res1
    assert "warn" not in res1
    risk1 = res1.get("code", 0.0)

    meta_safe = {"code:new": {"category": "fail", "module": "m", "license": "mit"}}
    res2 = pl.track_contributors(["code:new"], True, retrieval_metadata=meta_safe)
    risk2 = res2["code"]
    assert risk2 > risk1

    vectors = [("code", "v1", 0.0), ("other", "v2", 0.0)]
    layer.update_ranker(vectors, True)
    pl.patch_safety.threshold = 0.0
    updates = layer.update_ranker(vectors, True, risk_scores=res2)
    assert updates["code"] < updates["other"]


def test_patch_logger_respects_severity_threshold(tmp_path):
    ps = PatchSafety(max_alert_severity=0.5, failure_db_path=None)
    pl = PatchLogger(patch_safety=ps, max_alert_severity=0.5)
    start = _VIOLATIONS.labels("severity")._value.get()
    res = pl.track_contributors(
        ["code:sev"],
        True,
        retrieval_metadata={"code:sev": {"alignment_severity": 0.9}},
    )
    assert _VIOLATIONS.labels("severity")._value.get() == start + 1
    assert res == {}
