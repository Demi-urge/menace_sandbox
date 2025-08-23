import sys
import sqlite3

import db_router


def test_sandbox_dashboard_initialises_router(monkeypatch):
    calls = []

    def fake_init(menace_id, *a, **k):
        calls.append(menace_id)
        class Dummy:
            def get_connection(self, *a, **k):
                return sqlite3.connect(":memory:")
        db_router.GLOBAL_ROUTER = Dummy()
        return db_router.GLOBAL_ROUTER

    monkeypatch.setattr(db_router, "init_db_router", fake_init)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)
    for mod in [
        "menace.sandbox_dashboard",
        "menace.roi_tracker",
        "menace.metrics_dashboard",
        "menace.alignment_dashboard",
        "menace.readiness_index",
        "menace.synergy_weight_cli",
    ]:
        sys.modules.pop(mod, None)
    __import__("menace.sandbox_dashboard")  # noqa: F401
    assert len(calls) == 1
    assert db_router.GLOBAL_ROUTER is not None


def test_error_ontology_dashboard_initialises_router(monkeypatch):
    calls = []

    def fake_init(menace_id, *a, **k):
        calls.append(menace_id)
        class Dummy:
            def get_connection(self, *a, **k):
                return sqlite3.connect(":memory:")
        db_router.GLOBAL_ROUTER = Dummy()
        return db_router.GLOBAL_ROUTER

    monkeypatch.setattr(db_router, "init_db_router", fake_init)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)
    for mod in [
        "menace.error_ontology_dashboard",
        "menace.error_bot",
        "menace.knowledge_graph",
        "menace.error_cluster_predictor",
        "menace.metrics_dashboard",
    ]:
        sys.modules.pop(mod, None)
    __import__("menace.error_ontology_dashboard")  # noqa: F401
    assert len(calls) == 1
    assert db_router.GLOBAL_ROUTER is not None
