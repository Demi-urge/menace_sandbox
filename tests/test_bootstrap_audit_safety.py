"""Regression checks for bootstrap-specific database auditing paths."""

import sys
from types import SimpleNamespace

import db_router

from coding_bot_interface import _prepare_pipeline_for_bootstrap_impl


def test_bootstrap_pipeline_uses_audit_safe_router(monkeypatch, tmp_path):
    """Bootstrap pipeline creation should mark DB routing as audit-safe."""

    events: list[tuple[str, object]] = []

    monkeypatch.setattr(db_router, "_audit_bootstrap_safe_default", False)
    monkeypatch.setattr(db_router, "GLOBAL_ROUTER", None)
    monkeypatch.setattr(sys, "platform", "win32")

    def fail_if_audit_imported():
        raise AssertionError("audit logger should not be loaded during bootstrap")

    monkeypatch.setattr(db_router, "_ensure_log_db_access", fail_if_audit_imported)

    class DummyPipeline:
        def __init__(self, *, context_builder, bot_registry, data_bot, manager, **kwargs):
            events.append(("manager", manager))
            router = db_router.init_db_router(
                "bootstrap_test",
                str(tmp_path / "local.db"),
                str(tmp_path / "shared.db"),
            )
            events.append(("audit_safe", router.local_conn.audit_bootstrap_safe))
            router.local_conn.execute("CREATE TABLE IF NOT EXISTS foo(x INTEGER)")

    pipeline, promote = _prepare_pipeline_for_bootstrap_impl(
        pipeline_cls=DummyPipeline,
        context_builder=object(),
        bot_registry=SimpleNamespace(set_bootstrap_mode=lambda *_: None),
        data_bot=object(),
        bootstrap_safe=True,
    )

    assert pipeline is not None
    assert callable(promote)
    assert db_router._audit_bootstrap_safe_default is True
    assert db_router.GLOBAL_ROUTER.local_conn.audit_bootstrap_safe is True
    assert ("audit_safe", True) in events
