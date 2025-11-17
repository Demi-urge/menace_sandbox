import types
import logging

import menace_sandbox.quick_fix_engine as quick_fix_engine


def test_module_wrapper_skips_on_validation_flags(monkeypatch):
    called = False

    def _generate_patch(*_a, **_k):
        nonlocal called
        called = True
        return 1, []

    monkeypatch.setattr(quick_fix_engine, "generate_patch", _generate_patch)

    manager = types.SimpleNamespace(context_builder=object(), engine=None)
    success, patch_id, flags = quick_fix_engine.apply_validated_patch(
        "module.py",
        description="desc",
        flags=["validation_error"],
        provenance_token="token",
        manager=manager,
        context_builder=manager.context_builder,
    )

    assert success is False
    assert patch_id is None
    assert flags == ["validation_error"]
    assert called is False


def test_engine_apply_skips_on_validation_flags(monkeypatch):
    def _generate_patch(*_a, **_k):  # pragma: no cover - defensive
        raise AssertionError("generate_patch should not be invoked")

    monkeypatch.setattr(quick_fix_engine, "generate_patch", _generate_patch)

    engine = quick_fix_engine.QuickFixEngine.__new__(quick_fix_engine.QuickFixEngine)
    engine.logger = logging.getLogger("quick_fix_engine_test")
    engine.manager = types.SimpleNamespace()
    engine.context_builder = object()
    engine.helper_fn = None
    engine.patch_logger = None
    engine.graph = None

    success, patch_id, flags = engine.apply_validated_patch(
        "module.py", flags=["static_hint:error"], provenance_token="token"
    )

    assert success is False
    assert patch_id is None
    assert flags == ["static_hint:error"]
