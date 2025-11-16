import importlib
import types
import sys
import pytest

ss_stub = types.ModuleType("menace.sandbox_settings")
ss_stub.SandboxSettings = lambda: types.SimpleNamespace(
    patch_retries=1, patch_retry_delay=0.0
)
sys.modules["menace.sandbox_settings"] = ss_stub

patch_generation = importlib.import_module("menace.self_improvement.patch_generation")


def test_wrapper_forwards_builder_and_requires(monkeypatch):
    record = {}

    def fake_generate(*args, **kwargs):
        record["args"] = args
        record["kwargs"] = kwargs
        return 42

    monkeypatch.setattr(
        patch_generation, "_load_callable", lambda *a, **k: fake_generate
    )
    monkeypatch.setattr(
        patch_generation, "_call_with_retries", lambda func, *a, **k: func(*a, **k)
    )
    builder = object()
    manager = object()
    assert patch_generation.generate_patch("mod", manager, context_builder=builder) == 42
    assert record["kwargs"]["context_builder"] is builder
    with pytest.raises(TypeError):
        patch_generation.generate_patch("mod", manager)
