from __future__ import annotations

from pathlib import Path
import types

import pytest

from menace_sandbox.self_improvement import patch_application


class DummyBuilder:
    def __init__(self, token: str = "prov") -> None:
        self.provenance_token = token


def test_validate_patch_with_context_calls_quick_fix(monkeypatch, tmp_path):
    module = tmp_path / "mod.py"
    module.write_text("x=1\n")

    captured: dict[str, object] = {}

    def fake_create_builder(*_, **__):
        return DummyBuilder()

    def fake_validate(**kwargs):
        captured["validate_kwargs"] = kwargs
        return True, []

    def fake_apply(**kwargs):
        captured["apply_kwargs"] = kwargs
        return True, 42, []

    monkeypatch.setattr(patch_application, "create_context_builder", fake_create_builder)
    monkeypatch.setattr(patch_application.quick_fix, "validate_patch", lambda *_, **kw: fake_validate(**kw))
    monkeypatch.setattr(patch_application.quick_fix, "apply_validated_patch", lambda *_, **kw: fake_apply(**kw))

    manager = types.SimpleNamespace()
    result = patch_application.validate_patch_with_context(
        module_path=module,
        description="desc",
        repo_root=tmp_path,
        manager=manager,
        context_meta={"foo": "bar"},
    )

    validate_kwargs = captured["validate_kwargs"]
    assert validate_kwargs["description"] == "desc"
    assert validate_kwargs["manager"] is manager
    assert Path(validate_kwargs["module_path"]) == module

    apply_kwargs = captured["apply_kwargs"]
    assert apply_kwargs["flags"] == []
    assert apply_kwargs["context_meta"]["target_module"] == str(module)
    assert apply_kwargs["context_meta"]["foo"] == "bar"
    assert result["patch_id"] == 42


def test_validate_patch_with_context_merges_validation_context(monkeypatch, tmp_path):
    module = tmp_path / "mod.py"
    module.write_text("x=1\n")

    captured: dict[str, object] = {}

    def fake_validate(**kwargs):
        captured["validate_kwargs"] = kwargs
        return True, [], {"source": "validator"}

    def fake_apply(**kwargs):
        captured["apply_kwargs"] = kwargs
        return True, 99, []

    monkeypatch.setattr(patch_application, "create_context_builder", lambda *_, **__: DummyBuilder())
    monkeypatch.setattr(patch_application.quick_fix, "validate_patch", lambda *_, **kw: fake_validate(**kw))
    monkeypatch.setattr(patch_application.quick_fix, "apply_validated_patch", lambda *_, **kw: fake_apply(**kw))

    manager = types.SimpleNamespace()
    patch_application.validate_patch_with_context(
        module_path=module,
        description="desc",  # noqa: B009 - explicit description for clarity
        repo_root=tmp_path,
        manager=manager,
        context_meta={"foo": "bar"},
    )

    apply_ctx = captured["apply_kwargs"]["context_meta"]
    assert apply_ctx["source"] == "validator"
    assert apply_ctx["foo"] == "bar"
    assert apply_ctx["target_module"] == str(module)


def test_validate_patch_with_context_raises_on_flags(monkeypatch, tmp_path):
    module = tmp_path / "mod.py"
    module.write_text("x=1\n")

    monkeypatch.setattr(patch_application, "create_context_builder", lambda *a, **k: DummyBuilder())
    monkeypatch.setattr(
        patch_application.quick_fix,
        "validate_patch",
        lambda *_, **__: (True, ["schema_error"]),
    )

    with pytest.raises(RuntimeError) as excinfo:
        patch_application.validate_patch_with_context(
            module_path=module,
            description="desc",
            repo_root=tmp_path,
            manager=types.SimpleNamespace(),
        )

    assert "schema_error" in str(excinfo.value)
