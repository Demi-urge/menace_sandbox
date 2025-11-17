import json
import sys
import types

# Stub modules required by imports in menace_cli

def _auto_link(*a, **k):
    def decorator(func):
        return func
    return decorator

sys.modules.setdefault("auto_link", types.SimpleNamespace(auto_link=_auto_link))
sys.modules.setdefault("unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object))
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(
        publish_with_retry=lambda *a, **k: None,
        with_retry=lambda fn, *a, **k: fn(),
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={})
)

quick_fix_engine = types.SimpleNamespace()
sys.modules["quick_fix_engine"] = quick_fix_engine
quick_fix_engine.quick_fix = types.SimpleNamespace()

import menace_cli
import vector_service


class DummyContextBuilder:
    provenance_token = "tok"
    roi_tag_penalties: dict[str, float] = {}

    def build(self, *a, **k):
        return ""

    def refresh_db_weights(self):
        return None


# Force the CLI to use the lightweight dummy builder in tests.
menace_cli.create_context_builder = lambda: DummyContextBuilder()


# ---------------------------------------------------------------------------

def test_patch_success(monkeypatch, tmp_path, capsys):
    module = tmp_path / "mod.py"  # path-ignore
    module.write_text("x=1\n")

    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)
    quick_fix_engine.quick_fix.validate_patch = lambda *a, **k: (True, [])
    quick_fix_engine.quick_fix.apply_validated_patch = (
        lambda *a, **k: (True, 123, [])
    )

    class DummyDB:
        def get(self, pid):
            return types.SimpleNamespace(filename=str(module))

    monkeypatch.setattr(menace_cli, "PatchHistoryDB", lambda: DummyDB())
    monkeypatch.setattr(menace_cli, "PatchLogger", lambda patch_db=None: object())

    rc = menace_cli.main(["patch", str(module), "--desc", "fix"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["patch_id"] == 123
    assert out["files"][0] == str(module)


# ---------------------------------------------------------------------------

def test_patch_invalid_path(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)
    quick_fix_engine.quick_fix.validate_patch = lambda *a, **k: (False, ["missing"])

    rc = menace_cli.main([
        "patch",
        str(tmp_path / "missing.py"),  # path-ignore
        "--desc",
        "bad",
    ])
    assert rc == 1


# ---------------------------------------------------------------------------

def test_patch_bad_context(monkeypatch, tmp_path, capsys):
    module = tmp_path / "mod_bad_ctx.py"  # path-ignore
    module.write_text("x=1\n")
    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)
    quick_fix_engine.quick_fix.validate_patch = lambda *a, **k: (True, [])
    quick_fix_engine.quick_fix.apply_validated_patch = (
        lambda *a, **k: (True, 5, [])
    )

    rc = menace_cli.main([
        "patch",
        str(module),
        "--desc",
        "oops",
        "--context",
        "{not json}",
    ])
    assert rc == 1
    err = capsys.readouterr().err.lower()
    assert "invalid json context" in err


# ---------------------------------------------------------------------------

def test_patch_description_and_context(monkeypatch, tmp_path):
    module = tmp_path / "mod2.py"  # path-ignore
    module.write_text("x=1\n")
    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)

    captured = {}

    def fake_validate_patch(*_args, **kwargs):
        captured["description"] = kwargs.get("description")
        captured["context_meta_validate"] = kwargs.get("context_builder")
        return True, []

    def fake_apply_validated_patch(*_args, **kwargs):
        captured["description_apply"] = kwargs.get("description")
        captured["context"] = kwargs.get("context_meta")
        return True, 7, []

    quick_fix_engine.quick_fix.validate_patch = fake_validate_patch
    quick_fix_engine.quick_fix.apply_validated_patch = fake_apply_validated_patch
    monkeypatch.setattr(menace_cli, "PatchHistoryDB", lambda: types.SimpleNamespace(get=lambda pid: None))
    monkeypatch.setattr(menace_cli, "PatchLogger", lambda patch_db=None: object())

    menace_cli.main(
        [
            "patch",
            str(module),
            "--desc",
            "something",
            "--context",
            "{\"foo\": \"bar\"}",
        ]
    )

    assert captured["description"] == "something"
    assert captured["description_apply"] == "something"
    assert captured["context"]["foo"] == "bar"
    assert captured["context"]["target_module"] == str(module)
