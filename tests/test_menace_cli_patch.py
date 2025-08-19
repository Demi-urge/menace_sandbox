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
        publish_with_retry=lambda *a, **k: None, with_retry=lambda *a, **k: None
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={})
)

quick_fix_engine = types.SimpleNamespace()
sys.modules["quick_fix_engine"] = quick_fix_engine

import menace_cli
import patch_provenance
import vector_service


class DummyContextBuilder:
    def build(self, *a, **k):
        return ""


# ---------------------------------------------------------------------------

def test_patch_success(monkeypatch, tmp_path, capsys):
    module = tmp_path / "mod.py"
    module.write_text("x=1\n")

    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)

    monkeypatch.setattr(
        quick_fix_engine,
        "generate_patch",
        lambda module, **k: 123,
        raising=False,
    )
    monkeypatch.setattr(
        menace_cli,
        "get_patch_provenance",
        lambda pid: [{"origin": "db", "vector_id": "v1", "influence": 1.0}],
    )

    rc = menace_cli.main(["patch", str(module), "--desc", "fix"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["patch_id"] == 123
    assert out["provenance"][0]["vector_id"] == "v1"


# ---------------------------------------------------------------------------

def test_patch_invalid_path(monkeypatch, tmp_path):
    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)
    monkeypatch.setattr(
        quick_fix_engine, "generate_patch", lambda *a, **k: None, raising=False
    )

    rc = menace_cli.main([
        "patch",
        str(tmp_path / "missing.py"),
        "--desc",
        "bad",
    ])
    assert rc == 1


# ---------------------------------------------------------------------------

def test_patch_description_and_context(monkeypatch, tmp_path):
    module = tmp_path / "mod2.py"
    module.write_text("x=1\n")
    monkeypatch.setattr(vector_service, "ContextBuilder", DummyContextBuilder)

    captured = {}

    def fake_generate_patch(module, **kwargs):
        captured["description"] = kwargs.get("description")
        captured["context"] = kwargs.get("context")
        return 7

    monkeypatch.setattr(
        quick_fix_engine, "generate_patch", fake_generate_patch, raising=False
    )
    monkeypatch.setattr(menace_cli, "get_patch_provenance", lambda pid: [])

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
    assert captured["context"] == {"foo": "bar"}
