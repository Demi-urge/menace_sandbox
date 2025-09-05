from __future__ import annotations

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

import menace_cli
from scripts import new_db_template


def _dummy_run(cmd: list[str], root):
    name = cmd[-1]
    new_db_template.create_db_scaffold(name, root=root)
    return 0


def test_new_db_scaffold(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "__init__.py").write_text("__all__ = []\n")  # path-ignore

    monkeypatch.setattr(menace_cli, "_run", lambda cmd: _dummy_run(cmd, tmp_path))

    rc = menace_cli.main(["new-db", "sample"])
    assert rc == 0

    mod = tmp_path / "sample_db.py"  # path-ignore
    assert mod.exists()
    text = mod.read_text()
    assert "EmbeddableDBMixin" in text
    assert "build_context" in text
    assert "create_fts.sql" in text
    assert "detect_license" in text
    assert "redact_dict" in text

    init_text = (tmp_path / "__init__.py").read_text()  # path-ignore
    assert "from .sample_db import SampleDB" in init_text
    assert '"sample_db"' in init_text


def test_new_db_failure(monkeypatch):
    monkeypatch.setattr(menace_cli, "_run", lambda cmd: 1)
    rc = menace_cli.main(["new-db", "demo"])
    assert rc == 1
