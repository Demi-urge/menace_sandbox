import argparse
import sys
import types
from pathlib import Path


# Stub modules required by menace_cli imports
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
        with_retry=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={}),
)
sys.modules.setdefault("menace.plugins", types.SimpleNamespace(load_plugins=lambda *a, **k: None))

import menace_cli


def _capture_run(store):
    def runner(cmd):
        store["cmd"] = cmd
        return 0

    return runner


def test_handle_new_db_uses_resolved_path(monkeypatch):
    store = {}

    def fake_resolve(path):
        store["resolved"] = path
        return Path("/tmp/db_script.py")  # path-ignore

    monkeypatch.setattr(menace_cli, "resolve_path", fake_resolve)
    monkeypatch.setattr(menace_cli, "_run", _capture_run(store))

    args = argparse.Namespace(name="demo")
    rc = menace_cli.handle_new_db(args)
    assert rc == 0
    assert store["resolved"] == "scripts/new_db_template.py"  # path-ignore
    assert store["cmd"][1] == "/tmp/db_script.py"  # path-ignore


def test_handle_new_vector_uses_resolved_path(monkeypatch):
    store = {}

    def fake_resolve(path):
        store["resolved"] = path
        return Path("/tmp/vector_script.py")  # path-ignore

    monkeypatch.setattr(menace_cli, "resolve_path", fake_resolve)
    monkeypatch.setattr(menace_cli, "_run", _capture_run(store))

    args = argparse.Namespace(
        name="demo",
        root=None,
        register_router=False,
        create_migration=False,
    )
    rc = menace_cli.handle_new_vector(args)
    assert rc == 0
    assert store["resolved"] == "scripts/new_vector_module.py"  # path-ignore
    assert store["cmd"][1] == "/tmp/vector_script.py"  # path-ignore

