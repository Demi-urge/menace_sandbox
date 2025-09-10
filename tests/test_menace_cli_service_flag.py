import argparse

import menace_cli
import types
import sys


def test_cli_check_vector_service(monkeypatch):
    called = {"count": 0}

    def fake_require(*a, **k):
        called["count"] += 1
        return False

    monkeypatch.setattr(menace_cli, "_require_vector_service", fake_require)
    monkeypatch.setattr(menace_cli, "handle_patch", lambda args: 0)
    dummy_plugins = types.SimpleNamespace(load_plugins=lambda sub: None)
    monkeypatch.setitem(sys.modules, "menace.plugins", dummy_plugins)

    exit_code = menace_cli.main([
        "--check-vector-service",
        "patch",
        "mod",
        "--desc",
        "d",
    ])
    assert exit_code == 1
    assert called["count"] == 1
