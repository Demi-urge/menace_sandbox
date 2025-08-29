import ast
import importlib
import logging
import types
import time
from pathlib import Path
from unittest.mock import patch

import pytest


def _load_proxies():
    src = Path("self_improvement_engine.py").read_text()
    tree = ast.parse(src)
    wanted = {"_load_callable", "_call_with_retries", "integrate_orphans", "post_round_orphan_scan", "generate_patch"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    module = ast.Module(nodes, type_ignores=[])
    from typing import Callable, Any

    ns = {"importlib": importlib, "logging": logging, "time": time, "Callable": Callable, "Any": Any}
    exec(compile(module, "<ast>", "exec"), ns)
    return ns



def test_missing_dependency_raises_runtime_error():
    proxies = _load_proxies()
    proxies["time"].sleep = lambda *a, **k: None

    with patch("importlib.import_module", side_effect=ModuleNotFoundError):
        with pytest.raises(RuntimeError):
            proxies["integrate_orphans"]()
        with pytest.raises(RuntimeError):
            proxies["post_round_orphan_scan"]()
        with pytest.raises(RuntimeError):
            proxies["generate_patch"]()



def test_retry_succeeds_after_transient_failure():
    proxies = _load_proxies()
    proxies["time"].sleep = lambda *a, **k: None

    counts = {"integrate": 0, "scan": 0, "patch": 0}

    def stub_integrate(*args, **kwargs):
        counts["integrate"] += 1
        if counts["integrate"] < 2:
            raise ValueError("boom")
        return ["ok"]

    def stub_scan(*args, **kwargs):
        counts["scan"] += 1
        if counts["scan"] < 2:
            raise ValueError("boom")
        return {"status": "ok"}

    def stub_patch(*args, **kwargs):
        counts["patch"] += 1
        if counts["patch"] < 2:
            raise ValueError("boom")
        return 1

    module = types.SimpleNamespace(
        integrate_orphans=stub_integrate,
        post_round_orphan_scan=stub_scan,
        generate_patch=stub_patch,
    )

    def fake_import(name):
        return module

    with patch("importlib.import_module", side_effect=fake_import):
        assert proxies["integrate_orphans"](retries=3) == ["ok"]
        assert counts["integrate"] == 2
        assert proxies["post_round_orphan_scan"](retries=3) == {"status": "ok"}
        assert counts["scan"] == 2
        assert proxies["generate_patch"](retries=3) == 1
        assert counts["patch"] == 2
