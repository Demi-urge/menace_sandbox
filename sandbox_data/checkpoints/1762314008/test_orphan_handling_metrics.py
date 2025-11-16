import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Ensure root directory is on sys.path for absolute imports
sys.path.append(str(ROOT))

from dynamic_path_router import resolve_path

# Create package structure without executing heavy initialisers
root_pkg = types.ModuleType("menace_sandbox")
root_pkg.__path__ = [str(ROOT)]
sys.modules.setdefault("menace_sandbox", root_pkg)

self_pkg = types.ModuleType("menace_sandbox.self_improvement")
self_pkg.__path__ = [str(ROOT / "self_improvement")]
sys.modules.setdefault("menace_sandbox.self_improvement", self_pkg)

me_spec = importlib.util.spec_from_file_location(
    "menace_sandbox.metrics_exporter", resolve_path("metrics_exporter.py")
)
metrics_exporter = importlib.util.module_from_spec(me_spec)
sys.modules[me_spec.name] = metrics_exporter
assert me_spec.loader is not None
me_spec.loader.exec_module(metrics_exporter)

spec = importlib.util.spec_from_file_location(
    "menace_sandbox.self_improvement.orphan_handling",
    resolve_path("self_improvement/orphan_handling.py"),
)
orphan_handling = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = orphan_handling
assert spec.loader is not None
spec.loader.exec_module(orphan_handling)



def _reset_metrics() -> None:
    orphan_handling.orphan_integration_success_total.set(0)
    orphan_handling.orphan_integration_failure_total.set(0)


def test_integrate_orphans_success(monkeypatch):
    _reset_metrics()

    def dummy(*args, **kwargs):
        return ["mod_a"]

    monkeypatch.setattr(orphan_handling, "_load_orphan_module", lambda attr: dummy)
    monkeypatch.setattr(
        orphan_handling, "_call_with_retries", lambda func, *a, **kw: func(*a, **kw)
    )

    result = orphan_handling.integrate_orphans()
    assert result == ["mod_a"]
    assert orphan_handling.orphan_integration_success_total._value.get() == 1.0
    assert orphan_handling.orphan_integration_failure_total._value.get() == 0.0


def test_post_round_orphan_scan_failure(monkeypatch):
    _reset_metrics()

    def dummy(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orphan_handling, "_load_orphan_module", lambda attr: dummy)
    monkeypatch.setattr(
        orphan_handling, "_call_with_retries", lambda func, *a, **kw: func(*a, **kw)
    )

    with pytest.raises(RuntimeError):
        orphan_handling.post_round_orphan_scan()

    assert orphan_handling.orphan_integration_success_total._value.get() == 0.0
    assert orphan_handling.orphan_integration_failure_total._value.get() == 1.0

