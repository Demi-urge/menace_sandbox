from __future__ import annotations

import importlib
import sys
import types


def test_sandbox_runner_import_with_dict_settings(monkeypatch):
    settings_mod = types.ModuleType("sandbox_settings")
    settings_mod.SandboxSettings = lambda: {}
    monkeypatch.setitem(sys.modules, "sandbox_settings", settings_mod)

    rt_mod = types.ModuleType("sandbox_runner.resource_tuner")
    rt_mod.ResourceTuner = object
    wsr_mod = types.ModuleType("sandbox_runner.workflow_sandbox_runner")
    wsr_mod.WorkflowSandboxRunner = object
    mp_mod = types.ModuleType("sandbox_runner.metrics_plugins")
    mp_mod.discover_metrics_plugins = lambda *a, **k: []
    mp_mod.load_metrics_plugins = lambda *a, **k: []
    mp_mod.collect_plugin_metrics = lambda *a, **k: {}
    sp_mod = types.ModuleType("sandbox_runner.stub_providers")
    sp_mod.discover_stub_providers = lambda *a, **k: []
    sp_mod.load_stub_providers = lambda *a, **k: []
    od_mod = types.ModuleType("sandbox_runner.orphan_discovery")
    od_mod.discover_orphan_modules = lambda *a, **k: []
    od_mod.discover_recursive_orphans = lambda *a, **k: []
    oi_mod = types.ModuleType("sandbox_runner.orphan_integration")
    oi_mod.post_round_orphan_scan = lambda *a, **k: None
    oi_mod.integrate_and_graph_orphans = lambda *a, **k: {}
    oi_mod.integrate_orphans = lambda *a, **k: []

    monkeypatch.setitem(sys.modules, "sandbox_runner.resource_tuner", rt_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.workflow_sandbox_runner", wsr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_plugins", mp_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.stub_providers", sp_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_discovery", od_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.orphan_integration", oi_mod)

    sys.modules.pop("sandbox_runner", None)
    module = importlib.import_module("sandbox_runner")

    assert module._LIGHT_IMPORTS is True
