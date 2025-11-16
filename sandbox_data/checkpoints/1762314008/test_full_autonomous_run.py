import argparse
import os
import sys
import types
from pathlib import Path

from dynamic_path_router import resolve_path

os.environ["MENACE_LIGHT_IMPORTS"] = "1"

scipy_stub = types.ModuleType("scipy")
stats_stub = types.SimpleNamespace(pearsonr=lambda *a, **k: (0.0, 0.0), t=lambda *a, **k: 0.0)
scipy_stub.stats = stats_stub
scipy_stub.isscalar = lambda x: isinstance(x, (int, float))
scipy_stub.bool_ = bool
sys.modules.setdefault("scipy", scipy_stub)
sys.modules.setdefault("scipy.stats", stats_stub)

import sandbox_runner.cli as cli
import sandbox_recovery_manager as srm


def test_full_autonomous_run_with_recovery(monkeypatch, tmp_path):
    # stub heavy dependencies
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    import sandbox_runner.environment as env
    monkeypatch.setattr(
        env.shutil,
        "which",
        lambda name: None if "qemu" in name else f"/usr/bin/{name}",
        raising=False,
    )

    class DummyDash:
        def __init__(self, *a, **k):
            pass
        def run(self, host="0.0.0.0", port=0):
            pass

    monkeypatch.setattr(cli, "MetricsDashboard", DummyDash)

    dummy_preset = {"env": "dev"}
    monkeypatch.setattr(cli, "generate_presets", lambda n=None: [dummy_preset])

    calls = []

    def failing_main(preset, args):
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("boom")
        Path(args.sandbox_data_dir).mkdir(parents=True, exist_ok=True)
        roi_file = Path(args.sandbox_data_dir) / "roi_history.json"
        roi_file.write_text("[0.1]")
        tracker = types.SimpleNamespace(
            roi_history=[0.1],
            module_deltas={},
            metrics_history={},
            diminishing=lambda: 0.0,
            rankings=lambda: [("m", 0.1)],
        )
        return tracker

    recovery = srm.SandboxRecoveryManager(failing_main, retry_delay=0)
    monkeypatch.setattr(srm.time, "sleep", lambda s: None)

    def fake_capture(preset, args):
        return recovery.run(preset, args)

    monkeypatch.setattr(cli, "_capture_run", fake_capture)

    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(
        sandbox_data_dir=str(resolve_path("sandbox_data")),
        preset_count=1,
        max_iterations=1,
        dashboard_port=None,
        roi_cycles=1,
        synergy_cycles=1,
    )

    cli.full_autonomous_run(args)

    roi_file = resolve_path("sandbox_data") / "roi_history.json"
    assert roi_file.exists()
    assert calls == ["call", "call"]
