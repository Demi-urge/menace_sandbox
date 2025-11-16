import importlib.util
import sys
import types
import json
from pathlib import Path

from dynamic_path_router import resolve_path


def load_module():
    path = resolve_path("run_autonomous.py")  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_stubs(monkeypatch):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1"}]

    tracker_mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": [0.05]}
            self.roi_history = [0.1]

        def save_history(self, path):
            data = {
                "roi_history": self.roi_history,
                "module_deltas": self.module_deltas,
                "metrics_history": self.metrics_history,
            }
            Path(path).write_text(json.dumps(data))

        def load_history(self, path):
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args: sr_stub._sandbox_main({}, args)
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})

    def fake_sandbox_main(preset, args):
        tracker = DummyTracker()
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        tracker.save_history(str(roi_file))
        return tracker

    sr_stub._sandbox_main = fake_sandbox_main
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))

    return sr_stub, fake_sandbox_main

def test_autonomous_full_run(monkeypatch, tmp_path):
    sr_stub, orig_main = setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    class DummyProc:
        def __init__(self):
            self.terminated = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def wait(self, timeout=None):
            return 0
        def kill(self):
            self.terminated = True

    popen_calls = []
    req_mod = types.ModuleType("requests")
    req_mod.get = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail"))
    monkeypatch.setitem(sys.modules, "requests", req_mod)
    monkeypatch.setattr(mod.subprocess, "Popen", lambda *a, **k: popen_calls.append(a) or DummyProc())
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    roi_file = tmp_path / "roi_history.json"
    synergy_file = tmp_path / "synergy_history.json"
    assert roi_file.exists()
    assert synergy_file.exists()
    roi_data = json.loads(roi_file.read_text())
    syn_data = json.loads(synergy_file.read_text())
    assert roi_data.get("roi_history")
    assert isinstance(syn_data, list) and syn_data

    assert popen_calls
    assert sr_stub._sandbox_main is orig_main
