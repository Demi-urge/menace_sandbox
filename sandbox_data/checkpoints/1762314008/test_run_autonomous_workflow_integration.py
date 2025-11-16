import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod


def setup_stubs(monkeypatch, tmp_path):
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]

    tracker_mod = types.ModuleType("menace.roi_tracker")

    class DummyTracker:
        def __init__(self, *a, **k):
            self.module_deltas = {}
            self.metrics_history = {}
            self.roi_history = []

        def load_history(self, path):
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args, **k: sr_stub._sandbox_main({}, args)
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})

    call_state = {"count": 0}

    def fake_sandbox_main(preset, args):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("boom")
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        hist_file = data_dir / "roi_history.json"
        if hist_file.exists():
            data = json.loads(hist_file.read_text())
        else:
            data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
        run_num = len(data["roi_history"]) + 1
        data["roi_history"].append(0.1 * run_num)
        data["metrics_history"]["synergy_roi"].append(0.05 * run_num)
        hist_file.write_text(json.dumps(data))
        tracker = DummyTracker()
        tracker.roi_history = data["roi_history"]
        tracker.metrics_history = data["metrics_history"]
        return tracker

    sr_stub._sandbox_main = fake_sandbox_main
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))

    if "filelock" not in sys.modules:
        filelock_mod = types.ModuleType("filelock")
        class DummyLock:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
        filelock_mod.FileLock = DummyLock
        filelock_mod.Timeout = RuntimeError
        monkeypatch.setitem(sys.modules, "filelock", filelock_mod)

    return call_state


def run_single(monkeypatch, tmp_path):
    call_state = setup_stubs(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    class DummySettings:
        def __init__(self):
            self.sandbox_data_dir = str(tmp_path)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = True
            self.roi_cycles = None
            self.synergy_cycles = None
            self.roi_threshold = None
            self.synergy_threshold = None
            self.roi_confidence = None
            self.synergy_confidence = None
            self.synergy_threshold_window = None
            self.synergy_threshold_weight = None
            self.synergy_ma_window = None
            self.synergy_stationarity_confidence = None
            self.synergy_std_threshold = None
            self.synergy_variance_confidence = None
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--no-recursive-orphans",
        "--no-recursive-isolated",
        "--include-orphans",
        "--discover-orphans",
        "--no-discover-isolated",
    ])

    return call_state["count"]


def test_workflow_persistence(monkeypatch, tmp_path):
    count1 = run_single(monkeypatch, tmp_path)
    roi_file = tmp_path / "roi_history.json"
    synergy_file = tmp_path / "synergy_history.json"
    assert count1 == 2
    assert roi_file.exists() and synergy_file.exists()
    assert json.loads(roi_file.read_text()).get("roi_history") == [0.1]
    assert len(json.loads(synergy_file.read_text())) == 1

    count2 = run_single(monkeypatch, tmp_path)
    assert count2 == 2
    roi_data = json.loads(roi_file.read_text())
    synergy_data = json.loads(synergy_file.read_text())
    assert roi_data.get("roi_history") == [0.1, 0.2]
    assert len(synergy_data) == 2
    assert synergy_data[1]["synergy_roi"] == 0.1

