import importlib.util
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
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
            self.metrics_history = {"synergy_roi": [0.05]}
            self.roi_history = [0.1]

        def save_history(self, path: str) -> None:
            data = {
                "roi_history": self.roi_history,
                "module_deltas": self.module_deltas,
                "metrics_history": self.metrics_history,
            }
            Path(path).write_text(json.dumps(data))

        def load_history(self, path: str) -> None:
            try:
                data = json.loads(Path(path).read_text())
            except Exception:
                data = {}
            self.roi_history = data.get("roi_history", [])
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self) -> float:
            return 0.0

    tracker_mod.ROITracker = DummyTracker

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    bootstrap_stub = types.ModuleType("sandbox_runner.bootstrap")
    db_router_stub = types.ModuleType("db_router")

    bootstrap_stub.bootstrap_environment = lambda s, *a, **k: s
    bootstrap_stub._verify_required_dependencies = lambda *a, **k: None
    db_router_stub.init_db_router = lambda *a, **k: object()

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        hist = {
            "roi_history": [0.1],
            "module_deltas": {},
            "metrics_history": {"synergy_roi": [0.05]},
        }
        (data_dir / "roi_history.json").write_text(json.dumps(hist))
        return DummyTracker()

    cli_stub.full_autonomous_run = fake_run
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})

    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)
    monkeypatch.setitem(sys.modules, "db_router", db_router_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.bootstrap", bootstrap_stub)
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


def test_full_cycle(monkeypatch, tmp_path):
    setup_stubs(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")

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
    assert roi_file.exists() and synergy_file.exists()
    roi_data = json.loads(roi_file.read_text())
    syn_data = json.loads(synergy_file.read_text())
    assert roi_data.get("roi_history") == [0.1]
    assert syn_data == [{"synergy_roi": 0.05}]
