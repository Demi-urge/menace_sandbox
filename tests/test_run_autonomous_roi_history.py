import importlib.util
import sys
import types
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "run_autonomous.py"
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
            self.metrics_history = {}
            self.roi_history = []

        def load_history(self, path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self.roi_history = data.get("roi_history", [])

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")

    def fake_run(args):
        data_dir = Path(args.sandbox_data_dir or "sandbox_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        with open(roi_file, "w", encoding="utf-8") as fh:
            json.dump({"roi_history": [0.1], "module_deltas": {}, "metrics_history": {}}, fh)

    cli_stub.full_autonomous_run = fake_run
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))


def test_run_autonomous_writes_history(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")

    mod.main(["--max-iterations", "1", "--runs", "1", "--sandbox-data-dir", str(tmp_path)])

    history_file = tmp_path / "roi_history.json"
    assert history_file.exists()
    data = json.loads(history_file.read_text())
    assert data.get("roi_history") == [0.1]
