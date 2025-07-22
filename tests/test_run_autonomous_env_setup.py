import importlib.util
import sys
from pathlib import Path
import pytest

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
    import types

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

        def load_history(self, p):
            pass

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda args, **k: None
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)


def test_files_created(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setattr(mod, "generate_presets", lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}])
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")

    mod.main([])

    assert Path(".env").exists()
    assert (tmp_path / "sandbox_data" / "presets.json").exists()


def test_cli_overrides_env(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    captured = {}

    def capture(*a, **k):
        captured["roi"] = a[2]
        return set(), None

    monkeypatch.setattr(mod.sandbox_runner.cli, "_diminishing_modules", capture)
    monkeypatch.setenv("ROI_THRESHOLD", "1.5")

    mod.main(["--roi-threshold", "2.5"])

    assert captured.get("roi") == 2.5


def test_get_env_override(monkeypatch):
    setup_stubs(monkeypatch)
    mod = load_module()
    monkeypatch.setenv("TEST_FLOAT", "1.25")
    monkeypatch.setenv("TEST_INT", "7")

    assert mod._get_env_override("TEST_FLOAT", None) == 1.25
    assert mod._get_env_override("TEST_INT", None) == 7
    assert mod._get_env_override("TEST_FLOAT", 3.5) == 3.5


def test_main_exits_on_failed_install(monkeypatch, tmp_path):
    """Verify main exits when dependency installation fails."""
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")
    monkeypatch.setattr(mod, "_check_dependencies", lambda: False)
    with pytest.raises(SystemExit) as exc:
        mod.main([])
    assert exc.value.code != 0


def test_invalid_preset_file_exits(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "presets.json").write_text('[{"CPU_LIMIT": "foo"}]')
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setenv("VISUAL_AGENT_AUTOSTART", "0")

    with pytest.raises(SystemExit):
        mod.main([
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(data_dir),
        ])

