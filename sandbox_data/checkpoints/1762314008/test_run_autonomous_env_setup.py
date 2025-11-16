import importlib.util
import sys
from pathlib import Path
import json
import shutil
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[1]


def load_module(monkeypatch=None):
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    if monkeypatch is not None:
        monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
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
            self.module_entropy_deltas = {}
            self.roi_history = []

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
    monkeypatch.setenv("SAVE_SYNERGY_HISTORY", "0")
    monkeypatch.setenv("ENABLE_RELEVANCY_RADAR", "0")


def test_files_created(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setattr(mod, "generate_presets", lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}])

    mod.main([])

    assert Path(".env").exists()
    assert (tmp_path / "sandbox_data" / "presets.json").exists()


def test_cli_overrides_env(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "full_autonomous_run", lambda args, **k: None)
    monkeypatch.setenv("ENABLE_RELEVANCY_RADAR", "0")
    captured = {}

    def capture(*a, **k):
        captured["roi"] = a[2]
        captured["entropy_threshold"] = k.get("entropy_threshold")
        captured["entropy_consecutive"] = k.get("entropy_consecutive")
        return set(), None

    monkeypatch.setattr(mod.sandbox_runner.cli, "_diminishing_modules", capture)
    monkeypatch.setenv("ROI_THRESHOLD", "1.5")
    monkeypatch.setenv("ENTROPY_PLATEAU_THRESHOLD", "0.2")
    monkeypatch.setenv("ENTROPY_PLATEAU_CONSECUTIVE", "5")

    mod.main([
        "--roi-threshold",
        "2.5",
        "--entropy-plateau-threshold",
        "0.3",
        "--entropy-plateau-consecutive",
        "4",
    ])

    assert captured.get("roi") == 2.5
    assert captured.get("entropy_threshold") == 0.3
    assert captured.get("entropy_consecutive") == 4


def test_get_env_override(monkeypatch):
    setup_stubs(monkeypatch)
    mod = load_module(monkeypatch)
    monkeypatch.setenv("TEST_FLOAT", "1.25")
    monkeypatch.setenv("TEST_INT", "7")
    import types
    settings = types.SimpleNamespace(test_float="1.25", test_int="7")

    assert mod._get_env_override("TEST_FLOAT", None, settings) == 1.25
    assert mod._get_env_override("TEST_INT", None, settings) == 7
    assert mod._get_env_override("TEST_FLOAT", 3.5, settings) == 3.5


def test_main_exits_on_failed_install(monkeypatch, tmp_path):
    """Verify main exits when dependency installation fails."""
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: False)
    mod.main([])


def test_invalid_preset_file_exits(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "presets.json").write_text('[{"CPU_LIMIT": "foo"}]')
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    with pytest.raises((SystemExit, ValidationError)):
        mod.main([
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(data_dir),
        ])


def test_corrupt_synergy_history_load(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "synergy_history.json").write_text("{bad json")
    mod = load_module(monkeypatch)
    hist, ma = mod.load_previous_synergy(data_dir)
    assert hist == []
    assert ma == []


def test_main_ignores_corrupt_synergy(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "synergy_history.json").write_text("not json")
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(data_dir),
    ])


def test_main_recovers_corrupt_presets(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    (data_dir / "presets.json").write_text("{bad json")
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(data_dir),
    ])

    text = (data_dir / "presets.json").read_text()
    assert json.loads(text)


def test_main_ignores_corrupt_synergy_db(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    import sqlite3
    conn = sqlite3.connect(str(data_dir / "synergy_history.db"))
    conn.execute(
        "CREATE TABLE synergy_history (id INTEGER PRIMARY KEY AUTOINCREMENT, entry TEXT NOT NULL)"
    )
    conn.execute("INSERT INTO synergy_history(entry) VALUES ('not json')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    mod = load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(data_dir),
    ])

