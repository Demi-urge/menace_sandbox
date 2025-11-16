import importlib
import importlib.util
import json
import sqlite3
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_module(monkeypatch, tmp_path: Path):
    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(importlib, "import_module", importlib.import_module)
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda *a, **k: None
    sr_stub.cli = cli_stub
    sr_stub._sandbox_main = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")
        class DummyLock:
            def __init__(self, *a, **k):
                pass
            def acquire(self, timeout=0):
                pass
            def release(self):
                pass
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                pass
        fl.FileLock = DummyLock
        fl.Timeout = RuntimeError
        monkeypatch.setitem(sys.modules, "filelock", fl)
    if "dotenv" not in sys.modules:
        dmod = types.ModuleType("dotenv")
        dmod.dotenv_values = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, "dotenv", dmod)
    if "pydantic" not in sys.modules:
        pyd = types.ModuleType("pydantic")
        pyd.BaseModel = object
        pyd.ValidationError = type("ValidationError", (Exception,), {})
        pyd.validator = lambda *a, **k: (lambda f: f)
        pyd.BaseSettings = object
        pyd.Field = lambda default=None, **k: default
        monkeypatch.setitem(sys.modules, "pydantic", pyd)
    if "pydantic_settings" not in sys.modules:
        ps = types.ModuleType("pydantic_settings")
        ps.BaseSettings = object
        ps.SettingsConfigDict = dict
        monkeypatch.setitem(sys.modules, "pydantic_settings", ps)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "ensure_env", lambda *a, **k: None)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    class DummySettings:
        def __init__(self) -> None:
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
    return mod


def test_json_migrated_to_db(monkeypatch, tmp_path: Path) -> None:
    data = [{"synergy_roi": 0.1}, {"synergy_roi": 0.2}]
    json_file = tmp_path / "synergy_history.json"
    json_file.write_text(json.dumps(data))

    mod = _load_module(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    mod.main(["--check-settings", "--sandbox-data-dir", str(tmp_path)])

    db_file = tmp_path / "synergy_history.db"
    assert db_file.exists()
    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT entry FROM synergy_history ORDER BY id").fetchall()
    conn.close()
    loaded = [json.loads(r[0]) for r in rows]
    assert loaded == data
