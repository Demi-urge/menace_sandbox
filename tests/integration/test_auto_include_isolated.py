import importlib.util
import json
import shutil
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_run_autonomous(monkeypatch, data_dir, log):
    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda p: None
    env_gen = types.ModuleType("menace.environment_generator")
    env_gen.generate_presets = lambda n=None: [{}]
    env_gen.generate_presets_from_history = lambda *a, **k: [{}]
    env_gen.adapt_presets = types.SimpleNamespace(last_actions=[])
    startup = types.ModuleType("menace.startup_checks")
    startup.verify_project_dependencies = lambda: []
    dep_inst = types.ModuleType("menace.dependency_installer")
    dep_inst.install_packages = lambda *a, **k: None

    class DummyTracker:
        def __init__(self, *a, **k):
            self.roi_history = []
            self.module_deltas = {}
            self.metrics_history = {}

        def load_history(self, path):
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
                self.roi_history = data.get("roi_history", [])
                self.module_deltas = data.get("module_deltas", {})
                self.metrics_history = data.get("metrics_history", {})

        def save_history(self, path):
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                    }
                )
            )

        def diminishing(self):
            return 0.0

        def forecast(self):
            return 0.0, (0.0, 0.0)

    roi_mod = types.ModuleType("menace.roi_tracker")
    roi_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_gen)
    monkeypatch.setitem(sys.modules, "menace.startup_checks", startup)
    monkeypatch.setitem(sys.modules, "menace.dependency_installer", dep_inst)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", roi_mod)

    synergy_exp = types.ModuleType("menace.synergy_exporter")
    synergy_exp.SynergyExporter = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.synergy_exporter", synergy_exp)
    shd_mod = types.ModuleType("menace.synergy_history_db")
    shd_mod.migrate_json_to_db = lambda *a, **k: None
    shd_mod.insert_entry = lambda *a, **k: None
    shd_mod.connect_locked = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.synergy_history_db", shd_mod)

    sym_mon = types.ModuleType("synergy_monitor")
    sym_mon.ExporterMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)
    sym_mon.AutoTrainerMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)
    monkeypatch.setitem(sys.modules, "synergy_monitor", sym_mon)

    srm = types.ModuleType("sandbox_recovery_manager")
    srm.SandboxRecoveryManager = lambda func: types.SimpleNamespace(run=func, sandbox_main=func)
    monkeypatch.setitem(sys.modules, "sandbox_recovery_manager", srm)

    env_mod = types.ModuleType("sandbox_runner.environment")

    def fake_generate_workflows(mods, workflows_db="workflows.db"):
        log["workflows"] = list(mods)
        return [1]

    env_mod.generate_workflows_for_modules = fake_generate_workflows

    class DummyEngine:
        def __init__(self, *a, **k):
            self.module_clusters = {}

        def _refresh_module_map(self, modules):
            log["refresh"] = list(modules)
            data_dir.mkdir(parents=True, exist_ok=True)
            map_file = data_dir / "module_map.json"
            map_file.write_text(
                json.dumps({"modules": {Path(m).name: 1 for m in modules}, "groups": {"1": 1}})
            )
            env_mod.generate_workflows_for_modules(modules)

    class DummySelfTestService:
        def __init__(self, *a, integration_callback=None, **kw):
            self.integration_callback = integration_callback

        def run_once(self):
            data_dir.mkdir(parents=True, exist_ok=True)
            orphan_file = data_dir / "orphan_modules.json"
            mods = ["iso.py"]  # path-ignore
            orphan_file.write_text(json.dumps(mods))
            if self.integration_callback:
                self.integration_callback(mods)

    sr_mod = types.ModuleType("sandbox_runner")
    cli_mod = types.ModuleType("sandbox_runner.cli")
    cli_mod.full_autonomous_run = lambda args, **k: sr_mod._sandbox_main({}, args)
    cli_mod._diminishing_modules = lambda *a, **k: (set(), None)
    cli_mod._ema = lambda seq: (0.0, [])
    cli_mod._adaptive_threshold = lambda *a, **k: 0.0
    cli_mod._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_mod._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_mod.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})

    def fake_sandbox_main(preset, args):
        engine = DummyEngine()
        tester = DummySelfTestService(integration_callback=engine._refresh_module_map)
        tester.run_once()
        tracker = DummyTracker()
        tracker.roi_history.append(0.0)
        Path(args.sandbox_data_dir).mkdir(parents=True, exist_ok=True)
        tracker.save_history(str(Path(args.sandbox_data_dir) / "roi_history.json"))
        return tracker

    sr_mod._sandbox_main = fake_sandbox_main
    sr_mod.cli = cli_mod
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)

    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")
        class DummyLock:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc, tb):
                return False
        fl.FileLock = lambda *a, **k: DummyLock()
        fl.Timeout = RuntimeError
        monkeypatch.setitem(sys.modules, "filelock", fl)
    if "dotenv" not in sys.modules:
        dmod = types.ModuleType("dotenv")
        dmod.dotenv_values = lambda *a, **k: {}
        monkeypatch.setitem(sys.modules, "dotenv", dmod)
    if "pydantic" not in sys.modules:
        pyd = types.ModuleType("pydantic")
        pyd.BaseModel = object
        class _Root:
            @classmethod
            def __class_getitem__(cls, item):
                return cls
        pyd.RootModel = _Root
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

    path = ROOT / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setitem(sys.modules, "menace.audit_trail", types.SimpleNamespace(AuditTrail=lambda *a, **k: None))
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)

    class DummySettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(data_dir)
            self.sandbox_env_presets = None
            self.auto_dashboard_port = None
            self.save_synergy_history = False
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


def test_auto_include_isolated(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "dep.py").write_text("def x():\n    pass\n")  # path-ignore
    (repo / "iso.py").write_text("import dep\n")  # path-ignore
    data_dir = tmp_path / "data"
    log: dict = {}
    mod = _load_run_autonomous(monkeypatch, data_dir, log)

    monkeypatch.chdir(repo)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(data_dir),
        "--auto-include-isolated",
    ])

    orphans = json.loads((data_dir / "orphan_modules.json").read_text())
    assert "iso.py" in orphans  # path-ignore

    map_data = json.loads((data_dir / "module_map.json").read_text())
    assert "iso.py" in map_data["modules"]  # path-ignore

    assert log.get("workflows") == ["iso.py"]  # path-ignore
