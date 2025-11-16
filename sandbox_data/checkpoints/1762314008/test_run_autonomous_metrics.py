import importlib.util
import json
import sys
import types
from pathlib import Path

import metrics_exporter


def _load_run_autonomous(monkeypatch, tmp_path: Path):
    # Stub external modules used by run_autonomous
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r

    env_mod = types.ModuleType("menace.environment_generator")
    env_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]

    stats_mod = types.ModuleType("scipy.stats")
    stats_mod.t = types.SimpleNamespace(cdf=lambda *a, **k: 0.0)
    monkeypatch.setitem(sys.modules, "scipy", types.ModuleType("scipy"))
    monkeypatch.setitem(sys.modules, "scipy.stats", stats_mod)

    class DummyGauge:
        def __init__(self, *a, **k):
            self.value = 0.0

        def labels(self, *a, **k):
            return self

        def set(self, v):
            self.value = float(v)

        def inc(self, a=1.0):
            self.value += a

        def dec(self, a=1.0):
            self.value -= a

        def get(self):
            return self.value

    monkeypatch.setattr(metrics_exporter, "Gauge", DummyGauge, raising=False)
    metrics_exporter.roi_forecast_gauge = DummyGauge()
    metrics_exporter.synergy_forecast_gauge = DummyGauge()
    metrics_exporter.roi_threshold_gauge = DummyGauge()
    metrics_exporter.synergy_threshold_gauge = DummyGauge()
    metrics_exporter.synergy_adaptation_actions_total = DummyGauge()

    pyd = types.ModuleType("pydantic")

    class _Base:
        @classmethod
        def parse_obj(cls, obj):
            inst = cls()
            for k, v in obj.items():
                setattr(inst, k, v)
            return inst

        def dict(self, *a, **k):
            return self.__dict__.copy()

    pyd.BaseModel = _Base
    class _Root:
        @classmethod
        def __class_getitem__(cls, item):
            return cls
    pyd.RootModel = _Root
    pyd.ValidationError = type("ValidationError", (Exception,), {})
    pyd.validator = lambda *a, **k: (lambda f: f)
    pyd.Field = lambda default=None, **k: default
    pyd.BaseSettings = object

    ps = types.ModuleType("pydantic_settings")
    ps.BaseSettings = object
    ps.SettingsConfigDict = dict

    class DummyTracker:
        def __init__(self, *a, **k):
            self.roi_history = []
            self.module_deltas = {}
            self.metrics_history = {"synergy_roi": []}

        def save_history(self, path: str) -> None:
            data = {
                "roi_history": self.roi_history,
                "module_deltas": self.module_deltas,
                "metrics_history": self.metrics_history,
            }
            Path(path).write_text(json.dumps(data))

        def load_history(self, path: str) -> None:
            if Path(path).exists():
                data = json.loads(Path(path).read_text())
            else:
                data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {"synergy_roi": []})

        def diminishing(self) -> float:
            return 0.0

        def forecast(self):
            val = float(self.roi_history[-1]) if self.roi_history else 0.0
            return val, (val - 0.1, val + 0.1)

    tracker_mod = types.ModuleType("menace.roi_tracker")
    tracker_mod.ROITracker = DummyTracker

    synergy_exp = types.ModuleType("menace.synergy_exporter")
    synergy_exp.SynergyExporter = object

    shd_mod = types.ModuleType("menace.synergy_history_db")
    shd_mod.migrate_json_to_db = lambda *a, **k: None
    shd_mod.insert_entry = lambda *a, **k: None
    shd_mod.connect_locked = lambda *a, **k: None

    sym_mon = types.ModuleType("synergy_monitor")
    sym_mon.ExporterMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)
    sym_mon.AutoTrainerMonitor = lambda *a, **k: types.SimpleNamespace(start=lambda: None, stop=lambda: None, restart_count=0)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")

    run_idx = {"n": 0}

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        data_dir = Path(args.sandbox_data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        file = data_dir / "roi_history.json"
        if file.exists():
            data = json.loads(file.read_text())
        else:
            data = {"roi_history": [], "module_deltas": {}, "metrics_history": {"synergy_roi": []}}
        idx = run_idx["n"]
        data["roi_history"].append(0.1 + 0.05 * idx)
        data["metrics_history"]["synergy_roi"].append(0.05 / (idx + 1))
        file.write_text(json.dumps(data))
        run_idx["n"] += 1

    cli_stub.full_autonomous_run = fake_run
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: ((sum(seq) / len(seq)) if seq else 0.0, 0.0)
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.1
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, 1.0)

    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    sr_stub.__path__ = ["sandbox_runner"]

    class DummySettings:
        def __init__(self) -> None:
            self.sandbox_data_dir = str(tmp_path)
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

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", env_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)
    monkeypatch.setitem(sys.modules, "menace.synergy_exporter", synergy_exp)
    monkeypatch.setitem(sys.modules, "menace.synergy_history_db", shd_mod)
    monkeypatch.setitem(sys.modules, "pydantic", pyd)
    monkeypatch.setitem(sys.modules, "pydantic_settings", ps)
    monkeypatch.setitem(sys.modules, "synergy_monitor", sym_mon)
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    class DummyRecovery:
        def __init__(self, func):
            self.sandbox_main = func

        def run(self, preset, args):
            return self.sandbox_main(preset, args)

    monkeypatch.setitem(sys.modules, "sandbox_recovery_manager", types.SimpleNamespace(SandboxRecoveryManager=DummyRecovery))
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))
    class DummyLock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setitem(
        sys.modules,
        "filelock",
        types.SimpleNamespace(FileLock=lambda *a, **k: DummyLock(), Timeout=RuntimeError),
    )
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", types.SimpleNamespace(ensure_env=lambda p: None))

    monkeypatch.setattr(metrics_exporter, "start_metrics_server", lambda *a, **k: None)

    path = Path(__file__).resolve().parents[2] / "run_autonomous.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    import shutil
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)

    monkeypatch.setattr(mod, "ROITracker", DummyTracker)
    monkeypatch.setattr(mod, "SandboxSettings", DummySettings)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    return mod, run_idx


def test_run_autonomous_metrics(monkeypatch, tmp_path: Path) -> None:
    mod, _ = _load_run_autonomous(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "2",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--no-save-synergy-history",
        "--include-orphans",
        "--discover-orphans",
        "--no-discover-isolated",
    ])

    assert metrics_exporter.roi_forecast_gauge.labels().get() is not None
    assert metrics_exporter.synergy_forecast_gauge.labels().get() is not None
    assert metrics_exporter.synergy_threshold_gauge.labels().get() is not None

    log_file = tmp_path / "threshold_log.jsonl"
    assert log_file.exists()
    entries = [json.loads(l) for l in log_file.read_text().splitlines()]
    assert len(entries) == 2
    assert all("converged" in e for e in entries)
