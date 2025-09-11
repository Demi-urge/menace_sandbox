import importlib
import importlib.util
import json
import socket
import time
import shutil
import sys
import threading
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    import socket

    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def load_module():
    path = ROOT / "run_autonomous.py"  # path-ignore
    sys.modules.pop("menace", None)
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    return mod, spec


def setup_stubs(monkeypatch, tmp_path):
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

        def save_history(self, path: str) -> None:
            Path(path).write_text(
                json.dumps(
                    {
                        "roi_history": self.roi_history,
                        "module_deltas": self.module_deltas,
                        "metrics_history": self.metrics_history,
                    }
                )
            )

        def load_history(self, path: str) -> None:
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self) -> float:
            return 0.0

    tracker_mod.ROITracker = DummyTracker

    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")
    cli_stub.full_autonomous_run = lambda *a, **k: None
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    sr_stub._sandbox_main = lambda *a, **k: DummyTracker()
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))

    if "filelock" not in sys.modules:
        fl = types.ModuleType("filelock")

        class DummyLock:
            def __init__(self, *a, **k):
                pass

            def acquire(self, timeout=0):
                pass

            def release(self):
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

    monkeypatch.setenv("SYNERGY_EXPORTER_CHECK_INTERVAL", "0.01")

    return sr_stub


def test_exporter_auto_restart(monkeypatch, tmp_path: Path) -> None:
    sr_stub = setup_stubs(monkeypatch, tmp_path)

    se_mod = importlib.import_module("menace.synergy_exporter")
    captured = {"starts": 0}

    class CrashExporter(se_mod.SynergyExporter):
        def start(self) -> None:  # type: ignore[override]
            captured["starts"] += 1
            self._thread = threading.Thread(target=lambda: None, daemon=True)
            self._thread.start()
            self._thread.join(0.01)

    monkeypatch.setattr(se_mod, "SynergyExporter", CrashExporter)

    mod, spec = load_module()
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SynergyExporter", CrashExporter)
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
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "2",
            "--preset-count",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    assert captured["starts"] >= 2

    meta_log = tmp_path / "sandbox_meta.log"
    assert meta_log.exists()
    entries = [
        json.loads(l.split(" ", 1)[1]) for l in meta_log.read_text().splitlines()
    ]
    restarts = [
        e.get("restart_count")
        for e in entries
        if e.get("event") == "exporter_restarted"
    ]
    assert restarts and restarts[-1] == len(restarts)
    assert isinstance(se_mod.exporter_uptime.labels().get(), float)
    assert isinstance(se_mod.exporter_failures.labels().get(), float)


def test_exporter_busy_port(monkeypatch, tmp_path: Path) -> None:
    sr_stub = setup_stubs(monkeypatch, tmp_path)

    se_mod = importlib.import_module("menace.synergy_exporter")
    captured = {"ports": []}

    class RecordingExporter(se_mod.SynergyExporter):
        def start(self) -> None:  # type: ignore[override]
            captured["ports"].append(self.port)
            self._thread = threading.Thread(target=lambda: None, daemon=True)
            self._thread.start()
            self._thread.join(0.01)

    monkeypatch.setattr(se_mod, "SynergyExporter", RecordingExporter)

    mod, spec = load_module()
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SynergyExporter", RecordingExporter)
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
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    port = _free_port()
    busy = socket.socket()
    busy.bind(("", port))

    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    try:
        mod.main(
            [
                "--max-iterations",
                "1",
                "--runs",
                "1",
                "--preset-count",
                "1",
                "--sandbox-data-dir",
                str(tmp_path),
            ]
        )
    finally:
        busy.close()

    assert captured["ports"]
    assert captured["ports"][0] != port
    assert isinstance(se_mod.exporter_uptime.labels().get(), float)
    assert isinstance(se_mod.exporter_failures.labels().get(), float)


def test_exporter_stale_health(monkeypatch, tmp_path: Path) -> None:
    sr_stub = setup_stubs(monkeypatch, tmp_path)

    se_mod = importlib.import_module("menace.synergy_exporter")
    captured = {"starts": 0}

    class StaleExporter(se_mod.SynergyExporter):
        def start(self) -> None:  # type: ignore[override]
            captured["starts"] += 1
            self._thread = threading.Thread(target=lambda: time.sleep(0.5), daemon=True)
            self._thread.start()
            self._start_health_server()
            self.last_update = time.time() - 999

    monkeypatch.setattr(se_mod, "SynergyExporter", StaleExporter)

    mod, spec = load_module()
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SynergyExporter", StaleExporter)
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
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--preset-count",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    assert captured["starts"] >= 2
    meta_log = tmp_path / "sandbox_meta.log"
    assert meta_log.exists()
    entries = [
        json.loads(l.split(" ", 1)[1]) for l in meta_log.read_text().splitlines()
    ]
    restarts = [
        e.get("restart_count")
        for e in entries
        if e.get("event") == "exporter_restarted"
    ]
    assert restarts and restarts[-1] == len(restarts)


def test_exporter_cleanup(monkeypatch, tmp_path: Path) -> None:
    sr_stub = setup_stubs(monkeypatch, tmp_path)

    se_mod = importlib.import_module("menace.synergy_exporter")

    class RecordingExporter(se_mod.SynergyExporter):
        instances: list["RecordingExporter"] = []

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            RecordingExporter.instances.append(self)
            self.stopped = False

        def start(self) -> None:  # type: ignore[override]
            self._thread = threading.Thread(
                target=lambda: time.sleep(0.05), daemon=True
            )
            self._thread.start()

        def stop(self) -> None:  # type: ignore[override]
            self.stopped = True
            super().stop()

    monkeypatch.setattr(se_mod, "SynergyExporter", RecordingExporter)

    mod, spec = load_module()
    monkeypatch.setitem(sys.modules, "run_autonomous", mod)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(shutil, "which", lambda *_a, **_k: "/usr/bin/true")
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "SynergyExporter", RecordingExporter)
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
    monkeypatch.setattr(mod, "validate_presets", lambda p: list(p))

    monkeypatch.chdir(tmp_path)

    port = _free_port()
    monkeypatch.setenv("EXPORT_SYNERGY_METRICS", "1")
    monkeypatch.setenv("SYNERGY_METRICS_PORT", str(port))

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--preset-count",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    inst = RecordingExporter.instances[-1]
    assert inst.stopped is True
    assert inst._thread is not None and not inst._thread.is_alive()
    assert isinstance(se_mod.exporter_uptime.labels().get(), float)
    assert isinstance(se_mod.exporter_failures.labels().get(), float)
