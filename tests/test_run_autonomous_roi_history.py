import importlib.util
import sys
import types
import json
from pathlib import Path
import pytest
from dynamic_path_router import resolve_path
from tests.test_run_autonomous_env_vars import _load_module


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
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]
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
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self):
            return 0.0

    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sr_stub = types.ModuleType("sandbox_runner")
    cli_stub = types.ModuleType("sandbox_runner.cli")

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        data_dir = Path(resolve_path(args.sandbox_data_dir or "sandbox_data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        roi_file = data_dir / "roi_history.json"
        with open(roi_file, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "roi_history": [0.1],
                    "module_deltas": {},
                    "metrics_history": {"synergy_roi": [0.05]},
                },
                fh,
            )

    cli_stub.full_autonomous_run = fake_run
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, [])
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})

    def _adapt_conv(history, window, *, threshold=None, threshold_window=None, **k):
        if threshold is None:
            threshold_window = threshold_window or window
            threshold = cli_stub._adaptive_synergy_threshold(history, threshold_window)
        return cli_stub._synergy_converged(history, window, threshold, **k)

    cli_stub.adaptive_synergy_convergence = _adapt_conv
    sr_stub._sandbox_main = lambda p, a: None
    sr_stub.cli = cli_stub
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))


def _setup_metrics_stub(monkeypatch):
    stub = types.ModuleType("metrics_exporter")

    class DummyGauge:
        def __init__(self):
            self.value = None

        def set(self, v):
            self.value = v

    stub.start_metrics_server = lambda *a, **k: None
    stub.roi_threshold_gauge = DummyGauge()
    stub.synergy_threshold_gauge = DummyGauge()
    stub.synergy_forecast_gauge = DummyGauge()
    monkeypatch.setitem(sys.modules, "metrics_exporter", stub)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_exporter", stub)
    return stub


def test_run_autonomous_writes_history(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    mod.main(
        ["--max-iterations", "1", "--runs", "1", "--sandbox-data-dir", str(tmp_path)]
    )

    history_file = tmp_path / "roi_history.json"
    assert history_file.exists()
    data = json.loads(history_file.read_text())
    assert data.get("roi_history") == [0.1]


def test_synergy_history_reused(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)

    synergy_file = tmp_path / "synergy_history.json"
    synergy_file.write_text(json.dumps([{"synergy_roi": 0.1}]))
    captured = {}

    def capture(history, *a, **k):
        captured["len"] = len(history)
        return True, 0.0, {}

    monkeypatch.setattr(mod.sandbox_runner.cli, "_synergy_converged", capture)

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    new_hist = json.loads(synergy_file.read_text())
    assert len(new_hist) == 2
    assert captured.get("len") == 2


def test_adaptive_synergy_threshold_default(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    history = [{"synergy_roi": i * 0.1} for i in range(5)]
    (tmp_path / "synergy_history.json").write_text(json.dumps(history))

    captured = {}

    def fake_thr(hist, *a, **k):
        captured["len"] = len(hist)
        return 0.5

    def fake_conv(hist, cycles, threshold, **k):
        captured["cycles"] = cycles
        captured["thr"] = threshold
        return True, 0.0, {}

    monkeypatch.setattr(mod.sandbox_runner.cli, "_adaptive_synergy_threshold", fake_thr)
    monkeypatch.setattr(mod.sandbox_runner.cli, "_synergy_converged", fake_conv)

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    assert captured.get("len") == 6
    assert captured.get("thr") == 0.5
    assert captured.get("cycles") == 5


def test_previous_synergy_initialises_threshold(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    history = [{"synergy_roi": 0.1}, {"synergy_roi": 0.2}]
    (tmp_path / "synergy_history.json").write_text(json.dumps(history))

    lengths: list[int] = []

    def fake_thr(hist, *a, **k):
        lengths.append(len(hist))
        return 0.0

    monkeypatch.setattr(mod.sandbox_runner.cli, "_adaptive_synergy_threshold", fake_thr)
    monkeypatch.setattr(
        mod.sandbox_runner.cli,
        "_synergy_converged",
        lambda *a, **k: (True, 0.0, {}),
    )

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

    assert lengths and lengths[0] == len(history) + 1


def test_disable_synergy_history(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
            "--no-save-synergy-history",
        ]
    )

    assert not (tmp_path / "synergy_history.json").exists()


def test_env_disable_synergy_history(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    monkeypatch.setenv("SAVE_SYNERGY_HISTORY", "0")

    mod.main(
        [
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ]
    )

    assert not (tmp_path / "synergy_history.json").exists()


def test_invalid_synergy_history_exits(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    (tmp_path / "synergy_history.json").write_text('[{"synergy_roi": "bad"}]')

    with pytest.raises(SystemExit):
        mod.main([
            "--max-iterations",
            "1",
            "--runs",
            "1",
            "--sandbox-data-dir",
            str(tmp_path),
        ])


def test_auto_thresholds_uses_saved_synergy(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = load_module()
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)

    thresholds: list[float] = []
    hist_lengths: list[int] = []

    def adaptive(hist: list[dict[str, float]], window: int, **k) -> float:
        hist_lengths.append(len(hist))
        vals = [h.get("synergy_roi", 0.0) for h in hist[-window:]]
        return sum(vals) / len(vals) if vals else 0.0

    def conv(history, cycles, *, threshold=None, threshold_window=None, **k):
        if threshold_window is None:
            threshold_window = cycles
        thr = threshold if threshold is not None else adaptive(history, threshold_window)
        thresholds.append(thr)
        return True, 0.0, {}

    monkeypatch.setattr(mod.sandbox_runner.cli, "adaptive_synergy_convergence", conv)

    def make_run(val: float):
        def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
            data_dir = Path(resolve_path(args.sandbox_data_dir or "sandbox_data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            roi_file = data_dir / "roi_history.json"
            with open(roi_file, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "roi_history": [0.1],
                        "module_deltas": {},
                        "metrics_history": {"synergy_roi": [val]},
                    },
                    fh,
                )

        return fake_run

    monkeypatch.setattr(mod.sandbox_runner.cli, "full_autonomous_run", make_run(0.2))
    monkeypatch.setattr(mod, "full_autonomous_run", make_run(0.2))
    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--auto-thresholds",
    ])

    first_thr = thresholds[-1]

    monkeypatch.setattr(mod.sandbox_runner.cli, "full_autonomous_run", make_run(0.05))
    monkeypatch.setattr(mod, "full_autonomous_run", make_run(0.05))
    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--auto-thresholds",
    ])

    second_thr = thresholds[-1]

    assert hist_lengths[0] == 1
    assert hist_lengths[1] == 2
    assert second_thr < first_thr


def test_threshold_gauges_updated(monkeypatch, tmp_path):
    setup_stubs(monkeypatch)
    metrics_stub = _setup_metrics_stub(monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = _load_module(monkeypatch)
    monkeypatch.setattr(mod, "_check_dependencies", lambda *a, **k: True)
    monkeypatch.setattr(mod, "validate_presets", lambda p: p)
    cli_stub = mod.sandbox_runner.cli
    cli_stub._diminishing_modules = lambda *a, **k: (set(), None)
    cli_stub._ema = lambda seq: (0.0, 0.0)
    cli_stub._adaptive_threshold = lambda *a, **k: 0.0
    cli_stub._adaptive_synergy_threshold = lambda *a, **k: 0.0
    cli_stub._synergy_converged = lambda *a, **k: (True, 0.0, {})
    cli_stub.adaptive_synergy_convergence = lambda *a, **k: (True, 0.0, {})
    mod.ROITracker = lambda *a, **k: types.SimpleNamespace(
        module_deltas={},
        metrics_history={"synergy_roi": [0.05]},
        roi_history=[0.1],
        load_history=lambda p: None,
        diminishing=lambda: 0.0,
    )

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        data_dir = Path(resolve_path(args.sandbox_data_dir or "sandbox_data"))
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "roi_history.json", "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "roi_history": [0.1],
                    "module_deltas": {},
                    "metrics_history": {"synergy_roi": [0.05]},
                },
                fh,
            )

    monkeypatch.setattr(mod.sandbox_runner.cli, "full_autonomous_run", fake_run)
    monkeypatch.setattr(mod, "full_autonomous_run", fake_run)

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
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    mod.main([
        "--max-iterations",
        "1",
        "--runs",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
    ])

    assert metrics_stub.roi_threshold_gauge.value is not None
    assert metrics_stub.synergy_threshold_gauge.value is not None
    assert metrics_stub.synergy_forecast_gauge.value is not None
