import os
import sys
import types
import importlib
import json
from pathlib import Path

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _setup(monkeypatch, roi_seq, syn_seq):
    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "menace", pkg)

    auto_env = types.ModuleType("menace.auto_env_setup")
    auto_env.ensure_env = lambda p: None
    monkeypatch.setitem(sys.modules, "menace.auto_env_setup", auto_env)

    dep_inst = types.ModuleType("menace.dependency_installer")
    dep_inst.install_packages = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace.dependency_installer", dep_inst)
    sc_mod = types.ModuleType("menace.startup_checks")
    sc_mod.verify_project_dependencies = lambda: []
    sc_mod._parse_requirement = lambda r: r
    monkeypatch.setitem(sys.modules, "menace.startup_checks", sc_mod)

    eg_mod = types.ModuleType("menace.environment_generator")
    eg_mod.generate_presets = lambda n=None: [{"CPU_LIMIT": "1", "MEMORY_LIMIT": "1"}]
    monkeypatch.setitem(sys.modules, "menace.environment_generator", eg_mod)

    yaml_stub = types.ModuleType("yaml")
    yaml_stub.safe_load = lambda fh: {}
    monkeypatch.setitem(sys.modules, "yaml", yaml_stub)
    monkeypatch.setitem(sys.modules, "docker", types.ModuleType("docker"))

    dash_mod = types.ModuleType("menace.metrics_dashboard")
    dash_mod.MetricsDashboard = lambda *a, **k: types.SimpleNamespace(run=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "menace.metrics_dashboard", dash_mod)

    flask_mod = types.ModuleType("flask")
    flask_mod.Flask = lambda *a, **k: types.SimpleNamespace(add_url_rule=lambda *a, **k: None, run=lambda *a, **k: None)
    flask_mod.jsonify = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "flask", flask_mod)

    class DummyTracker:
        def __init__(self, roi=0.0, syn=0.0):
            self.roi_history = [roi]
            self.module_deltas = {"mod": [roi]}
            self.metrics_history = {"synergy_roi": [syn]}

        def save_history(self, path: str) -> None:
            data = {
                "roi_history": self.roi_history,
                "module_deltas": self.module_deltas,
                "metrics_history": self.metrics_history,
            }
            Path(path).write_text(json.dumps(data))

        def load_history(self, path: str) -> None:
            data = json.loads(Path(path).read_text())
            self.roi_history = data.get("roi_history", [])
            self.module_deltas = data.get("module_deltas", {})
            self.metrics_history = data.get("metrics_history", {})

        def diminishing(self) -> float:
            return 0.01

        def rankings(self):
            total = sum(self.module_deltas.get("mod", []))
            return [("mod", total, total)]

    tracker_mod = types.ModuleType("menace.roi_tracker")
    tracker_mod.ROITracker = DummyTracker
    monkeypatch.setitem(sys.modules, "menace.roi_tracker", tracker_mod)

    sys.modules.pop("sandbox_runner.cli", None)
    cli = importlib.import_module("sandbox_runner.cli")

    monkeypatch.setattr(cli, "_diminishing_modules", lambda *a, **k: ({"mod"}, {}))

    counter = {"idx": 0}

    def fake_run(args, *, synergy_history=None, synergy_ma_history=None):
        i = counter["idx"]
        roi = roi_seq[i] if i < len(roi_seq) else roi_seq[-1]
        syn = syn_seq[i] if i < len(syn_seq) else syn_seq[-1]
        counter["idx"] += 1
        data_dir = Path(args.sandbox_data_dir or "sandbox_data")
        data_dir.mkdir(parents=True, exist_ok=True)
        DummyTracker(roi, syn).save_history(str(data_dir / "roi_history.json"))

    monkeypatch.setattr(cli, "full_autonomous_run", fake_run)

    sr_mod = types.ModuleType("sandbox_runner")
    sr_mod._sandbox_main = lambda p, a: None
    sr_mod.cli = cli
    monkeypatch.setitem(sys.modules, "sandbox_runner", sr_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cli", cli)

    mod = _load_module()
    monkeypatch.setattr(mod, "full_autonomous_run", fake_run)
    monkeypatch.setattr(mod, "generate_presets", eg_mod.generate_presets)
    monkeypatch.setattr(mod.environment_generator, "generate_presets", eg_mod.generate_presets)
    monkeypatch.setattr(mod, "ROITracker", DummyTracker)

    return counter, mod


def _load_module():
    import importlib.util
    path = Path("run_autonomous.py")  # path-ignore
    if "run_autonomous" in sys.modules:
        return sys.modules["run_autonomous"]
    spec = importlib.util.spec_from_file_location("run_autonomous", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_autonomous"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_converges(monkeypatch, tmp_path):
    counter, mod = _setup(monkeypatch, [0.1, 0.1, 0.1], [0.1, 0.05, 0.02])
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    logs = []
    monkeypatch.setattr(mod.logger, "info", lambda m, *a, **k: logs.append(m))

    mod.main([
        "--max-iterations",
        "10",
        "--runs",
        "10",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--roi-cycles",
        "1",
        "--synergy-cycles",
        "3",
    ])

    assert counter["idx"] == 3
    assert any("convergence reached" in m for m in logs)


def test_no_premature_convergence(monkeypatch, tmp_path):
    counter, mod = _setup(monkeypatch, [0.1] * 6, [0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
    monkeypatch.setattr(mod, "_check_dependencies", lambda: True)
    logs = []
    monkeypatch.setattr(mod.logger, "info", lambda m, *a, **k: logs.append(m))

    mod.main([
        "--max-iterations",
        "6",
        "--runs",
        "6",
        "--preset-count",
        "1",
        "--sandbox-data-dir",
        str(tmp_path),
        "--roi-cycles",
        "1",
        "--synergy-cycles",
        "3",
    ])

    assert counter["idx"] == 6
    assert not any("convergence reached" in m for m in logs)
