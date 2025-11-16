import importlib.util
import sys
import types
from pathlib import Path

import environment_generator as eg
import roi_tracker as rt


def _load_cycle(monkeypatch):
    pkg = types.ModuleType("sandbox_runner")
    pkg.build_section_prompt = lambda *a, **k: ""
    pkg.GPT_SECTION_PROMPT_MAX_LENGTH = 10
    pkg.SANDBOX_ENV_PRESETS = [{}]
    env_mod = types.ModuleType("sandbox_runner.environment")
    env_mod.SANDBOX_ENV_PRESETS = [{}]
    mp_mod = types.ModuleType("sandbox_runner.metrics_plugins")
    mp_mod.collect_plugin_metrics = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "sandbox_runner", pkg)
    monkeypatch.setitem(sys.modules, "sandbox_runner.environment", env_mod)
    monkeypatch.setitem(sys.modules, "sandbox_runner.metrics_plugins", mp_mod)
    pkg.environment = env_mod
    pkg.metrics_plugins = mp_mod

    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(stdout=""))
    radon_mod = types.ModuleType("radon.metrics")
    radon_mod.mi_visit = lambda *a, **k: 0
    monkeypatch.setitem(sys.modules, "radon.metrics", radon_mod)
    pyl_lint = types.ModuleType("pylint.lint")
    pyl_rep = types.ModuleType("pylint.reporters.text")
    pyl_lint.Run = lambda *a, **k: types.SimpleNamespace(linter=types.SimpleNamespace(stats=types.SimpleNamespace(global_note=0)))
    pyl_rep.TextReporter = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pylint.lint", pyl_lint)
    monkeypatch.setitem(sys.modules, "pylint.reporters.text", pyl_rep)

    path = Path(__file__).resolve().parents[1] / "sandbox_runner" / "cycle.py"  # path-ignore
    spec = importlib.util.spec_from_file_location(
        "sandbox_runner.cycle", str(path), submodule_search_locations=[str(path.parent)]
    )
    cycle = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "sandbox_runner.cycle", cycle)
    spec.loader.exec_module(cycle)
    return cycle


class DummyCtx:
    def __init__(self, repo: Path):
        self.repo = repo
        self.cycles = 1
        self.models = None
        self.orchestrator = types.SimpleNamespace(run_cycle=lambda models: None)
        self.improver = types.SimpleNamespace(
            run_cycle=lambda: types.SimpleNamespace(roi=types.SimpleNamespace(roi=0.1))
        )
        self.tester = types.SimpleNamespace(_run_once=lambda: None)
        self.sandbox = types.SimpleNamespace(analyse_and_fix=lambda limit=1: None)
        self.predicted_roi = None
        self.predicted_lucrativity = None
        self.prev_roi = 0.0
        self.dd_bot = types.SimpleNamespace(scan=lambda: [])
        self.res_db = None
        self.module_counts = {}
        self.meta_log = types.SimpleNamespace(
            last_patch_id=0,
            flagged_sections=set(),
            log_cycle=lambda *a, **k: None,
            rankings=lambda: {},
            diminishing=lambda threshold=None: [],
        )
        self.data_bot = types.SimpleNamespace(collect=lambda *a, **k: None)
        self.patch_db_path = repo / "patch.db"
        self.patch_db_path.touch()
        self.engine = types.SimpleNamespace(
            apply_patch=lambda *a, **k: (1, False, 0.0),
            rollback_patch=lambda *a, **k: None,
        )
        self.offline_suggestions = False
        self.brainstorm_history = []
        self.conversations = {}
        self.base_roi_tolerance = 0.01
        self.roi_tolerance = 0.01
        self.gpt_client = None
        self.pre_roi_bot = None
        self.adapt_presets = False
        self.plugins = None
        self.extra_metrics = None

    def changed_modules(self, last):
        return ["mod.py"], last  # path-ignore


def test_synergy_predictions(monkeypatch, tmp_path):
    tracker = rt.ROITracker()
    vals = [4.9, 4.8, 4.9]
    prev = 0.0
    for i, val in enumerate(vals):
        roi = 0.1 * (i + 1)
        tracker.update(prev, roi, metrics={"security_score": 70, "synergy_security_score": val})
        prev = roi

    presets = [{"SECURITY_LEVEL": 2}]
    out = eg.adapt_presets(tracker, [presets[0].copy()])
    assert out[0]["SECURITY_LEVEL"] == 2

    monkeypatch.setattr(tracker, "predict_synergy_metric", lambda name: 6.0)
    out2 = eg.adapt_presets(tracker, [presets[0].copy()])
    assert out2[0]["SECURITY_LEVEL"] > 2

    cycle = _load_cycle(monkeypatch)
    repo = tmp_path
    repo.joinpath("mod.py").write_text("print(1)")  # path-ignore
    ctx = DummyCtx(repo)
    cycle._sandbox_cycle_runner(ctx, "mod.py:sec", "print(1)", tracker, scenario="s")  # path-ignore
    assert tracker.synergy_metrics_history["synergy_safety_rating"]
