import types
import sys
from pathlib import Path
from tests.test_menace_master import _setup_mm_stubs


class DummyAudit:
    def __init__(self, path):
        self.path = Path(path)
        self.records = []

    def record(self, data):
        self.records.append(data)


class _SandboxMetaLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.audit = DummyAudit(path)
        self.records = []
        self.module_deltas = {}
        self.module_entropy_deltas = {}
        self.flagged_sections = set()
        self.last_patch_id = 0
    def log_cycle(
        self,
        cycle: int,
        roi: float,
        modules: list[str],
        reason: str,
        *,
        entropy_delta: float = 0.0,
        exec_time: float = 0.0,
    ) -> None:
        prev = self.records[-1][1] if self.records else 0.0
        delta = roi - prev
        self.records.append((cycle, roi, delta, modules, reason))
        for m in modules:
            self.module_deltas.setdefault(m, []).append(delta)
            self.module_entropy_deltas.setdefault(m, []).append(entropy_delta)
        try:
            self.audit.record(
                {
                    "cycle": cycle,
                    "roi": roi,
                    "delta": delta,
                    "modules": modules,
                    "reason": reason,
                }
            )
        except Exception:
            pass

    def diminishing(self, threshold: float | None = None, consecutive: int = 3) -> list[str]:
        flags = []
        thr = 0.0 if threshold is None else float(threshold)
        eps = 1e-3
        for m, vals in self.module_deltas.items():
            if m in self.flagged_sections:
                continue
            if len(vals) < consecutive:
                continue
            window = vals[-consecutive:]
            mean = sum(window) / consecutive
            if len(window) > 1:
                var = sum((v - mean) ** 2 for v in window) / len(window)
                std = var ** 0.5
            else:
                std = 0.0
            if abs(mean) <= thr and std < eps:
                flags.append(m)
                self.flagged_sections.add(m)
        return flags


class DummyEngine:
    def __init__(self):
        self.val = 0.0

    def run_cycle(self):
        self.val += 0.01
        return types.SimpleNamespace(roi=types.SimpleNamespace(roi=self.val))


class DummyTracker:
    def __init__(self):
        self.roi_history = []
        self.metrics_history = {}

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.roi_history.append(curr)
        for k, v in (metrics or {}).items():
            self.metrics_history.setdefault(k, []).append(v)
        return 0.0, [], False

    def diminishing(self):
        return 0.011


def test_section_metrics_and_diminishing(tmp_path):
    log = _SandboxMetaLogger(tmp_path / "log.txt")
    engine = DummyEngine()
    tracker = DummyTracker()
    for i in range(4):
        res = engine.run_cycle()
        tracker.update(0.01 * i, res.roi.roi, modules=["mod.py"], metrics={"m": i})  # path-ignore
        log.log_cycle(i, res.roi.roi, ["mod.py"], "test")  # path-ignore
    flags = log.diminishing(tracker.diminishing())
    assert "mod.py" in flags  # path-ignore
    assert tracker.metrics_history["m"] == [0, 1, 2, 3]


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg, _, sub = name.partition(".")
    pkg_mod = sys.modules.get(pkg)
    if pkg_mod and sub:
        setattr(pkg_mod, sub, mod)
    return mod


class DummyBot:
    def __init__(self, *a, **k):
        pass


class DummySandbox:
    def __init__(self, *a, **k):
        pass

    def analyse_and_fix(self):
        pass


class DummyTracker2:
    def __init__(self, *a, **k):
        self.calls = []

    def update(self, prev, curr, modules=None, resources=None, metrics=None):
        self.calls.append({"modules": modules, "metrics": metrics})
        return 0.0, [], False

    def forecast(self):
        return 0.0, (0.0, 0.0)

    def diminishing(self):
        return 0.0

    def record_metric_prediction(self, *a, **k):
        pass

    def record_prediction(self, *a, **k):
        pass

    def rolling_mae(self, window=None):
        return 0.0

    def load_history(self, path):
        pass

    def save_history(self, path):
        pass


async def _fake_worker(snippet, env_input, threshold):
    import sandbox_runner.environment as env

    cpu_lim = float(env_input.get("CPU_LIMIT", 0))
    mem_lim = env._parse_size(env_input.get("MEMORY_LIMIT", 0))
    metrics = {"cpu": cpu_lim / 2, "memory": mem_lim / 2}
    return {"exit_code": 1}, [(0.0, 0.0, metrics)]


def test_run_repo_section_simulations_plugins(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")  # path-ignore

    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.patch_suggestion_db", PatchSuggestionDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker2)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot)
    _stub_module(monkeypatch, "networkx", DiGraph=object)
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )
    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", lambda *a, **k: {"risk_flags_triggered": []})
    monkeypatch.setattr(env, "_section_worker", _fake_worker)

    def plugin(prev, roi, resources):
        return {"plugin_metric": resources.get("cpu", 0) + resources.get("memory", 0)}

    monkeypatch.setattr(sandbox_runner.metrics_plugins, "discover_metrics_plugins", lambda env=None: [plugin])

    def collect(plugs, prev, roi, res):
        merged = {}
        for fn in plugs:
            merged.update(fn(prev, roi, res))
        return merged

    monkeypatch.setattr(sandbox_runner.metrics_plugins, "collect_plugin_metrics", collect)

    presets = {
        "m.py": [  # path-ignore
            {"SCENARIO_NAME": "dev", "CPU_LIMIT": "1", "MEMORY_LIMIT": "32Mi"},
            {"SCENARIO_NAME": "prod", "CPU_LIMIT": "2", "MEMORY_LIMIT": "64Mi"},
        ]
    }

    tracker = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets
    )

    assert len(tracker.calls) == 2
    seen = {call["modules"][1]: call for call in tracker.calls}
    for preset in presets["m.py"]:  # path-ignore
        name = preset["SCENARIO_NAME"]
        metrics = seen[name]["metrics"]
        assert "plugin_metric" in metrics
        assert f"plugin_metric:{name}" in metrics
        assert metrics["cpu"] <= float(preset["CPU_LIMIT"])
        assert metrics["memory"] <= env._parse_size(preset["MEMORY_LIMIT"])


def test_run_repo_section_simulations_details(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")

    (tmp_path / "m.py").write_text("def f():\n    return 1\n")  # path-ignore

    _stub_module(monkeypatch, "menace.self_improvement_policy", SelfImprovementPolicy=DummyBot)
    _stub_module(monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot)
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummySandbox)
    _stub_module(monkeypatch, "menace.self_coding_engine", SelfCodingEngine=DummyBot)
    _stub_module(monkeypatch, "menace.code_database", PatchHistoryDB=DummyBot, CodeDB=DummyBot)
    _stub_module(monkeypatch, "menace.patch_suggestion_db", PatchSuggestionDB=DummyBot)
    _stub_module(monkeypatch, "menace.menace_memory_manager", MenaceMemoryManager=DummyBot)
    _stub_module(monkeypatch, "menace.roi_tracker", ROITracker=DummyTracker2)
    _stub_module(monkeypatch, "menace.audit_trail", AuditTrail=DummyBot)
    _stub_module(monkeypatch, "menace.error_bot", ErrorBot=DummyBot, ErrorDB=lambda p: DummyBot())
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot, DataBot=DummyBot)
    _stub_module(monkeypatch, "menace.discrepancy_detection_bot", DiscrepancyDetectionBot=DummyBot)
    _stub_module(monkeypatch, "menace.pre_execution_roi_bot", PreExecutionROIBot=DummyBot)
    _stub_module(monkeypatch, "networkx", DiGraph=object)
    sqla = types.ModuleType("sqlalchemy")
    sqla_engine = types.ModuleType("sqlalchemy.engine")
    sqla_engine.Engine = object
    monkeypatch.setitem(sys.modules, "sqlalchemy", sqla)
    monkeypatch.setitem(sys.modules, "sqlalchemy.engine", sqla_engine)
    _stub_module(monkeypatch, "jinja2", Template=lambda *a, **k: None)

    import sandbox_runner
    import sandbox_runner.environment as env

    monkeypatch.setattr(
        sandbox_runner,
        "scan_repo_sections",
        lambda p, modules=None: {"m.py": {"sec": ["pass"]}},  # path-ignore
        raising=False,
    )
    monkeypatch.setattr(sandbox_runner, "simulate_execution_environment", lambda *a, **k: {"risk_flags_triggered": []})
    monkeypatch.setattr(env, "_section_worker", _fake_worker)

    presets = {
        "m.py": [  # path-ignore
            {"SCENARIO_NAME": "dev", "CPU_LIMIT": "1", "MEMORY_LIMIT": "32Mi"},
            {"SCENARIO_NAME": "prod", "CPU_LIMIT": "2", "MEMORY_LIMIT": "64Mi"},
        ]
    }

    tracker, details = sandbox_runner.run_repo_section_simulations(
        str(tmp_path), input_stubs=[{}], env_presets=presets, return_details=True
    )

    assert tracker.calls
    assert set(details["m.py"]) == {"dev", "prod"}  # path-ignore
    for scen in ("dev", "prod"):
        rec = details["m.py"][scen][0]  # path-ignore
        assert rec["section"] == "sec"
        assert rec["preset"]["SCENARIO_NAME"] == scen
