import importlib.util
import sys
import types
from pathlib import Path

import logging
import pytest

# Import stub environment used by other self_improvement tests.  This module
# populates ``sys.modules`` with lightweight stand-ins for the numerous
# dependencies pulled in by ``self_improvement`` during import.  Importing it as
# a side effect keeps the tests here focused solely on the error handling logic
# being exercised.
from . import test_self_improvement_logging  # noqa: F401


def _load_module(monkeypatch, tmp_path):
    """Load the self_improvement package with minimal sandbox settings."""

    # Patch the stubbed ``sandbox_settings`` module before importing
    # ``self_improvement`` so its global initialisation succeeds.
    import sys
    import types as _types

    ss = _types.ModuleType("sandbox_settings")
    class _Settings:
        def __init__(self):
            self.test_redundant_modules = True
            self.exclude_dirs = ""
            self.sandbox_repo_path = str(tmp_path)
            self.sandbox_data_dir = str(tmp_path)

        def __getattr__(self, name):  # pragma: no cover - provide safe defaults
            return 1

    ss.SandboxSettings = _Settings
    ss.load_sandbox_settings = lambda: _Settings()
    sys.modules["sandbox_settings"] = ss
    sys.modules["menace.sandbox_settings"] = ss

    # Ensure optional menace modules required during import expose minimal APIs.
    neuro = sys.modules.setdefault("menace.neuroplasticity", _types.ModuleType("neuro"))
    setattr(neuro, "PathwayDB", object)
    log_mod = sys.modules.setdefault("menace.logging_utils", _types.ModuleType("log"))
    setattr(log_mod, "log_record", lambda **k: {})
    setattr(log_mod, "get_logger", lambda name=None: logging.getLogger(name or "test"))
    setattr(log_mod, "setup_logging", lambda: None)
    setattr(log_mod, "set_correlation_id", lambda *_: None)
    db_mod = sys.modules.setdefault("data_bot", _types.ModuleType("data_bot"))
    setattr(db_mod, "MetricsDB", object)
    sys.modules.setdefault("menace.data_bot", db_mod)
    roi_mod = sys.modules.setdefault("menace.roi_results_db", _types.ModuleType("roi"))
    setattr(roi_mod, "ROIResultsDB", object)
    gpt_mod = sys.modules.setdefault("menace.gpt_memory", _types.ModuleType("gpt"))
    setattr(gpt_mod, "GPTMemoryManager", object)

    sys.modules.setdefault("menace.sandbox_settings", ss)

    pg_mod = _types.ModuleType("menace.self_improvement.patch_generation")
    setattr(pg_mod, "generate_patch", lambda *a, **k: None)
    sys.modules["menace.self_improvement.patch_generation"] = pg_mod

    sr_pkg = sys.modules.setdefault("sandbox_runner", _types.ModuleType("sandbox_runner"))
    sr_boot = _types.ModuleType("sandbox_runner.bootstrap")
    setattr(sr_boot, "initialize_autonomous_sandbox", lambda *a, **k: None)
    sys.modules["sandbox_runner.bootstrap"] = sr_boot

    # ``SandboxSettings`` already defined on stub module above

    path = Path(__file__).resolve().parent.parent / "self_improvement" / "__init__.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace.self_improvement", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    # Avoid touching the real filesystem by redirecting helper functions to the
    # temporary directory used by the test.
    monkeypatch.setattr(mod, "_repo_path", lambda: tmp_path)
    monkeypatch.setattr(mod, "_data_dir", lambda: tmp_path)

    # Provide stub environment with ``auto_include_modules`` placeholder.
    mod.environment = types.SimpleNamespace(
        auto_include_modules=lambda *a, **k: (None, {"added": []}),
        try_integrate_into_workflows=lambda *a, **k: None,
    )

    return mod


def _make_engine(mod):
    """Create a minimal object that supports ``_update_orphan_modules``."""

    class _BT:
        def __init__(self):
            self.values: list[float] = []

        def to_dict(self):
            return {"side_effects": list(self.values)}

        def get(self, name):
            return sum(self.values) / len(self.values) if self.values else 0.0

        def std(self, name):
            if len(self.values) < 2:
                return 0.0
            avg = self.get(name)
            return (sum((v - avg) ** 2 for v in self.values) / len(self.values)) ** 0.5

        def update(self, **metrics):
            if "side_effects" in metrics:
                self.values.append(float(metrics["side_effects"]))

    engine = types.SimpleNamespace(
        logger=logging.getLogger("test"),
        orphan_traces={},
        baseline_tracker=_BT(),
    )
    engine._collect_recursive_modules = lambda mods: mods
    engine._test_orphan_modules = lambda mods: mods
    return engine


def test_try_integrate_failure_logged_and_raised(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)
    engine = _make_engine(mod)
    engine._integrate_orphans = lambda paths: set()
    engine._refresh_module_map = lambda mods: None

    def fail(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod.environment, "try_integrate_into_workflows", fail)
    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        mod.SelfImprovementEngine._update_orphan_modules(engine, ["a.py"])  # path-ignore
    assert "workflow integration failed" in caplog.text


def test_integrate_orphans_failure_logged(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)
    engine = _make_engine(mod)
    engine._refresh_module_map = lambda mods: None

    monkeypatch.setattr(mod.environment, "try_integrate_into_workflows", lambda *a, **k: None)

    def fail(paths):
        raise RuntimeError("nope")

    engine._integrate_orphans = fail
    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        mod.SelfImprovementEngine._update_orphan_modules(engine, ["b.py"])  # path-ignore
    assert "orphan integration failed" in caplog.text


def test_refresh_module_map_failure_logged(monkeypatch, tmp_path, caplog):
    mod = _load_module(monkeypatch, tmp_path)
    engine = _make_engine(mod)

    monkeypatch.setattr(mod.environment, "try_integrate_into_workflows", lambda *a, **k: None)
    engine._integrate_orphans = lambda paths: set()

    def fail(mods):
        raise RuntimeError("bad map")

    engine._refresh_module_map = fail
    caplog.set_level("ERROR")
    with pytest.raises(RuntimeError):
        mod.SelfImprovementEngine._update_orphan_modules(engine, ["c.py"])  # path-ignore
    assert "module map refresh failed" in caplog.text

