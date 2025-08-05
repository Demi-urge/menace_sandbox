import types
import sys
from pathlib import Path

import pytest

import sandbox_runner.cycle as cycle
import sandbox_runner as pkg


def test_cycle_validates_orphans(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "root.py").write_text("import dep_pass\nimport dep_fail\n")
    (repo / "dep_pass.py").write_text("VALUE = 1\n")
    (repo / "dep_fail.py").write_text("VALUE = 1\n")
    (repo / "redundant.py").write_text("VALUE = 1\n")

    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    def fake_discover(path):
        assert Path(path) == repo
        return {
            "root": {"parents": [], "redundant": False},
            "dep_pass": {"parents": ["root"], "redundant": False},
            "dep_fail": {"parents": ["root"], "redundant": False},
            "redundant": {"parents": [], "redundant": True},
        }

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)

    calls = []
    sts_mod = types.ModuleType("self_test_service")

    class DummySTS:
        def __init__(self, pytest_args, **kwargs):
            self.arg = pytest_args

        def run_once(self):
            calls.append(self.arg)
            if self.arg == "dep_fail.py":
                return {"failed": 1}, []
            return {"failed": 0}, [self.arg]

    sts_mod.SelfTestService = DummySTS
    monkeypatch.setitem(sys.modules, "self_test_service", sts_mod)

    def fake_auto_include(mods, recursive=False, validate=False):
        passed = []
        for m in list(mods):
            svc = sts_mod.SelfTestService(pytest_args=m)
            res, passed_mods = svc.run_once()
            if not res.get("failed"):
                passed.extend(passed_mods)
        mods[:] = passed

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)

    monkeypatch.setattr(
        cycle,
        "ResourceTuner",
        lambda: types.SimpleNamespace(adjust=lambda tracker, presets: presets),
    )
    cycle.SANDBOX_ENV_PRESETS = [{}]

    def fake_info(msg, *a, **k):
        if msg == "patch application":
            raise RuntimeError("stop")

    monkeypatch.setattr(cycle.logger, "info", fake_info)

    ctx = types.SimpleNamespace(
        prev_roi=0.0,
        cycles=1,
        orchestrator=types.SimpleNamespace(run_cycle=lambda models: None),
        improver=types.SimpleNamespace(
            run_cycle=lambda: types.SimpleNamespace(roi=types.SimpleNamespace(roi=0.0)),
            module_index=None,
        ),
        tester=types.SimpleNamespace(run_once=lambda: None),
        sandbox=types.SimpleNamespace(analyse_and_fix=lambda *a, **k: None),
        repo=repo,
        module_map=set(),
        orphan_traces={},
        tracker=types.SimpleNamespace(register_metrics=lambda *a, **k: None),
        models=[],
        module_counts={},
        meta_log=types.SimpleNamespace(last_patch_id=None),
        settings=types.SimpleNamespace(
            auto_include_isolated=True, recursive_isolated=True
        ),
    )

    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)

    assert calls == ["root.py", "dep_pass.py", "dep_fail.py"]
    assert ctx.module_map == {"root.py", "dep_pass.py"}
    assert ctx.orphan_traces["redundant.py"]["redundant"] is True
    assert "dep_fail.py" not in ctx.module_map
