import types
import pytest
import sys
from pathlib import Path

import sandbox_runner.cycle as cycle
import sandbox_runner as pkg


def test_cycle_skips_redundant_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    monkeypatch.setenv("SANDBOX_RECURSIVE_ISOLATED", "1")
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    # Record calls to auto_include_modules
    calls = []

    def fake_auto_include(mods, recursive=False):
        calls.append(list(mods))

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    mod = types.ModuleType("scripts.discover_isolated_modules")

    def discover(repo_path, *, recursive=True):
        assert Path(repo_path) == tmp_path
        return ["foo.py"]

    mod.discover_isolated_modules = discover
    pkg_mod = types.ModuleType("scripts")
    pkg_mod.discover_isolated_modules = mod
    monkeypatch.setitem(sys.modules, "scripts", pkg_mod)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", mod)
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
        repo=tmp_path,
        module_map=set(),
        orphan_traces={"foo.py": {"parents": [], "redundant": True}},
        tracker=object(),
        models=[],
        module_counts={},
        meta_log=types.SimpleNamespace(last_patch_id=None),
    )

    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)

    assert calls == []
    assert ctx.orphan_traces == {"foo.py": {"parents": [], "redundant": True}}
