import types
import sys
from pathlib import Path

import pytest

import sandbox_runner.cycle as cycle
import sandbox_runner.environment as env
import orphan_analyzer
import sandbox_runner as pkg


def _prepare(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    monkeypatch.setenv("SANDBOX_RECURSIVE_ISOLATED", "1")
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    # Record calls to auto_include_modules
    calls = []

    def fake_auto_include(mods, recursive=False):
        calls.append((list(mods), recursive))

    monkeypatch.setattr(env, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "auto_include_modules", env.auto_include_modules)

    def discover_isolated_modules(repo_path, *, recursive=False):
        assert Path(repo_path) == tmp_path
        assert recursive
        names = ["isolated.py", "helper.py"]
        result = []
        for name in names:
            path = Path(repo_path) / name
            if not orphan_analyzer.analyze_redundancy(path):
                result.append(name)
        return result

    mod = types.ModuleType("scripts.discover_isolated_modules")
    mod.discover_isolated_modules = discover_isolated_modules
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
        orphan_traces={},
        tracker=object(),
        models=[],
        module_counts={},
        meta_log=types.SimpleNamespace(last_patch_id=None),
    )
    return ctx, calls


def test_auto_include_isolated_with_helper(monkeypatch, tmp_path):
    ctx, calls = _prepare(monkeypatch, tmp_path)
    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: False)
    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    assert calls == [(["isolated.py", "helper.py"], True)]
    assert ctx.module_map == {"isolated.py", "helper.py"}


def test_auto_include_skips_redundant(monkeypatch, tmp_path):
    ctx, calls = _prepare(monkeypatch, tmp_path)

    def analyze(path):
        return path.name == "helper.py"

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", analyze)
    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)
    assert calls == [(["isolated.py"], True)]
    assert ctx.module_map == {"isolated.py"}
