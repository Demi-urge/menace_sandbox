from pathlib import Path
import json
import sys
import types

import pytest

import sandbox_runner.cycle as cycle
import sandbox_runner as pkg


def _prepare(monkeypatch, repo, data_dir):
    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))

    # avoid heavy prompt building
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    map_path = data_dir / "module_map.json"
    map_path.write_text(json.dumps({"modules": {}, "groups": {}}))

    # isolate discovery returns only the main module; dependency discovered recursively
    def fake_discover_orphans(path):
        assert Path(path) == repo
        return {
            "dep": {"parents": ["iso"], "redundant": False},
            "old": {"parents": [], "redundant": True},
        }

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover_orphans)

    iso_mod = types.ModuleType("scripts.discover_isolated_modules")

    def fake_discover_isolated(path, *, recursive=True):
        assert Path(path) == repo
        assert recursive is True
        return ["iso.py"]  # path-ignore

    iso_mod.discover_isolated_modules = fake_discover_isolated
    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.discover_isolated_modules = iso_mod
    monkeypatch.setitem(sys.modules, "scripts", scripts_pkg)
    monkeypatch.setitem(sys.modules, "scripts.discover_isolated_modules", iso_mod)

    calls = []

    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append(sorted(mods))
        data = json.loads(map_path.read_text())
        for m in mods:
            data["modules"][Path(m).name] = 1
        map_path.write_text(json.dumps(data))
        return object(), {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *a, **k: None)
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
    return ctx, calls, map_path


def test_recursive_module_auto_inclusion(tmp_path, monkeypatch):
    repo = tmp_path
    (repo / "iso.py").write_text("import dep\n")  # path-ignore
    (repo / "dep.py").write_text("x = 1\n")  # path-ignore
    (repo / "old.py").write_text("x = 2\n")  # path-ignore

    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()

    ctx, calls, map_path = _prepare(monkeypatch, repo, data_dir)

    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)

    # dependency and main module are self-tested and integrated
    assert calls and calls[0] == ["dep.py", "iso.py"]  # path-ignore
    data = json.loads(map_path.read_text())
    assert set(data["modules"]) == {"iso.py", "dep.py"}  # path-ignore
    # redundant module is cached but not integrated
    assert ctx.orphan_traces.get("old.py", {}).get("redundant") is True  # path-ignore
    assert "old.py" not in data["modules"]  # path-ignore
