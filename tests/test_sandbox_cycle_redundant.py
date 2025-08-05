import types
from pathlib import Path

import pytest

import sandbox_runner.cycle as cycle
import sandbox_runner as pkg


def test_cycle_skips_redundant_modules(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_AUTO_INCLUDE_ISOLATED", "1")
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    # Record calls to auto_include_modules
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append((list(mods), recursive, validate))

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *a, **k: None)

    (tmp_path / "foo.py").write_text("pass\n")

    def fake_discover(repo_path):
        assert Path(repo_path) == tmp_path
        return {"foo": {"parents": []}}

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)

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
