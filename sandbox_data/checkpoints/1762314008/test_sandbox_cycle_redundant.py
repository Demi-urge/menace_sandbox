import types
from pathlib import Path

import pytest

import sandbox_runner.cycle as cycle
import sandbox_runner as pkg


def test_cycle_skips_redundant_modules(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)
    monkeypatch.setattr(pkg, "build_section_prompt", lambda *a, **k: "", raising=False)
    monkeypatch.setattr(pkg, "GPT_SECTION_PROMPT_MAX_LENGTH", 0, raising=False)

    # Record calls to auto_include_modules
    calls: list[tuple[list[str], bool, bool]] = []

    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append((list(mods), recursive, validate))
        return object(), {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *a, **k: None)

    (tmp_path / "foo.py").write_text("pass\n")  # path-ignore

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
        orphan_traces={"foo.py": {"parents": [], "redundant": True}},  # path-ignore
        tracker=types.SimpleNamespace(register_metrics=lambda *a, **k: None, merge_history=lambda *a, **k: None),
        models=[],
        module_counts={},
        meta_log=types.SimpleNamespace(last_patch_id=None),
        settings=types.SimpleNamespace(
            auto_include_isolated=True, recursive_isolated=True
        ),
    )

    with pytest.raises(RuntimeError):
        cycle._sandbox_cycle_runner(ctx, None, None, ctx.tracker)

    assert calls == []
    assert ctx.orphan_traces == {"foo.py": {"parents": [], "redundant": True}}  # path-ignore


def test_redundant_modules_can_be_tested(monkeypatch, tmp_path):
    monkeypatch.delenv("SANDBOX_AUTO_INCLUDE_ISOLATED", raising=False)
    monkeypatch.delenv("SANDBOX_RECURSIVE_ISOLATED", raising=False)

    calls: list[list[str]] = []

    def fake_auto_include(mods, recursive=False, validate=False):
        calls.append(list(mods))
        return object(), {"added": [], "failed": [], "redundant": list(mods)}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "append_orphan_classifications", lambda *a, **k: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *a, **k: None)

    cache_vals = [
        {},
        {"foo.py": {"classification": "legacy", "redundant": True}},  # path-ignore
    ]

    def fake_load_cache(_repo):
        return cache_vals.pop(0)

    monkeypatch.setattr(cycle, "load_orphan_cache", fake_load_cache)

    (tmp_path / "foo.py").write_text("pass\n")  # path-ignore

    def fake_discover(repo_path):
        assert Path(repo_path) == tmp_path
        return {"foo": {"parents": []}}

    monkeypatch.setattr(cycle, "discover_recursive_orphans", fake_discover)

    class DummyMetric:
        def __init__(self):
            self.value = 0

        def inc(self, v=1):
            self.value += v

        def dec(self, v=1):
            self.value -= v

    for name in [
        "orphan_modules_tested_total",
        "orphan_modules_reintroduced_total",
        "orphan_modules_failed_total",
        "orphan_modules_redundant_total",
        "orphan_modules_legacy_total",
        "orphan_modules_reclassified_total",
    ]:
        monkeypatch.setattr(cycle, name, DummyMetric(), raising=False)

    ctx = types.SimpleNamespace(
        repo=tmp_path,
        module_map=set(),
        orphan_traces={"foo.py": {"parents": [], "classification": "redundant", "redundant": True}},  # path-ignore
        tracker=types.SimpleNamespace(merge_history=lambda *a, **k: None),
        settings=types.SimpleNamespace(
            auto_include_isolated=True, recursive_isolated=True, test_redundant_modules=True
        ),
    )

    cycle.include_orphan_modules(ctx)

    assert calls == [["foo.py"]]  # path-ignore
    assert ctx.orphan_traces["foo.py"]["classification"] == "legacy"  # path-ignore
