import json
import time
import types

from sandbox_runner import cycle


def test_reruns_flagged_module_on_change(tmp_path, monkeypatch):
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text("VALUE = 1\n")

    monkeypatch.setattr(cycle, "discover_recursive_orphans", lambda repo: {})
    import scripts.discover_isolated_modules as dim

    monkeypatch.setattr(dim, "discover_isolated_modules", lambda *a, **k: [])
    import orphan_analyzer
    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", lambda p: False)

    calls = []

    def fake_auto(mods, recursive=True, validate=False):
        calls.append(list(mods))
        if len(calls) == 1:
            return None, {"added": [], "failed": list(mods), "redundant": []}
        return None, {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto)
    monkeypatch.setattr(cycle, "append_orphan_classifications", lambda *a, **k: None)

    class DummyMetric:
        def __init__(self):
            self.inc_calls = 0
            self.dec_calls = 0

        def inc(self, n=1):
            self.inc_calls += n

        def dec(self, n=1):
            self.dec_calls += n

    metrics = {
        "orphan_modules_reintroduced_total": DummyMetric(),
        "orphan_modules_tested_total": DummyMetric(),
        "orphan_modules_failed_total": DummyMetric(),
        "orphan_modules_redundant_total": DummyMetric(),
        "orphan_modules_legacy_total": DummyMetric(),
        "orphan_modules_reclassified_total": DummyMetric(),
    }
    for name, metric in metrics.items():
        monkeypatch.setattr(cycle, name, metric, raising=False)

    class Settings:
        auto_include_isolated = True
        recursive_isolated = False

    ctx = types.SimpleNamespace(
        repo=tmp_path,
        settings=Settings(),
        module_map=set(),
        orphan_traces={"mod.py": {"classification": "candidate", "parents": []}},  # path-ignore
        tracker=types.SimpleNamespace(merge_history=lambda self, other: None),
    )

    cycle.include_orphan_modules(ctx)
    assert calls == [["mod.py"]]  # path-ignore
    first_mtime = ctx.orphan_traces["mod.py"]["mtime"]  # path-ignore
    cache_path = tmp_path / "sandbox_data" / "orphan_modules.json"
    data = json.loads(cache_path.read_text())
    assert data["mod.py"]["failed"] is True  # path-ignore
    assert "mtime" in data["mod.py"]  # path-ignore

    time.sleep(1)
    mod.write_text("VALUE = 2\n")

    cycle.include_orphan_modules(ctx)
    assert calls[1] == ["mod.py"]  # path-ignore
    assert "mod.py" not in ctx.orphan_traces  # path-ignore
    data = json.loads(cache_path.read_text())
    assert "mod.py" not in data  # path-ignore
    assert metrics["orphan_modules_reintroduced_total"].inc_calls == 1
