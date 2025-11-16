from pathlib import Path

import orphan_analyzer
from sandbox_runner import cycle
from menace_sandbox.roi_tracker import ROITracker


def test_include_orphan_modules_validates_and_integrates(monkeypatch, tmp_path):
    (tmp_path / "isolated.py").write_text("import helper\nimport legacy\n")  # path-ignore
    (tmp_path / "helper.py").write_text("VALUE = 1\n")  # path-ignore
    (tmp_path / "legacy.py").write_text("VALUE = 2\n")  # path-ignore

    def fake_analyze(path: Path) -> bool:
        return path.name == "legacy.py"  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "analyze_redundancy", fake_analyze)

    calls: dict[str, object] = {}

    tracker = ROITracker()
    tracker.roi_history = [0.2]

    def fake_auto_include(mods, recursive=False, validate=False):
        calls["mods"] = list(mods)
        calls["recursive"] = recursive
        calls["validate"] = validate
        return tracker, {"added": list(mods), "failed": [], "redundant": []}

    monkeypatch.setattr(cycle, "auto_include_modules", fake_auto_include)
    monkeypatch.setattr(cycle, "append_orphan_cache", lambda *_: None)
    monkeypatch.setattr(cycle, "prune_orphan_cache", lambda *_: None)
    monkeypatch.setattr(cycle, "append_orphan_classifications", lambda *_: None)

    class DummyMetric:
        def inc(self, *_):
            pass

        def dec(self, *_):
            pass

    monkeypatch.setattr(cycle, "orphan_modules_tested_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_reintroduced_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_failed_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_redundant_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_legacy_total", DummyMetric())
    monkeypatch.setattr(cycle, "orphan_modules_reclassified_total", DummyMetric())

    monkeypatch.setattr(
        cycle,
        "discover_recursive_orphans",
        lambda repo: {
            "isolated": {"classification": "candidate"},
            "helper": {"classification": "candidate"},
            "legacy": {"classification": "candidate"},
        },
    )

    import scripts.discover_isolated_modules as dim

    monkeypatch.setattr(dim, "discover_isolated_modules", lambda *_, **__: ["isolated.py"])  # path-ignore

    class Settings:
        auto_include_isolated = True
        recursive_isolated = True

    class Ctx:
        def __init__(self, repo: Path) -> None:
            self.repo = repo
            self.settings = Settings()
            self.module_map: set[str] = set()
            self.orphan_traces: dict[str, dict[str, object]] = {}
            self.tracker = ROITracker()

    ctx = Ctx(tmp_path)
    cycle.include_orphan_modules(ctx)

    assert sorted(calls["mods"]) == ["helper.py", "isolated.py"]  # path-ignore
    assert calls["recursive"] and calls["validate"]
    assert ctx.module_map == {"isolated.py", "helper.py"}  # path-ignore
    assert "legacy.py" not in calls["mods"]  # path-ignore
    assert sorted(ctx.orphan_traces.keys()) == ["legacy.py"]  # path-ignore
    assert ctx.orphan_traces["legacy.py"]["redundant"] is True  # path-ignore
    assert ctx.tracker.roi_history == [0.2]
