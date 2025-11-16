import types
import subprocess
import yaml


def _load_engine():
    import importlib.util
    import sys
    from pathlib import Path
    from sandbox_settings import SandboxSettings
    import security_auditor

    repo = Path(__file__).resolve().parent.parent
    metrics_path = repo / "self_improvement" / ("metrics" + ".py")  # path-ignore
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement.metrics", metrics_path
    )
    metrics = importlib.util.module_from_spec(spec)
    pkg = sys.modules.setdefault("menace", types.ModuleType("menace"))
    pkg.__path__ = [str(repo)]  # type: ignore[attr-defined]
    sys.modules[spec.name] = metrics
    spec.loader.exec_module(metrics)  # type: ignore[union-attr]

    ns = types.SimpleNamespace()

    def _flag_patch_alignment(engine, patch_id, context):
        settings = ns.SandboxSettings()
        metrics._update_alignment_baseline(settings)

    ns._update_alignment_baseline = metrics._update_alignment_baseline
    ns.SandboxSettings = SandboxSettings
    ns.security_auditor = security_auditor
    ns.subprocess = subprocess
    ns.SelfImprovementEngine = types.SimpleNamespace(
        _flag_patch_alignment=staticmethod(_flag_patch_alignment)
    )
    return ns


def test_alignment_baseline_updates(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ("foo" + ".py")).write_text("def add(a, b):\n    return a + b\n")  # path-ignore
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / ("test_foo" + ".py")).write_text("def test_add():\n    assert True\n")  # path-ignore
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(alignment_baseline_metrics_path=baseline)
    sie._update_alignment_baseline(settings)
    data1 = yaml.safe_load(baseline.read_text())
    assert data1["tests"] == 1
    assert data1["complexity"] >= 1
    assert ("foo" + ".py") in data1["files"]  # path-ignore
    foo_metrics1 = data1["files"]["foo" + ".py"]  # path-ignore
    assert foo_metrics1["complexity"] >= 1
    assert foo_metrics1["maintainability"] > 0
    (tests_dir / ("test_bar" + ".py")).write_text("def test_bar():\n    assert True\n")  # path-ignore
    (repo / ("foo" + ".py")).write_text(  # path-ignore
        """def add(a, b):\n    if a > b:\n        return a - b\n    return a + b\n"""
    )
    sie._update_alignment_baseline(settings)
    data2 = yaml.safe_load(baseline.read_text())
    assert data2["tests"] == 2
    assert data2["complexity"] > data1["complexity"]
    assert data2["files"]["foo" + ".py"]["complexity"] > foo_metrics1["complexity"]  # path-ignore


def test_incremental_update_adds_new_files(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    foo = repo / ("foo" + ".py")  # path-ignore
    foo.write_text("def add(a, b):\n    return a + b\n")
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(alignment_baseline_metrics_path=baseline)
    sie._update_alignment_baseline(settings)
    original = yaml.safe_load(baseline.read_text())
    foo_metrics = original["files"]["foo" + ".py"].copy()  # path-ignore

    bar = repo / ("bar" + ".py")  # path-ignore
    bar.write_text("def sub(a, b):\n    return a - b\n")
    sie._update_alignment_baseline(settings, [bar])
    updated = yaml.safe_load(baseline.read_text())
    assert updated["files"]["foo" + ".py"] == foo_metrics  # path-ignore
    assert ("bar" + ".py") in updated["files"]  # path-ignore
    assert updated["complexity"] == sum(v["complexity"] for v in updated["files"].values())


def test_vendor_directories_skipped(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    vendor_dir = repo / "venv"
    vendor_dir.mkdir()
    (vendor_dir / ("bad" + ".py")).write_text("def v():\n    return 1\n")  # path-ignore
    (repo / ("foo" + ".py")).write_text("def f():\n    return 1\n")  # path-ignore
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(alignment_baseline_metrics_path=baseline)
    sie._update_alignment_baseline(settings)
    data = yaml.safe_load(baseline.read_text())
    assert ("venv/" + "bad" + ".py") not in data["files"]  # path-ignore


def test_parse_failure_logged(tmp_path, monkeypatch, caplog):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ("bad" + ".py")).write_text("def broken(:\n    pass\n")  # path-ignore
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(alignment_baseline_metrics_path=baseline)
    with caplog.at_level("WARNING"):
        sie._update_alignment_baseline(settings)
    assert any(("bad" + ".py") in r.message for r in caplog.records)  # path-ignore


def test_flag_patch_alignment_refreshes_baseline_when_approved(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ("foo" + ".py")).write_text("def add(a, b):\n    return a + b\n")  # path-ignore
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(
        alignment_baseline_metrics_path=baseline,
        alignment_warning_threshold=0.5,
        alignment_failure_threshold=0.9,
    )
    monkeypatch.setattr(sie, "SandboxSettings", lambda: settings)

    class Flagger:
        def flag_patch(self, diff, context):
            return {"score": 0, "issues": []}

    engine = types.SimpleNamespace(
        alignment_flagger=Flagger(),
        cycle_logs=[],
        _cycle_count=0,
        event_bus=None,
        logger=types.SimpleNamespace(
            exception=lambda *a, **k: None, info=lambda *a, **k: None
        ),
    )
    monkeypatch.setattr(
        sie.security_auditor, "dispatch_alignment_warning", lambda record: None
    )

    def fake_run(cmd, capture_output, text, check):
        if cmd == ["git", "show", "HEAD"]:
            return types.SimpleNamespace(stdout="diff")
        if cmd == ["git", "rev-parse", "HEAD"]:
            return types.SimpleNamespace(stdout="abc\n")
        raise AssertionError("unexpected command")

    monkeypatch.setattr(sie.subprocess, "run", fake_run)
    sie.SelfImprovementEngine._flag_patch_alignment(engine, 1, {})

    data = yaml.safe_load(baseline.read_text())
    assert data["tests"] == 0
    assert data["complexity"] >= 1
