import importlib.util
import os
import sys
import types
import yaml


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement",
        os.path.join(os.path.dirname(__file__), "..", "self_improvement.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_alignment_baseline_updates(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "foo.py").write_text("def add(a, b):\n    return a + b\n")
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text("def test_add():\n    assert True\n")
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(alignment_baseline_metrics_path=str(baseline))
    sie._update_alignment_baseline(settings)
    data1 = yaml.safe_load(baseline.read_text())
    assert data1["tests"] == 1
    assert data1["complexity"] >= 1
    (tests_dir / "test_bar.py").write_text("def test_bar():\n    assert True\n")
    (repo / "foo.py").write_text(
        """def add(a, b):\n    if a > b:\n        return a - b\n    return a + b\n"""
    )
    sie._update_alignment_baseline(settings)
    data2 = yaml.safe_load(baseline.read_text())
    assert data2["tests"] == 2
    assert data2["complexity"] > data1["complexity"]


def test_flag_patch_alignment_refreshes_baseline_when_approved(tmp_path, monkeypatch):
    sie = _load_engine()
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "foo.py").write_text("def add(a, b):\n    return a + b\n")
    baseline = tmp_path / "baseline.yaml"
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    settings = types.SimpleNamespace(
        alignment_baseline_metrics_path=str(baseline),
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
