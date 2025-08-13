import importlib.util
import os
import sys
import yaml


def _load_engine():
    spec = importlib.util.spec_from_file_location(
        "menace.self_improvement_engine",
        os.path.join(os.path.dirname(__file__), "..", "self_improvement_engine.py"),
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
    monkeypatch.setenv("ALIGNMENT_BASELINE_METRICS_PATH", str(baseline))
    sie._update_alignment_baseline()
    data1 = yaml.safe_load(baseline.read_text())
    assert data1["tests"] == 1
    assert data1["complexity"] >= 1
    (tests_dir / "test_bar.py").write_text("def test_bar():\n    assert True\n")
    (repo / "foo.py").write_text(
        """def add(a, b):\n    if a > b:\n        return a - b\n    return a + b\n"""
    )
    sie._update_alignment_baseline()
    data2 = yaml.safe_load(baseline.read_text())
    assert data2["tests"] == 2
    assert data2["complexity"] > data1["complexity"]
