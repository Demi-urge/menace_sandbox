import json
import types
import sys
import logging
from pathlib import Path


def test_engine_skips_heavy_side_effects(monkeypatch, tmp_path):
    import menace_sandbox.self_improvement as sie
    from menace_sandbox.self_improvement.baseline_tracker import BaselineTracker

    (tmp_path / "a.py").write_text("print('hi')")  # path-ignore
    (tmp_path / "b.py").write_text("print('bye')")  # path-ignore
    data_dir = tmp_path / "sandbox_data"
    data_dir.mkdir()
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(tmp_path))

    class DummySTS:
        def __init__(self, *a, **k):
            pass

        def run_once(self):
            return {"failed": False}

    monkeypatch.setitem(
        sys.modules,
        "self_test_service",
        types.SimpleNamespace(SelfTestService=DummySTS),
    )

    def fake_classify(path, include_meta=False):
        assert include_meta
        name = Path(path).name
        return (
            "candidate",
            {"side_effects": 5 if name == "a.py" else 11},  # path-ignore
        )

    monkeypatch.setattr(sie, "classify_module", fake_classify)

    class DummyLogger:
        def info(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    dummy = types.SimpleNamespace(
        logger=DummyLogger(), baseline_tracker=BaselineTracker()
    )

    # Seed baseline with a safe module
    res = sie.SelfImprovementEngine._test_orphan_modules(dummy, ["a.py"])  # path-ignore
    assert res == {"a.py"}  # path-ignore

    # Second module exceeds dynamic threshold and is skipped
    res = sie.SelfImprovementEngine._test_orphan_modules(dummy, ["b.py"])  # path-ignore
    assert res == set()

    data = json.loads((data_dir / "orphan_modules.json").read_text())
    assert data["b.py"]["reason"] == "heavy_side_effects"  # path-ignore

