import json
import types
import sys
import logging


def test_engine_skips_heavy_side_effects(monkeypatch, tmp_path):
    import menace_sandbox.self_improvement as sie

    mod = tmp_path / "a.py"
    mod.write_text("print('hi')")
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
        return "candidate", {"side_effects": 11}

    monkeypatch.setattr(sie, "classify_module", fake_classify)

    class DummyLogger:
        def info(self, *a, **k):
            pass

        def exception(self, *a, **k):
            pass

    dummy = types.SimpleNamespace(logger=DummyLogger())
    res = sie.SelfImprovementEngine._test_orphan_modules(dummy, ["a.py"])
    assert res == set()

    data = json.loads((data_dir / "orphan_modules.json").read_text())
    assert data["a.py"]["reason"] == "heavy_side_effects"

