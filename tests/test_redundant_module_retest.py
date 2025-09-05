import json
import types
from pathlib import Path

import menace.self_improvement as sie


def test_redundant_modules_retested(monkeypatch, tmp_path):
    repo = tmp_path
    (repo / "red.py").write_text("x = 1\n")  # path-ignore
    data_dir = repo / "sandbox_data"
    data_dir.mkdir()
    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(data_dir))
    (data_dir / "orphan_modules.json").write_text(json.dumps(["red.py"]))  # path-ignore
    (data_dir / "orphan_classifications.json").write_text(
        json.dumps({"red.py": {"classification": "redundant"}})  # path-ignore
    )

    calls: dict[str, object] = {}

    def fake_auto_include(mods, recursive=False, validate=False):
        calls["mods"] = list(mods)
        calls["validate"] = validate

    monkeypatch.setattr(sie.environment, "auto_include_modules", fake_auto_include)

    eng = types.SimpleNamespace(
        logger=types.SimpleNamespace(
            info=lambda *a, **k: None, exception=lambda *a, **k: None
        )
    )
    eng.retest_redundant_modules = types.MethodType(
        sie.SelfImprovementEngine.retest_redundant_modules, eng
    )

    eng.retest_redundant_modules()

    assert calls.get("mods") == ["red.py"]  # path-ignore
    assert calls.get("validate") is True
