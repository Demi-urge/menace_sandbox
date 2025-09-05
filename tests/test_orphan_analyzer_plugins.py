from __future__ import annotations

from pathlib import Path
import sys
import importlib.metadata as metadata


def _candidate_module(tmp_path: Path) -> Path:
    mod = tmp_path / "mod.py"  # path-ignore
    mod.write_text('"""doc"""\n')
    return mod


def test_env_var_classifier(monkeypatch, tmp_path):
    mod = _candidate_module(tmp_path)
    plugin = tmp_path / "plugin_env.py"  # path-ignore
    plugin.write_text(
        "def classify(path, metrics):\n    return 'legacy'\n"
    )

    monkeypatch.syspath_prepend(tmp_path)
    sys.modules.pop("orphan_analyzer", None)
    import orphan_analyzer
    assert orphan_analyzer.classify_module(mod) == "candidate"

    monkeypatch.setenv("ORPHAN_ANALYZER_CLASSIFIERS", "plugin_env:classify")
    sys.modules.pop("orphan_analyzer", None)
    import orphan_analyzer

    assert orphan_analyzer.classify_module(mod) == "legacy"


def test_entry_point_classifier(monkeypatch, tmp_path):
    mod = _candidate_module(tmp_path)

    sys.modules.pop("orphan_analyzer", None)
    import orphan_analyzer
    assert orphan_analyzer.classify_module(mod) == "candidate"

    def ep_classifier(path: Path, metrics: dict) -> str:
        return "legacy"

    class DummyEP:
        def load(self):  # pragma: no cover - trivial
            return ep_classifier

    def fake_entry_points(*, group: str):
        if group == "orphan_analyzer.classifiers":
            return [DummyEP()]
        return []

    monkeypatch.setattr(metadata, "entry_points", fake_entry_points)
    sys.modules.pop("orphan_analyzer", None)
    import orphan_analyzer

    assert orphan_analyzer.classify_module(mod) == "legacy"
