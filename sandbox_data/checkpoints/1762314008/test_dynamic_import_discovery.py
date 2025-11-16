import textwrap

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans


def test_dynamic_import_detection(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            """
            import importlib

            def load():
                importlib.import_module("dmod1")
                __import__("dmod2")
            """
        )
    )
    (tmp_path / "dmod1.py").write_text("x = 1\n")  # path-ignore
    (tmp_path / "dmod2.py").write_text("y = 2\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod1"]["parents"] == ["dynamic_source"]
    assert mapping["dmod2"]["parents"] == ["dynamic_source"]
