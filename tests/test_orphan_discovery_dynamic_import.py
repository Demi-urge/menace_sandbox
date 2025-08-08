import textwrap

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans


def test_variable_and_fstring_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(
        textwrap.dedent(
            '''
            import importlib

            name = "dmod1"
            prefix = "dmod"

            def load():
                importlib.import_module(name)
                __import__(f"{prefix}2")
            '''
        )
    )
    (tmp_path / "dmod1.py").write_text("x = 1\n")
    (tmp_path / "dmod2.py").write_text("y = 2\n")

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod1"]["parents"] == ["dynamic_source"]
    assert mapping["dmod2"]["parents"] == ["dynamic_source"]


def test_tuple_and_format_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(
        textwrap.dedent(
            '''
            import importlib

            parts = ("dmod", "3")
            prefix = "dmod"

            def load():
                importlib.import_module("".join(parts))
                importlib.import_module("{}4".format(prefix))
            '''
        )
    )
    (tmp_path / "dmod3.py").write_text("x = 3\n")
    (tmp_path / "dmod4.py").write_text("y = 4\n")

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod3"]["parents"] == ["dynamic_source"]
    assert mapping["dmod4"]["parents"] == ["dynamic_source"]
