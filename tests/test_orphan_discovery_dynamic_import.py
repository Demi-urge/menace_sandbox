import textwrap

import orphan_analyzer
from sandbox_runner.orphan_discovery import discover_recursive_orphans


def test_variable_and_fstring_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
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
    (tmp_path / "dmod1.py").write_text("x = 1\n")  # path-ignore
    (tmp_path / "dmod2.py").write_text("y = 2\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod1"]["parents"] == ["dynamic_source"]
    assert mapping["dmod2"]["parents"] == ["dynamic_source"]


def test_tuple_and_format_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
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
    (tmp_path / "dmod3.py").write_text("x = 3\n")  # path-ignore
    (tmp_path / "dmod4.py").write_text("y = 4\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod3"]["parents"] == ["dynamic_source"]
    assert mapping["dmod4"]["parents"] == ["dynamic_source"]


def test_concat_and_chained_format_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            '''
            import importlib

            suffix = "5"
            other = "6"

            def load():
                importlib.import_module("dmod" + suffix)
                importlib.import_module("dmod{}".format(other).lower())
            '''
        )
    )
    (tmp_path / "dmod5.py").write_text("x = 5\n")  # path-ignore
    (tmp_path / "dmod6.py").write_text("y = 6\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod5"]["parents"] == ["dynamic_source"]
    assert mapping["dmod6"]["parents"] == ["dynamic_source"]


def test_percent_format_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            '''
            import importlib

            suffix = "7"
            parts = ("dmod", "8")

            def load():
                importlib.import_module("dmod%s" % suffix)
                importlib.import_module("%s%s" % parts)
            '''
        )
    )
    (tmp_path / "dmod7.py").write_text("x = 7\n")  # path-ignore
    (tmp_path / "dmod8.py").write_text("y = 8\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod7"]["parents"] == ["dynamic_source"]
    assert mapping["dmod8"]["parents"] == ["dynamic_source"]


def test_env_var_imports(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            '''
            import importlib
            import os

            def load():
                importlib.import_module(os.getenv("MOD_NAME"))
            '''
        )
    )
    (tmp_path / "dmod_env.py").write_text("z = 9\n")  # path-ignore

    monkeypatch.setenv("MOD_NAME", "dmod_env")
    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod_env"]["parents"] == ["dynamic_source"]


def test_spec_from_file_location(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            '''
            import importlib.util

            def load():
                spec = importlib.util.spec_from_file_location("dmod_spec", "dmod_spec.py")  # path-ignore
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            '''
        )
    )
    (tmp_path / "dmod_spec.py").write_text("x = 10\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod_spec"]["parents"] == ["dynamic_source"]


def test_source_file_loader(tmp_path, monkeypatch):
    (tmp_path / "dynamic_source.py").write_text(  # path-ignore
        textwrap.dedent(
            '''
            import importlib.machinery

            def load():
                importlib.machinery.SourceFileLoader("dmod_loader", "dmod_loader.py").load_module()  # path-ignore
            '''
        )
    )
    (tmp_path / "dmod_loader.py").write_text("x = 11\n")  # path-ignore

    monkeypatch.setattr(orphan_analyzer, "classify_module", lambda p: "candidate")

    mapping = discover_recursive_orphans(str(tmp_path))
    assert mapping["dynamic_source"]["parents"] == []
    assert mapping["dmod_loader"]["parents"] == ["dynamic_source"]
