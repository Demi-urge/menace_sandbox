from __future__ import annotations

import ast
from types import SimpleNamespace

from tools.qa import check_invalid_stubs


def _findings(source: str):
    tree = ast.parse(source)
    return check_invalid_stubs.ModuleAnalyzer("sample.py", tree).run()


def _rules(source: str) -> set[str]:
    return {f.rule for f in _findings(source)}


def test_flags_placeholder_assignments_for_runtime_symbols() -> None:
    source = """
from types import SimpleNamespace

API_CLIENT = None
WORKER_SERVICE = object
BROKER = SimpleNamespace(send=lambda *_a, **_k: None)
"""
    rules = _rules(source)
    assert "placeholder_none" in rules
    assert "placeholder_object" in rules
    assert "placeholder_simple_namespace" in rules


def test_allows_non_runtime_private_assignments() -> None:
    source = """
from types import SimpleNamespace

_internal = None
helper = SimpleNamespace(run=lambda: 1)
"""
    assert _rules(source) == set()


def test_flags_pass_only_method_in_fallback_shim_class() -> None:
    source = """
class CacheFallbackShim:
    def fetch(self):
        pass
"""
    assert "shim_method_pass" in _rules(source)


def test_flags_dict_add_container_mismatch() -> None:
    source = """
def run():
    registry = {}
    registry.add("x")
"""
    assert "container_mismatch_add_on_dict" in _rules(source)


def test_ignores_tests_and_fixtures_paths(tmp_path, monkeypatch) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "fixtures").mkdir()
    (tmp_path / "pkg" / "runtime.py").write_text("API_CLIENT = None\n", encoding="utf-8")
    (tmp_path / "tests" / "test_runtime.py").write_text("API_CLIENT = None\n", encoding="utf-8")
    (tmp_path / "fixtures" / "runtime_fixture.py").write_text("API_CLIENT = None\n", encoding="utf-8")

    monkeypatch.setattr(check_invalid_stubs, "ROOT", tmp_path)

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout="pkg/runtime.py\ntests/test_runtime.py\nfixtures/runtime_fixture.py\n")

    monkeypatch.setattr(check_invalid_stubs.subprocess, "run", _fake_run)

    files = check_invalid_stubs._iter_tracked_python_files(["."])
    assert [f.relative_to(tmp_path).as_posix() for f in files] == ["pkg/runtime.py"]
