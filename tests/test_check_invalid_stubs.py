from __future__ import annotations

import ast

from scripts.check_invalid_stubs import _ModuleAnalyzer


def _codes(source: str) -> set[str]:
    tree = ast.parse(source)
    findings = _ModuleAnalyzer("sample.py", tree).run()
    return {f.code for f in findings}


def test_flags_none_called_without_guard() -> None:
    source = """
SVC = None


def run():
    return SVC()
"""
    assert "none_called" in _codes(source)


def test_allows_guarded_none_call() -> None:
    source = """
SVC = None


def run():
    if SVC is not None:
        return SVC()
    return None
"""
    assert "none_called" not in _codes(source)


def test_flags_object_and_simple_namespace_shims() -> None:
    source = """
from types import SimpleNamespace

SVC = object
BROKER = SimpleNamespace(send=lambda *_a, **_k: None)
"""
    codes = _codes(source)
    assert "object_fallback" in codes
    assert "simple_namespace_shim" in codes


def test_flags_module_level_branch_pass_when_shim_present() -> None:
    source = """
SVC = None

if True:
    pass
"""
    assert "bare_pass" in _codes(source)
