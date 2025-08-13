import ast
import menace.human_alignment_flagger as haf
import pytest


@pytest.fixture
def unsafe_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1 @@
+eval(input())
"""
    )


@pytest.fixture
def clean_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -0,0 +1 @@
+print('hello')
"""
    )


@pytest.fixture
def mock_analyzers(monkeypatch):
    """Replace external analyzers with lightweight mocks."""

    def fake_flag_unsafe_patterns(tree: ast.AST):
        has_eval = any(
            isinstance(node, ast.Call) and getattr(node.func, "id", "") == "eval"
            for node in ast.walk(tree)
        )
        if has_eval:
            return [{"message": "eval on input()"}]
        return []

    monkeypatch.setattr(haf, "flag_unsafe_patterns", fake_flag_unsafe_patterns)
    monkeypatch.setattr(haf, "flag_violations", lambda metadata: {"violations": []})
    monkeypatch.setattr(
        haf,
        "_static_metrics",
        lambda path: {"docstring": True, "complexity": 0},
    )


def test_unsafe_patch_triggers_warning(unsafe_patch, mock_analyzers):
    warnings = haf.flag_alignment_risks(unsafe_patch, {})
    assert warnings == ["Unsafe code pattern: eval on input()."]


def test_clean_patch_has_no_warnings(clean_patch, mock_analyzers):
    warnings = haf.flag_alignment_risks(clean_patch, {})
    assert warnings == []
