import menace.human_alignment_flagger as haf
import pytest


@pytest.fixture
def unsafe_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1 @@
-import logging
-logging.info('hi')
+pass
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


def test_unsafe_patch_triggers_warning(unsafe_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(unsafe_patch, {})
    assert any(
        "Logging removed" in issue["message"] for issue in report["issues"]
    )


def test_clean_patch_has_no_warnings(clean_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(clean_patch, {})
    assert report["issues"] == []


def test_forbidden_keyword_in_code_improvement_triggers_warning():
    code = "def introduce_reward_hack():\n    return 'auto_reward'\n"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": "module.py", "code": code}],
        metrics={"accuracy": 0.9},
        logs=[],
    )
    assert any(
        violation.get("matched_keyword") == "auto_reward"
        for warning in warnings["ethics"]
        for violation in warning.get("violations", [])
    )


def test_complexity_without_docstring_raises_maintainability_warning():
    code = "def complex(x):\n"
    code += "    if x:\n        pass\n" * 11
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": "module.py", "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        issue.get("issue") == "high cyclomatic complexity"
        and issue.get("file") == "module.py"
        for issue in warnings["maintainability"]
    )
    assert any(
        issue.get("issue") == "missing docstring"
        and issue.get("file") == "module.py"
        for issue in warnings["maintainability"]
    )
