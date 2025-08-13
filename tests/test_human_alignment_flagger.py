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


@pytest.fixture
def comment_removal_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,2 +1 @@
-# explanatory comment
-x = compute()
+x = compute()
"""
    )


@pytest.fixture
def obfuscation_patch() -> str:
    return (
        """diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -1,2 +1,2 @@
--result = do()
--return result
+r = do()
+return r
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


def test_comment_removal_triggers_opacity(comment_removal_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(comment_removal_patch, {})
    assert any("Opacity" in issue["message"] for issue in report["issues"])


def test_obfuscation_triggers_opacity(obfuscation_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(obfuscation_patch, {})
    assert any("Opacity" in issue["message"] for issue in report["issues"])


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


# ---------------------------------------------------------------------------
# flag_alignment_issues -----------------------------------------------------


@pytest.fixture
def eval_diff() -> dict:
    return {"module.py": {"added": ["result = eval('2+2')"], "removed": []}}


@pytest.fixture
def removed_logging_diff() -> dict:
    return {
        "module.py": {
            "added": ["def process():", "    return 1"],
            "removed": [
                "def process():",
                "    logging.info('start')",
                "    return 1",
            ],
        }
    }


@pytest.fixture
def high_complexity_diff() -> dict:
    lines = [
        "def complex():",
        "    if a and b or c:",
        "        for i in range(10):",
        "            if i % 2 and i % 3 or i % 5:",
        "                while x:",
        "                    try:",
        "                        pass",
        "                    except Exception:",
        "                        if z:",
        "                            pass",
        "    elif d or e:",
        "        pass",
    ]
    return {"complex.py": {"added": lines, "removed": []}}


@pytest.fixture
def clean_diff() -> dict:
    return {"clean.py": {"added": ["x = 1"], "removed": []}}


@pytest.fixture
def comment_removal_diff() -> dict:
    return {"foo.py": {"added": ["x = compute()"], "removed": ["# note", "x = compute()"]}}


@pytest.fixture
def obfuscation_diff() -> dict:
    return {
        "bar.py": {
            "added": ["r = do()", "return r"],
            "removed": ["result = do()", "return result"],
        }
    }


def test_flag_alignment_issues_eval(eval_diff):
    findings = haf.flag_alignment_issues(eval_diff)
    assert any(f.get("category") == "risky_construct" for f in findings)


def test_flag_alignment_issues_removed_logging(removed_logging_diff):
    findings = haf.flag_alignment_issues(removed_logging_diff)
    assert any(f.get("category") == "missing_logging" for f in findings)


def test_flag_alignment_issues_high_complexity(high_complexity_diff):
    findings = haf.flag_alignment_issues(high_complexity_diff)
    assert any(f.get("category") == "high_complexity" for f in findings)


def test_flag_alignment_issues_clean(clean_diff):
    findings = haf.flag_alignment_issues(clean_diff)
    assert findings == []


def test_flag_alignment_issues_comment_density(comment_removal_diff):
    findings = haf.flag_alignment_issues(comment_removal_diff)
    assert any(
        f.get("category") == "opacity" and "comment" in f.get("snippet", "")
        for f in findings
    )


def test_flag_alignment_issues_obfuscation(obfuscation_diff):
    findings = haf.flag_alignment_issues(obfuscation_diff)
    assert any(
        f.get("category") == "opacity" and "identifier" in f.get("snippet", "")
        for f in findings
    )
