import menace.human_alignment_flagger as haf
import pytest
from sandbox_settings import AlignmentRules, SandboxSettings

MODULE = "module" + ".py"  # path-ignore
WORKFLOW = "workflow" + ".py"  # path-ignore
COMPLEX = "complex" + ".py"  # path-ignore
CLEAN = "clean" + ".py"  # path-ignore
FOO = "foo" + ".py"  # path-ignore
BAR = "bar" + ".py"  # path-ignore
RUN = "run" + ".py"  # path-ignore
FS = "fs" + ".py"  # path-ignore
LINT = "lint" + ".py"  # path-ignore
BE = "be" + ".py"  # path-ignore
COMP = "comp" + ".py"  # path-ignore
DYN = "dyn" + ".py"  # path-ignore
TYPES = "types" + ".py"  # path-ignore


@pytest.fixture
def unsafe_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py  # path-ignore
--- a/foo.py  # path-ignore
+++ b/foo.py  # path-ignore
@@ -1,2 +1 @@
-import logging
-logging.info('hi')
+pass
"""
    )


@pytest.fixture
def clean_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py  # path-ignore
--- a/foo.py  # path-ignore
+++ b/foo.py  # path-ignore
@@ -0,0 +1 @@
+print('hello')
"""
    )


@pytest.fixture
def comment_removal_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py  # path-ignore
--- a/foo.py  # path-ignore
+++ b/foo.py  # path-ignore
@@ -1,2 +1 @@
-# explanatory comment
-x = compute()
+x = compute()
"""
    )


@pytest.fixture
def obfuscation_patch() -> str:
    return (
        """diff --git a/bar.py b/bar.py  # path-ignore
--- a/bar.py  # path-ignore
+++ b/bar.py  # path-ignore
@@ -1,2 +1,2 @@
--result = do()
--return result
+r = do()
+return r
"""
    )


@pytest.fixture
def exec_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py  # path-ignore
--- a/foo.py  # path-ignore
+++ b/foo.py  # path-ignore
@@ -0,0 +1 @@
+exec('print(1)')
"""
    )


@pytest.fixture
def network_patch() -> str:
    return (
        """diff --git a/foo.py b/foo.py  # path-ignore
--- a/foo.py  # path-ignore
+++ b/foo.py  # path-ignore
@@ -0,0 +2 @@
+import requests
+requests.get('http://example.com')
"""
    )


@pytest.fixture
def unsandboxed_fs_patch() -> str:
    return (
        """diff --git a/fs.py b/fs.py  # path-ignore
--- a/fs.py  # path-ignore
+++ b/fs.py  # path-ignore
@@ -0,0 +1 @@
+open('/etc/passwd', 'w')
"""
    )


@pytest.fixture
def linter_patch() -> str:
    return (
        """diff --git a/lint.py b/lint.py  # path-ignore
--- a/lint.py  # path-ignore
+++ b/lint.py  # path-ignore
@@ -0,0 +1 @@
+x = 1  # noqa
"""
    )


@pytest.fixture
def subprocess_patch() -> str:
    return (
        """diff --git a/run.py b/run.py  # path-ignore
--- a/run.py  # path-ignore
+++ b/run.py  # path-ignore
@@ -0,0 +2 @@
+import subprocess
+subprocess.run('ls', shell=True)
"""
    )


@pytest.fixture
def semantic_subprocess_patch() -> str:
    return (
        """diff --git a/run.py b/run.py  # path-ignore
--- a/run.py  # path-ignore
+++ b/run.py  # path-ignore
@@ -0,0 +2 @@
+import subprocess
+subprocess.call('ls', shell=True)
"""
    )


@pytest.fixture
def missing_type_hints_patch() -> str:
    return (
        """diff --git a/types.py b/types.py  # path-ignore
--- a/types.py  # path-ignore
+++ b/types.py  # path-ignore
@@ -0,0 +2 @@
+def add(a, b):
+    return a + b
"""
    )


@pytest.fixture
def high_complexity_patch() -> str:
    return (
        """diff --git a/complex.py b/complex.py  # path-ignore
--- a/complex.py  # path-ignore
+++ b/complex.py  # path-ignore
@@ -0,0 +12 @@
+def complex(x):
+    if a:
+        if b:
+            if c:
+                if d:
+                    if e:
+                        if f:
+                            if g:
+                                if h:
+                                    if i:
+                                        if j:
+                                            if k:
+                                                return 1
"""
    )


@pytest.fixture
def todo_patch() -> str:
    return (
        """diff --git a/todo.py b/todo.py  # path-ignore
--- a/todo.py  # path-ignore
+++ b/todo.py  # path-ignore
@@ -0,0 +1 @@
# TODO: fix
"""
    )


@pytest.fixture
def yaml_security_patch() -> str:
    return (
        """diff --git a/config.yaml b/config.yaml
--- a/config.yaml
+++ b/config.yaml
@@ -0,0 +1 @@
+allow_privilege_escalation: true
"""
    )


@pytest.fixture
def json_credential_patch() -> str:
    return (
        """diff --git a/config.json b/config.json
--- a/config.json
+++ b/config.json
@@ -0,0 +1 @@
+{"password": "secret"}
"""
    )


@pytest.fixture
def secret_diff_patch() -> str:
    return (
        """diff --git a/secrets.py b/secrets.py  # path-ignore
--- a/secrets.py  # path-ignore
+++ b/secrets.py  # path-ignore
@@ -0,0 +1 @@
+password = \"supersecret\"
"""
    )


@pytest.fixture
def license_diff_patch() -> str:
    return (
        """diff --git a/LICENSE.txt b/LICENSE.txt
--- a/LICENSE.txt
+++ b/LICENSE.txt
@@ -0,0 +1 @@
+This code is released under the GNU General Public License
"""
    )


@pytest.fixture
def shell_exec_flag_patch() -> str:
    return (
        """diff --git a/run.sh b/run.sh
--- a/run.sh
+++ b/run.sh
@@ -0,0 +1 @@
+chmod +x /tmp/script.sh
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


def test_removed_comment_flagged(comment_removal_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(comment_removal_patch, {})
    assert any("Comment removed" in issue["message"] for issue in report["issues"])


def test_comment_density_threshold_override(comment_removal_patch):
    settings = SandboxSettings(
        alignment_rules=AlignmentRules(max_comment_density_drop=1.0)
    )
    flagger = haf.HumanAlignmentFlagger(settings)
    report = flagger.flag_patch(comment_removal_patch, {})
    assert not any(
        "comment density decreased" in issue["message"].lower()
        for issue in report["issues"]
    )


def test_obfuscation_triggers_opacity(obfuscation_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(obfuscation_patch, {})
    assert any("Opacity" in issue["message"] for issue in report["issues"])


def test_exec_usage_flagged(exec_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(exec_patch, {})
    assert any("Dynamic execution" in issue["message"] for issue in report["issues"])


def test_network_call_flagged(network_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(network_patch, {})
    assert any("Network call" in issue["message"] for issue in report["issues"])


def test_network_call_threshold_override(network_patch):
    settings = SandboxSettings(
        alignment_rules=AlignmentRules(allowed_network_calls=1)
    )
    flagger = haf.HumanAlignmentFlagger(settings)
    report = flagger.flag_patch(network_patch, {})
    assert not any(
        "Network call" in issue["message"] for issue in report["issues"]
    )


def test_network_call_severity_override(network_patch):
    settings = SandboxSettings(
        alignment_rules=AlignmentRules(network_call_severity=1)
    )
    flagger = haf.HumanAlignmentFlagger(settings)
    report = flagger.flag_patch(network_patch, {})
    issue = next(
        i for i in report["issues"] if "Network call" in i["message"]
    )
    assert issue["tier"] == "info"


def test_unsandboxed_fs_flagged(unsandboxed_fs_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(unsandboxed_fs_patch, {})
    assert any(
        "Unsandboxed filesystem mutation" in issue["message"] for issue in report["issues"]
    )


def test_linter_suppression_flagged(linter_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(linter_patch, {})
    assert any("Linter suppression" in issue["message"] for issue in report["issues"])


def test_unsafe_subprocess_flagged(subprocess_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(subprocess_patch, {})
    assert any("Unsafe subprocess" in issue["message"] for issue in report["issues"])


def test_semantic_subprocess_flagged(semantic_subprocess_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(semantic_subprocess_patch, {})
    assert any(
        "Semantic similarity to unsafe pattern" in issue["message"]
        for issue in report["issues"]
    )


def test_missing_type_hints_flagged(missing_type_hints_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(missing_type_hints_patch, {})
    assert any("Missing type hints" in issue["message"] for issue in report["issues"])


def test_high_complexity_flagged(high_complexity_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(high_complexity_patch, {})
    assert any("High cyclomatic complexity" in issue["message"] for issue in report["issues"])


def test_yaml_security_feature_disabled_flagged(yaml_security_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(yaml_security_patch, {})
    assert any(
        "Security feature disabled" in issue["message"]
        for issue in report["issues"]
    )


def test_json_embedded_credential_flagged(json_credential_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(json_credential_patch, {})
    assert any(
        "Embedded credential" in issue["message"] for issue in report["issues"]
    )


def test_secret_rule_detects_secret(secret_diff_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(secret_diff_patch, {})
    assert any("Secret detected" in issue["message"] for issue in report["issues"])


def test_license_rule_detects_restricted_license(license_diff_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(license_diff_patch, {})
    assert any(
        "Restricted license" in issue["message"] for issue in report["issues"]
    )


def test_shell_execution_flag_flagged(shell_exec_flag_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(shell_exec_flag_patch, {})
    assert any(
        "Execution flag" in issue["message"] for issue in report["issues"]
    )


def test_report_includes_score_and_tiers(unsafe_patch):
    flagger = haf.HumanAlignmentFlagger()
    report = flagger.flag_patch(unsafe_patch, {})
    assert report["score"] == 2
    assert report["tiers"].get("warn", 0) >= 1
    assert all("tier" in issue for issue in report["issues"])


def test_forbidden_keyword_in_code_improvement_triggers_warning():
    code = "def introduce_reward_hack():\n    return 'auto_reward'\n"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
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
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    complexity_warning = next(
        (
            issue
            for issue in warnings["maintainability"]
            if issue.get("issue") == "high cyclomatic complexity"
            and issue.get("file") == MODULE
        ),
        None,
    )
    assert complexity_warning is not None
    assert any(
        func.get("name") == "complex" and func.get("score", 0) > 10
        for func in complexity_warning.get("functions", [])
    )
    assert any(
        issue.get("issue") == "missing docstring"
        and issue.get("file") == MODULE
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_missing_type_hints():
    code = "def add(a, b):\n    return a + b\n"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    hint_warning = next(
        (
            issue
            for issue in warnings["maintainability"]
            if issue.get("issue") == "missing type hints"
            and issue.get("file") == MODULE
        ),
        None,
    )
    assert hint_warning is not None
    assert "add" in hint_warning.get("functions", [])


def test_flag_improvement_warns_on_unsafe_subprocess():
    code = "import subprocess\nsubprocess.run('ls', shell=True)"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": WORKFLOW, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        "unsafe subprocess" in issue.get("issue", "")
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_warns_on_unsandboxed_fs():
    code = "open('/etc/passwd', 'w')"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": WORKFLOW, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        "unsandboxed" in issue.get("issue", "")
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_obfuscated_names():
    code = "def f():\n    a = 1\n    b = a\n    return b"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        issue.get("issue") == "obfuscated variable names"
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_exec_call():
    code = "def f():\n    exec('print(1)')"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        "exec" in issue.get("issue", "") for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_eval_call():
    code = "def f():\n    return eval('2+2')"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        "eval" in issue.get("issue", "") for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_network_call():
    code = "import requests\nrequests.get('http://example.com')"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        "network call" in issue.get("issue", "") for issue in warnings["maintainability"]
    )


def test_flag_improvement_respects_network_call_threshold():
    code = "import requests\nrequests.get('http://example.com')"
    settings = SandboxSettings(
        alignment_rules=AlignmentRules(allowed_network_calls=1)
    )
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
        settings=settings,
    )
    assert not any(
        "network call" in issue.get("issue", "")
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_broad_except():
    code = "def f():\n    try:\n        1/0\n    except:\n        pass"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={},
        logs=[],
    )
    assert any(
        issue.get("issue") == "broad exception handler"
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_detects_removed_type_hints():
    code = "def add(a, b):\n    return a + b"
    diff = (
        "--- a/module.py\n"  # path-ignore
        "+++ b/module.py\n"  # path-ignore
        "@@\n"
        "-def add(a: int, b: int) -> int:\n"
        "-    return a + b\n"
        "+def add(a, b):\n"
        "+    return a + b\n"
    )
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code, "diff": diff}],
        metrics={},
        logs=[],
    )
    assert any(
        issue.get("issue") == "removed type hints"
        for issue in warnings["maintainability"]
    )


def test_flag_improvement_compares_baseline(tmp_path):
    baseline = tmp_path / "baseline.yaml"
    baseline.write_text("tests: 1\ncomplexity: 1\n")
    settings = SandboxSettings(alignment_baseline_metrics_path=baseline)
    code = "def f(x):\n    if x:\n        return x\n"
    warnings = haf.flag_improvement(
        workflow_changes=[{"file": MODULE, "code": code}],
        metrics={"accuracy": 0.9},
        logs=[],
        settings=settings,
    )
    issues = [w.get("issue", "") for w in warnings["maintainability"]]
    assert any("test count decreased" in issue for issue in issues)
    assert any("complexity increased" in issue for issue in issues)


# ---------------------------------------------------------------------------
# flag_alignment_issues -----------------------------------------------------


@pytest.fixture
def eval_diff() -> dict:
    return {MODULE: {"added": ["result = eval('2+2')"], "removed": []}}


@pytest.fixture
def removed_logging_diff() -> dict:
    return {
        MODULE: {
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
    return {COMPLEX: {"added": lines, "removed": []}}


@pytest.fixture
def clean_diff() -> dict:
    return {CLEAN: {"added": ["x = 1"], "removed": []}}


@pytest.fixture
def comment_removal_diff() -> dict:
    return {FOO: {"added": ["x = compute()"], "removed": ["# note", "x = compute()"]}}


@pytest.fixture
def obfuscation_diff() -> dict:
    return {
        BAR: {
            "added": ["r = do()", "return r"],
            "removed": ["result = do()", "return result"],
        }
    }


@pytest.fixture
def unsafe_subprocess_diff() -> dict:
    return {
        RUN: {
            "added": ["import subprocess", "subprocess.run('ls', shell=True)"],
            "removed": [],
        }
    }


@pytest.fixture
def unsandboxed_fs_diff() -> dict:
    return {FS: {"added": ["open('/etc/passwd', 'w')"], "removed": []}}


@pytest.fixture
def linter_suppression_diff() -> dict:
    return {LINT: {"added": ["x = 1  # noqa"], "removed": []}}


@pytest.fixture
def bare_except_diff() -> dict:
    return {
        BE: {
            "added": ["try:", "    1/0", "except:", "    pass"],
            "removed": [],
        }
    }


@pytest.fixture
def compile_diff() -> dict:
    return {
        COMP: {"added": ["code = compile('2+2', '<string>', 'eval')"], "removed": []}
    }


@pytest.fixture
def dynamic_import_diff() -> dict:
    return {DYN: {"added": ["mod = __import__('os')"], "removed": []}}


@pytest.fixture
def missing_type_hints_diff() -> dict:
    return {
        TYPES: {"added": ["def add(a, b):", "    return a + b"], "removed": []}
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


def test_flag_alignment_issues_unsafe_subprocess(unsafe_subprocess_diff):
    findings = haf.flag_alignment_issues(unsafe_subprocess_diff)
    assert any(f.get("category") == "unsafe_subprocess" for f in findings)


def test_flag_alignment_issues_unsandboxed_fs(unsandboxed_fs_diff):
    findings = haf.flag_alignment_issues(unsandboxed_fs_diff)
    assert any(f.get("category") == "filesystem_mutation" for f in findings)


def test_flag_alignment_issues_comment_removed(comment_removal_diff):
    findings = haf.flag_alignment_issues(comment_removal_diff)
    assert any(f.get("category") == "comment_removed" for f in findings)


def test_flag_alignment_issues_linter_suppression(linter_suppression_diff):
    findings = haf.flag_alignment_issues(linter_suppression_diff)
    assert any(f.get("category") == "linter_suppression" for f in findings)


def test_flag_alignment_issues_bare_except(bare_except_diff):
    findings = haf.flag_alignment_issues(bare_except_diff)
    assert any(
        f.get("category") == "risky_construct" and "bare except" in f.get("snippet", "")
        for f in findings
    )


def test_flag_alignment_issues_compile(compile_diff):
    findings = haf.flag_alignment_issues(compile_diff)
    assert any(f.get("category") == "risky_construct" for f in findings)


def test_flag_alignment_issues_dynamic_import(dynamic_import_diff):
    findings = haf.flag_alignment_issues(dynamic_import_diff)
    assert any(f.get("category") == "dynamic_import" for f in findings)


def test_flag_alignment_issues_missing_type_hints(missing_type_hints_diff):
    findings = haf.flag_alignment_issues(missing_type_hints_diff)
    assert any(f.get("category") == "missing_type_hints" for f in findings)


def test_custom_complexity_threshold_suppresses_warning(high_complexity_diff):
    settings = SandboxSettings(
        alignment_rules=AlignmentRules(max_complexity_score=50)
    )
    findings = haf.flag_alignment_issues(high_complexity_diff, settings=settings)
    assert not any(f.get("category") == "high_complexity" for f in findings)


def test_rule_callable_param(todo_patch):
    def direct_rule(path, added, removed):
        if path.endswith("todo" + ".py"):  # path-ignore
            return [{"severity": 1, "message": "direct rule"}]
        return []

    flagger = haf.HumanAlignmentFlagger(rules=[direct_rule])
    report = flagger.flag_patch(todo_patch, {})
    assert any("direct rule" in issue["message"] for issue in report["issues"])
