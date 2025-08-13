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
def flagger(monkeypatch):
    def fake_flag_violations(entry):
        code = entry.get("generated_code", "")
        if "eval" in code:
            return {
                "violations": [
                    {
                        "field": "generated_code",
                        "category": "unsafe",
                        "matched_keyword": "eval",
                    }
                ],
                "severity": 5,
            }
        return {"violations": [], "severity": 0}

    monkeypatch.setattr(haf, "flag_violations", fake_flag_violations)
    return haf.HumanAlignmentFlagger()


def test_unsafe_patch_triggers_warning(unsafe_patch, flagger):
    report = flagger.flag_patch(unsafe_patch, {})
    assert any("eval" in issue["message"] for issue in report["issues"])


def test_clean_patch_has_no_warnings(clean_patch, flagger):
    report = flagger.flag_patch(clean_patch, {})
    assert report["issues"] == []
