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
