import pytest

from menace.error_ontology import (
    ErrorCategory,
    classify_exception,
)


def test_classify_by_exception_type():
    err = KeyError("missing")
    stack = "Traceback...\nKeyError: missing"
    assert classify_exception(err, stack) is ErrorCategory.RuntimeFault


def test_extended_exception_types():
    assert (
        classify_exception(ZeroDivisionError("bad divide"), "")
        is ErrorCategory.RuntimeFault
    )
    assert (
        classify_exception(AttributeError("no attr"), "")
        is ErrorCategory.RuntimeFault
    )
    assert classify_exception(OSError("lib"), "") is ErrorCategory.DependencyMismatch


def test_classify_by_keyword():
    err = Exception("boom")
    stack = "operation failed due to dependency missing foo"
    assert classify_exception(err, stack) is ErrorCategory.DependencyMismatch


def test_extended_keyword_and_module():
    err = Exception("boom")
    stack = "cannot import name xyz from foo"
    assert classify_exception(err, stack) is ErrorCategory.DependencyMismatch

    class PkgError(Exception):
        __module__ = "pkg_resources"

    err2 = PkgError("bad pkg")
    assert classify_exception(err2, "") is ErrorCategory.DependencyMismatch


def test_classify_by_module():
    class ImportLibError(Exception):
        __module__ = "importlib"

    err = ImportLibError("bad import")
    assert classify_exception(err, "") is ErrorCategory.DependencyMismatch


def test_classify_unknown():
    err = Exception("something")
    assert classify_exception(err, "") is ErrorCategory.Unknown

