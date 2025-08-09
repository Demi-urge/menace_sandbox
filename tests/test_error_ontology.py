import pytest

from menace.error_ontology import (
    ErrorCategory,
    classify_exception,
)


def test_classify_by_exception_type():
    err = KeyError("missing")
    stack = "Traceback...\nKeyError: missing"
    assert classify_exception(err, stack) is ErrorCategory.RuntimeFault


def test_classify_by_keyword():
    err = Exception("boom")
    stack = "operation failed due to dependency missing foo"
    assert classify_exception(err, stack) is ErrorCategory.DependencyMismatch


def test_classify_by_module():
    class ImportLibError(Exception):
        __module__ = "importlib"

    err = ImportLibError("bad import")
    assert classify_exception(err, "") is ErrorCategory.DependencyMismatch


def test_classify_unknown():
    err = Exception("something")
    assert classify_exception(err, "") is ErrorCategory.Unknown

