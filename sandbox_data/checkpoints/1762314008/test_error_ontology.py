import pytest

from urllib.error import HTTPError

from menace.error_ontology import (
    ErrorCategory,
    classify_exception,
)
from menace.sandbox_recovery_manager import SandboxRecoveryError


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


def test_permission_http_and_sandbox_exception_types():
    assert classify_exception(PermissionError("no access"), "") is ErrorCategory.RuntimeFault

    http_err = HTTPError("http://x", 404, "not found", hdrs=None, fp=None)
    assert classify_exception(http_err, "") is ErrorCategory.ExternalAPI

    sbe = SandboxRecoveryError("boom")
    assert classify_exception(sbe, "") is ErrorCategory.RuntimeFault


def test_classify_by_keyword():
    err = Exception("boom")
    stack = "operation failed due to dependency missing foo"
    assert classify_exception(err, stack) is ErrorCategory.DependencyMismatch


def test_keyword_new_categories():
    err = Exception("boom")
    assert (
        classify_exception(err, "processing halted, out of memory")
        is ErrorCategory.ResourceLimit
    )
    assert (
        classify_exception(err, "request timed out after 5s")
        is ErrorCategory.Timeout
    )
    assert (
        classify_exception(err, "external api returned 500")
        is ErrorCategory.ExternalAPI
    )


def test_new_keyword_categories():
    err = Exception("boom")
    assert (
        classify_exception(err, "file write failed: permission denied")
        is ErrorCategory.RuntimeFault
    )
    assert (
        classify_exception(err, "remote server replied 403 forbidden")
        is ErrorCategory.ExternalAPI
    )
    assert (
        classify_exception(err, "cuda error: device lost")
        is ErrorCategory.ResourceLimit
    )
    assert (
        classify_exception(err, "accelerator error detected")
        is ErrorCategory.ResourceLimit
    )


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


def test_new_categories_by_type_and_module():
    assert classify_exception(MemoryError("oom"), "") is ErrorCategory.ResourceLimit
    assert classify_exception(TimeoutError("slow"), "") is ErrorCategory.Timeout

    class RequestsError(Exception):
        __module__ = "requests"

    err = RequestsError("bad response")
    assert classify_exception(err, "") is ErrorCategory.ExternalAPI


def test_classify_unknown():
    err = Exception("something")
    assert classify_exception(err, "") is ErrorCategory.Unknown

