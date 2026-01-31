import json
import os
import time
import menace.error_logger as elog
from menace.error_logger import ErrorClassifier
from menace.error_ontology import ErrorCategory, ErrorType, classify_error


def test_classifier_identifies_taxonomy(monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    clf = ErrorClassifier()
    assert clf.classify("Traceback...\nKeyError: boom") is ErrorType.EdgeCaseFailure
    assert (
        clf.classify("something went wrong\ndependency missing foo")
        is ErrorType.ImportError
    )
    assert clf.classify("Traceback...\nAssertionError: nope") is ErrorType.ContractViolation
    assert (
        clf.classify("operation failed due to unexpected type")
        is ErrorType.TypeErrorMismatch
    )
    assert (
        clf.classify("ZeroDivisionError: divide by zero")
        is ErrorType.UnhandledException
    )
    assert clf.classify("cannot import name foo") is ErrorType.ImportError
    assert clf.classify("MemoryError: cannot allocate") is ErrorType.UnhandledException
    assert clf.classify("request timed out after 5s") is ErrorType.UnhandledException
    assert (
        clf.classify("ConnectionError: external api down")
        is ErrorType.UnhandledException
    )


def test_classifier_custom_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    cfg = {
        "InvalidInput": {
            "regex": ["ValueError"],
            "semantic": ["invalid value"],
        }
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(cfg))
    clf = ErrorClassifier(config_path=str(path))
    assert clf.classify("ValueError: bad data") is ErrorType.InvalidInput
    assert clf.classify("totally invalid value for field") is ErrorType.InvalidInput


def test_classifier_reloads_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    cfg = {"InvalidInput": {"regex": ["ValueError"]}}
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(cfg))
    clf = ErrorClassifier(config_path=str(path))
    assert clf.classify("ValueError: nope") is ErrorType.InvalidInput
    cfg = {"EdgeCaseFailure": {"regex": ["ValueError"]}}
    path.write_text(json.dumps(cfg))
    os.utime(path, (path.stat().st_atime + 1, path.stat().st_mtime + 1))
    assert clf.classify("ValueError: nope") is ErrorType.EdgeCaseFailure


def test_classify_error_schema_and_category():
    result = classify_error("type error: cannot add")
    assert set(result.keys()) == {"status", "data", "errors", "meta"}
    assert result["data"]["category"] == ErrorCategory.TypeErrorMismatch.value
