import json

import json
import os
import time
import menace.error_logger as elog
from menace.error_logger import ErrorClassifier
from menace.error_ontology import ErrorType


def test_classifier_identifies_taxonomy(monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    clf = ErrorClassifier()
    assert clf.classify("Traceback...\nKeyError: boom") is ErrorType.RUNTIME_FAULT
    assert clf.classify("something went wrong\ndependency missing foo") is ErrorType.DEPENDENCY_MISMATCH
    assert clf.classify("Traceback...\nAssertionError: nope") is ErrorType.LOGIC_MISFIRE
    assert clf.classify("operation failed due to unexpected type") is ErrorType.SEMANTIC_BUG
    assert clf.classify("ZeroDivisionError: divide by zero") is ErrorType.RUNTIME_FAULT
    assert clf.classify("cannot import name foo") is ErrorType.DEPENDENCY_MISMATCH
    assert clf.classify("MemoryError: cannot allocate") is ErrorType.RESOURCE_LIMIT
    assert clf.classify("request timed out after 5s") is ErrorType.TIMEOUT
    assert clf.classify("ConnectionError: external api down") is ErrorType.EXTERNAL_API


def test_classifier_custom_rules(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    cfg = {
        "SemanticBug": {
            "regex": ["ValueError"],
            "semantic": ["invalid value"],
        }
    }
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(cfg))
    clf = ErrorClassifier(config_path=str(path))
    assert clf.classify("ValueError: bad data") is ErrorType.SEMANTIC_BUG
    assert clf.classify("totally invalid value for field") is ErrorType.SEMANTIC_BUG


def test_classifier_reloads_on_change(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    cfg = {"SemanticBug": {"regex": ["ValueError"]}}
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(cfg))
    clf = ErrorClassifier(config_path=str(path))
    assert clf.classify("ValueError: nope") is ErrorType.SEMANTIC_BUG
    cfg = {"RuntimeFault": {"regex": ["ValueError"]}}
    path.write_text(json.dumps(cfg))
    os.utime(path, (path.stat().st_atime + 1, path.stat().st_mtime + 1))
    assert clf.classify("ValueError: nope") is ErrorType.RUNTIME_FAULT
