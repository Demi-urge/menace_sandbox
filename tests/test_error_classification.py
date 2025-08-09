import json

from menace.error_logger import ErrorClassifier
from menace.error_ontology import ErrorType


def test_classifier_identifies_taxonomy():
    clf = ErrorClassifier()
    assert clf.classify("Traceback...\nKeyError: boom") is ErrorType.RUNTIME_FAULT
    assert clf.classify("something went wrong\ndependency missing foo") is ErrorType.DEPENDENCY_MISMATCH
    assert clf.classify("Traceback...\nAssertionError: nope") is ErrorType.LOGIC_MISFIRE
    assert clf.classify("operation failed due to unexpected type") is ErrorType.SEMANTIC_BUG


def test_classifier_custom_rules(tmp_path):
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
