import yaml
from datetime import datetime

import menace.error_logger as elog
from menace.error_logger import ErrorClassifier, TelemetryEvent
from menace.error_ontology import ErrorType
from menace.error_bot import ErrorDB


def test_classifier_updates_from_db(tmp_path, monkeypatch):
    monkeypatch.setattr(elog, "get_embedder", lambda: None)
    db = ErrorDB(path=tmp_path / "errors.db")
    cfg_path = tmp_path / "rules.yaml"
    clf = ErrorClassifier(config_path=str(cfg_path))

    event = TelemetryEvent(
        task_id=None,
        bot_id=None,
        error_type=ErrorType.SEMANTIC_BUG,
        category=ErrorType.SEMANTIC_BUG,
        root_cause="special failure",
        stack_trace="Traceback... special failure",
        root_module="mod",
        module="mod",
        module_counts={"mod": 1},
        inferred_cause="special failure",
        timestamp=datetime.utcnow().isoformat(),
        resolution_status="unresolved",
        patch_id=None,
        deploy_id=None,
    )

    for _ in range(5):
        db.add_telemetry(event)

    assert "special failure" not in clf.semantic_map
    clf.update_rules_from_db(db, min_count=3)
    assert "special failure" in clf.semantic_map

    data = yaml.safe_load(cfg_path.read_text())
    assert "special failure" in data["SemanticBug"]["semantic"]

