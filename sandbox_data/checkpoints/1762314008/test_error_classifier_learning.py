import os
import sys
import types
import yaml
from datetime import datetime

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
sys.modules.setdefault(
    "menace.data_bot", types.SimpleNamespace(MetricsDB=object, DataBot=object)
)

import menace.error_logger as elog  # noqa: E402
from menace.error_logger import ErrorClassifier, TelemetryEvent  # noqa: E402
from menace.error_ontology import ErrorType  # noqa: E402
from menace.error_bot import ErrorDB  # noqa: E402

import pytest  # noqa: E402


@pytest.mark.parametrize("scope, src", [("local", None)])
def test_classifier_updates_from_db(tmp_path, monkeypatch, scope, src):
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
        db.add_telemetry(event, source_menace_id=src)

    assert "special failure" not in clf.semantic_map
    clf.learn_error_phrases(db, min_count=3, scope=scope)
    assert "special failure" in clf.semantic_map

    data = yaml.safe_load(cfg_path.read_text())
    assert "special failure" in data["SemanticBug"]["semantic"]
