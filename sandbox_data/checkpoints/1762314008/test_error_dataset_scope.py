import json
import sys
import types

# Provide a lightweight stub for the heavy data_bot dependency.
data_bot_stub = types.ModuleType("menace_sandbox.data_bot")
class _MetricsDB:  # pragma: no cover - simple stub
    pass
class _DataBot:  # pragma: no cover - simple stub
    pass
data_bot_stub.MetricsDB = _MetricsDB
data_bot_stub.DataBot = _DataBot
sys.modules.setdefault("menace_sandbox.data_bot", data_bot_stub)
sys.modules.setdefault("data_bot", data_bot_stub)

from menace_sandbox.error_bot import ErrorDB
from menace_sandbox.micro_models.error_dataset import export_dataset


def test_export_dataset_scope_filters(tmp_path):
    db_path = tmp_path / "errors.db"
    db = ErrorDB(db_path)
    db.conn.execute(
        "INSERT INTO telemetry(stack_trace, cause, error_type, source_menace_id) VALUES (?,?,?,?)",
        ("trace1", "cause1", "type1", "m1"),
    )
    db.conn.execute(
        "INSERT INTO telemetry(stack_trace, cause, error_type, source_menace_id) VALUES (?,?,?,?)",
        ("trace2", "cause2", "type2", "m2"),
    )
    db.conn.commit()

    out = tmp_path / "local.jsonl"
    count = export_dataset(db_path, out, scope="local", source_menace_id="m1")
    assert count == 1
    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert records[0]["stack_trace"] == "trace1"

    out2 = tmp_path / "global.jsonl"
    count = export_dataset(db_path, out2, scope="global", source_menace_id="m1")
    assert count == 1
    records = [json.loads(line) for line in out2.read_text().splitlines()]
    assert records[0]["stack_trace"] == "trace2"
