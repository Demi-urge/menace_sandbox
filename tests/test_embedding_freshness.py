import json
import logging
import time
from pathlib import Path
import sys
import types

import pytest


async def _noop_schedule_backfill(*, dbs=None):
    return None


def test_ensure_embeddings_fresh_logs_and_raises(monkeypatch, tmp_path, caplog):
    # Stub heavy dependencies before import
    trans_mod = types.ModuleType("transformers")
    trans_mod.AutoModel = object
    trans_mod.AutoTokenizer = object
    sys.modules["transformers"] = trans_mod
    monkeypatch.setattr("dynamic_path_router.resolve_path", lambda p: Path(tmp_path / p))

    import vector_service.embedding_backfill as eb

    monkeypatch.setattr(eb, "_TIMESTAMP_FILE", Path(tmp_path / "ts.json"))
    monkeypatch.setattr(eb, "_load_registry", lambda path=None: {})
    monkeypatch.setattr(eb, "schedule_backfill", _noop_schedule_backfill)

    db_path = Path(tmp_path / "dummy.db")
    db_path.write_text("x")

    ts = time.time() + 100
    eb._TIMESTAMP_FILE.write_text(json.dumps({"dummy": ts}))

    with caplog.at_level(logging.ERROR):
        with pytest.raises(eb.StaleEmbeddingsError) as exc:
            eb.ensure_embeddings_fresh(["dummy"], retries=1, delay=0)

    assert exc.value.stale_dbs == {"dummy": "embedding metadata missing"}
    assert "dummy (embedding metadata missing)" in caplog.text

