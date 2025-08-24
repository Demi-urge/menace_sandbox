import pytest
import sys
import types


class _EmbeddableStub:
    def __init__(self, *a, **k):
        pass

    def add_embedding(self, *a, **k):  # pragma: no cover - simple stub
        pass

    def encode_text(self, text):  # pragma: no cover - simple stub
        return [0.0]


sys.modules.setdefault(
    "vector_service", types.SimpleNamespace(EmbeddableDBMixin=_EmbeddableStub)
)
sys.modules.setdefault(
    "auto_link", types.SimpleNamespace(auto_link=lambda *_a, **_k: (lambda f: f))
)
sys.modules.setdefault(
    "unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
sys.modules.setdefault(
    "retry_utils",
    types.SimpleNamespace(
        publish_with_retry=lambda *a, **k: True,
        with_retry=lambda func, **k: func(),
    ),
)
sys.modules.setdefault(
    "alert_dispatcher",
    types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={})
)

import menace.code_database as cdbm  # noqa: E402


def test_code_fts_search(tmp_path):
    db = cdbm.CodeDB(tmp_path / "c.db")
    if not getattr(db, "has_fts", False):
        pytest.skip("fts5 not available")
    rec1 = cdbm.CodeRecord(code="print('alpha')", summary="first alpha")
    db.add(rec1)
    rec2 = cdbm.CodeRecord(code="print('beta')", summary="second beta")
    db.add(rec2)
    res = db.search("alpha")
    summaries = [r["summary"] for r in res]
    assert "first alpha" in summaries
