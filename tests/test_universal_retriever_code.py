import sys
import types


def test_code_snippet_retrieval(tmp_path, monkeypatch):
    # stub out heavy dependencies imported by CodeDB
    monkeypatch.setitem(
        sys.modules,
        "menace.unified_event_bus",
        types.SimpleNamespace(UnifiedEventBus=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.retry_utils",
        types.SimpleNamespace(
            publish_with_retry=lambda *a, **k: True,
            with_retry=lambda func, **k: func(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "menace.alert_dispatcher",
        types.SimpleNamespace(send_discord_alert=lambda *a, **k: None, CONFIG={}),
    )

    from menace.code_database import CodeDB, CodeRecord
    from menace.universal_retriever import UniversalRetriever

    code_db = CodeDB(path=tmp_path / "code.db")
    code_db.add(
        CodeRecord(
            code="print('hello')",
            summary="greeting snippet",
            complexity_score=2.0,
        )
    )

    retriever = UniversalRetriever(code_db=code_db)

    hits = retriever.retrieve("greeting", top_k=1)

    assert hits and hits[0].origin_db == "code"
    assert "print('hello')" in hits[0].metadata["code"]
    assert "complexity" in hits[0].metadata["contextual_metrics"]
