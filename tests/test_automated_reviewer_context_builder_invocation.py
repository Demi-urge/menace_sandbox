import types
import sys
import importlib


class RecordingBuilder:
    def __init__(self, *_, **__):
        self.calls = []

    def build(self, payload, **_):  # pragma: no cover - simple stub
        self.calls.append(payload)
        return "ctx"


class DummyCognitionLayer:
    def __init__(self, *, context_builder=None, **__):
        self.context_builder = context_builder


sys.modules["vector_service"] = types.SimpleNamespace(
    CognitionLayer=DummyCognitionLayer, ContextBuilder=RecordingBuilder
)
import menace_sandbox.automated_reviewer as ar
importlib.reload(ar)


def test_reviewer_uses_context_builder():
    builder = RecordingBuilder(
        bot_db="bots.db", code_db="code.db", error_db="errors.db", workflow_db="workflows.db"
    )
    db = types.SimpleNamespace(update_bot=lambda *a, **k: None)
    esc = types.SimpleNamespace(handle=lambda *a, **k: None)

    reviewer = ar.AutomatedReviewer(context_builder=builder, bot_db=db, escalation_manager=esc)
    reviewer.handle({"bot_id": "99", "severity": "critical"})

    assert builder.calls, "context_builder.build was not invoked"
