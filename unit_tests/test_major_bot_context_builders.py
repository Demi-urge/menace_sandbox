import os
import sys
from pathlib import Path
import types
import pytest
from menace.coding_bot_interface import manager_generate_helper

# Stub modules that these bots depend on to keep tests lightweight.

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


class RecordingBuilder:
    """Minimal ContextBuilder capturing build calls."""

    def __init__(self, *args, **kwargs):
        self.calls: list[str] = []

    def build(self, payload: str, **_kwargs):  # pragma: no cover - simple stub
        self.calls.append(payload)
        return "context"

    def refresh_db_weights(self):  # pragma: no cover - simple stub
        return None


class DummyCognitionLayer:
    def __init__(self, *, context_builder=None, **_kwargs):
        self.context_builder = context_builder

    def query(self, prompt: str, **_kwargs):  # pragma: no cover - simple stub
        return self.context_builder.build(prompt, session_id="s"), "sid"


vector_stub = types.SimpleNamespace(
    ContextBuilder=RecordingBuilder,
    CognitionLayer=DummyCognitionLayer,
    FallbackResult=object,
    ErrorResult=object,
    EmbeddingBackfill=type("EmbeddingBackfill", (), {"run": lambda self, *a, **k: None}),
    Retriever=object,
)
sys.modules.setdefault("vector_service", vector_stub)
sys.modules.setdefault("vector_service.context_builder", vector_stub)

sys.modules.setdefault(
    "snippet_compressor", types.SimpleNamespace(compress_snippets=lambda meta, **k: meta)
)

# Stubs required for quick_fix_engine module imports
sys.modules.setdefault("menace_sandbox.error_bot", types.SimpleNamespace(ErrorDB=object))
sys.modules.setdefault(
    "menace_sandbox.self_coding_manager",
    types.SimpleNamespace(SelfCodingManager=object),
)
sys.modules.setdefault(
    "menace_sandbox.knowledge_graph", types.SimpleNamespace(KnowledgeGraph=object)
)
sys.modules.setdefault(
    "menace_sandbox.patch_provenance", types.SimpleNamespace(PatchLogger=object)
)
sys.modules.setdefault(
    "menace_sandbox.codebase_diff_checker",
    types.SimpleNamespace(
        generate_code_diff=lambda *a, **k: {},
        flag_risky_changes=lambda *a, **k: False,
    ),
)
sys.modules.setdefault(
    "menace_sandbox.human_alignment_flagger",
    types.SimpleNamespace(_collect_diff_data=lambda *a, **k: {}),
)
sys.modules.setdefault(
    "menace_sandbox.human_alignment_agent",
    types.SimpleNamespace(
        HumanAlignmentAgent=lambda: types.SimpleNamespace(
            evaluate_changes=lambda *a, **k: {}
        )
    ),
)
sys.modules.setdefault(
    "menace_sandbox.violation_logger",
    types.SimpleNamespace(log_violation=lambda *a, **k: None),
)
sys.modules.setdefault(
    "sandbox_runner", types.SimpleNamespace(post_round_orphan_scan=lambda *a, **k: None)
)
sys.modules.setdefault(
    "menace_sandbox.coding_bot_interface",
    types.SimpleNamespace(
        self_coding_managed=lambda cls: cls,
        manager_generate_helper=manager_generate_helper,
    ),
)

sys.modules.setdefault(
    "db_router",
    types.SimpleNamespace(
        DBRouter=object,
        GLOBAL_ROUTER=None,
        init_db_router=lambda *a, **k: None,
        LOCAL_TABLES={},
        SHARED_TABLES={},
        queue_insert=lambda *a, **k: None,
    ),
)
sys.modules.setdefault("audit", types.SimpleNamespace(log_db_access=lambda *a, **k: None))
sys.modules.setdefault(
    "stripe_detection",
    types.SimpleNamespace(
        PAYMENT_KEYWORDS=[],
        HTTP_LIBRARIES=[],
        contains_payment_keyword=lambda text: False,
    ),
)


def _resolve_path(p: str) -> Path:
    root = Path(os.environ.get("SANDBOX_REPO_PATH", "."))
    cand = root / p
    if not cand.exists():
        raise FileNotFoundError(p)
    return cand


dynamic_path_router = types.SimpleNamespace(
    resolve_path=_resolve_path,
    path_for_prompt=lambda p: Path(p).as_posix(),
    clear_cache=lambda: None,
)
sys.modules["dynamic_path_router"] = dynamic_path_router

sys.modules["context_builder_util"] = types.SimpleNamespace(
    ensure_fresh_weights=lambda builder: builder.refresh_db_weights()
)

from menace_sandbox.automated_reviewer import AutomatedReviewer  # noqa: E402
import menace_sandbox.quick_fix_engine as qfe  # noqa: E402
from menace_sandbox.quick_fix_engine import QuickFixEngine, generate_patch  # noqa: E402


def test_automated_reviewer_requires_context_builder():
    with pytest.raises(TypeError):
        AutomatedReviewer(bot_db=object(), escalation_manager=object())  # type: ignore[call-arg]


def test_automated_reviewer_uses_context_builder():
    builder = RecordingBuilder("bots.db", "code.db", "errors.db", "workflows.db")
    db = types.SimpleNamespace(update_bot=lambda *a, **k: None)
    esc = types.SimpleNamespace(handle=lambda *a, **k: None)
    reviewer = AutomatedReviewer(context_builder=builder, bot_db=db, escalation_manager=esc)
    reviewer.handle({"bot_id": "1", "severity": "critical"})
    assert builder.calls, "context_builder.build was not invoked"


def test_quick_fix_engine_requires_context_builder():
    error_db = object()
    manager = types.SimpleNamespace()
    with pytest.raises(TypeError):
        QuickFixEngine(error_db, manager)  # type: ignore[call-arg]


def test_quick_fix_engine_uses_context_builder(tmp_path, monkeypatch):
    builder = RecordingBuilder("bots.db", "code.db", "errors.db", "workflows.db")

    class DummyEngine:
        def apply_patch(self, path, desc, **kwargs):
            return 1, "", 0.0

    (tmp_path / "mod.py").write_text("x = 1\n")  # path-ignore
    monkeypatch.setattr(qfe, "resolve_path", lambda p: tmp_path / p)
    monkeypatch.setattr(qfe, "path_for_prompt", lambda p: Path(p).as_posix())

    eng = DummyEngine()
    manager = types.SimpleNamespace(engine=eng, register_patch_cycle=lambda *a, **k: None)
    patch_id = generate_patch(
        "mod", manager, eng, context_builder=builder, description="fix bug"
    )
    assert patch_id == 1
    assert builder.calls and builder.calls[0] == "fix bug"


def test_apply_validated_patch_emits_rejection_event(tmp_path, monkeypatch):
    builder = RecordingBuilder("bots.db", "code.db", "errors.db", "workflows.db")

    class EventBus:
        def __init__(self):
            self.handlers: dict[str, list] = {}

        def publish(self, topic, payload):
            for cb in self.handlers.get(topic, []):
                cb(topic, payload)

        def subscribe(self, topic, cb):
            self.handlers.setdefault(topic, []).append(cb)

    events: list[dict] = []

    bus = EventBus()
    bus.subscribe("self_coding:patch_rejected", lambda t, p: events.append(p))

    calls: list[tuple] = []

    class DB:
        def record_validation(self, *a):
            calls.append(a)

    mgr = types.SimpleNamespace(
        engine=object(),
        event_bus=bus,
        data_bot=DB(),
        bot_registry=object(),
        bot_name="bot",
        register_bot=lambda *a, **k: None,
    )
    monkeypatch.setattr(qfe, "generate_patch", lambda *a, **k: (1, ["flag"]))
    monkeypatch.setattr(qfe.subprocess, "run", lambda *a, **k: None)
    engine = QuickFixEngine(object(), mgr, context_builder=builder)
    passed, pid, _flags = engine.apply_validated_patch(tmp_path / "m.py", "d")
    assert not passed and pid is None
    assert events and events[0] == {
        "bot": "bot",
        "module": str(tmp_path / "m.py"),
        "flags": ["flag"],
    }
    assert calls and calls[0] == ("bot", str(tmp_path / "m.py"), False, ["flag"])
