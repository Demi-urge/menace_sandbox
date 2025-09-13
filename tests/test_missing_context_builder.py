import types
import sys
from pathlib import Path
import pytest

# Provide lightweight stubs for the vector_service module used by the targets

class DummyContextBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, *a, **k):  # pragma: no cover - simple stub
        return ""


vector_service_stub = types.SimpleNamespace(
    ContextBuilder=DummyContextBuilder,
    FallbackResult=object,
    ErrorResult=object,
    Retriever=object,
    EmbeddingBackfill=object,
    CognitionLayer=object,
    EmbeddableDBMixin=object,
    SharedVectorService=object,
)
sys.modules.setdefault("vector_service", vector_service_stub)

# Stub modules imported by quick_fix_engine to avoid heavy dependencies
sys.modules.setdefault(
    "menace_sandbox.error_bot",
    types.SimpleNamespace(ErrorDB=object),
)  # noqa: E501
sys.modules.setdefault(
    "menace_sandbox.self_coding_manager",
    types.SimpleNamespace(SelfCodingManager=object),
)  # noqa: E501
sys.modules.setdefault(
    "menace_sandbox.knowledge_graph",
    types.SimpleNamespace(KnowledgeGraph=object),
)  # noqa: E501
sys.modules.setdefault(
    "menace_sandbox.patch_provenance",
    types.SimpleNamespace(PatchLogger=object),
)  # noqa: E501

# Ensure package imports work when tests executed directly
sys.path.append(str(Path(__file__).resolve().parents[2]))

from menace_sandbox.bot_development_bot import BotDevelopmentBot  # noqa: E402
from menace_sandbox.quick_fix_engine import QuickFixEngine  # noqa: E402
from menace_sandbox.automated_reviewer import AutomatedReviewer  # noqa: E402
from menace_sandbox.implementation_optimiser_bot import ImplementationOptimiserBot  # noqa: E402


def test_bot_development_bot_requires_context_builder(tmp_path):
    with pytest.raises(ValueError):
        BotDevelopmentBot(repo_base=tmp_path, context_builder=object())


def test_quick_fix_engine_requires_context_builder():
    error_db = object()
    manager = types.SimpleNamespace()
    with pytest.raises(RuntimeError):
        QuickFixEngine(
            error_db,
            manager,
            context_builder=None,
            helper_fn=lambda *a, **k: "",
        )


def test_automated_reviewer_requires_context_builder():
    bot_db = types.SimpleNamespace()
    escalation_manager = types.SimpleNamespace(handle=lambda *a, **k: None)
    with pytest.raises(TypeError):
        AutomatedReviewer(
            context_builder=object(),
            bot_db=bot_db,
            escalation_manager=escalation_manager,
        )


def test_implementation_optimiser_bot_requires_context_builder():
    with pytest.raises(ValueError):
        ImplementationOptimiserBot(context_builder=None)  # type: ignore[arg-type]
