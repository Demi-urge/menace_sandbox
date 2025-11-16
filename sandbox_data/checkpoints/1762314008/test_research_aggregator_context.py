import types
import pytest
import sys
from pathlib import Path

# Stub heavy dependencies
root = Path(__file__).resolve().parents[2]
sys.path.append(str(root))
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(root)]
sys.modules.setdefault("menace_sandbox", pkg)


class _StubContextBuilder:
    def refresh_db_weights(self):
        pass

    def build(self, *a, **k):  # pragma: no cover - simple stub
        return ""


vector_service_stub = types.SimpleNamespace(
    ContextBuilder=_StubContextBuilder, EmbeddableDBMixin=object
)
sys.modules.setdefault("vector_service", vector_service_stub)

# Additional stubs to avoid heavy imports
sys.modules.setdefault(
    "menace_sandbox.unified_event_bus", types.SimpleNamespace(UnifiedEventBus=object)
)
def _auto_link_stub(*a, **k):
    def decorator(fn):
        return fn
    return decorator


sys.modules.setdefault(
    "menace_sandbox.auto_link", types.SimpleNamespace(auto_link=_auto_link_stub)
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_enhancement_bot",
    types.SimpleNamespace(
        EnhancementDB=object,
        ChatGPTEnhancementBot=object,
        Enhancement=types.SimpleNamespace,
    ),
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_prediction_bot",
    types.SimpleNamespace(ChatGPTPredictionBot=object, IdeaFeatures=object),
)
sys.modules.setdefault(
    "menace_sandbox.text_research_bot", types.SimpleNamespace(TextResearchBot=object)
)
sys.modules.setdefault(
    "menace_sandbox.video_research_bot", types.SimpleNamespace(VideoResearchBot=object)
)
sys.modules.setdefault(
    "menace_sandbox.chatgpt_research_bot",
    types.SimpleNamespace(
        ChatGPTResearchBot=object,
        Exchange=types.SimpleNamespace,
        summarise_text=lambda *a, **k: "",
    ),
)
sys.modules.setdefault(
    "menace_sandbox.database_manager",
    types.SimpleNamespace(get_connection=lambda *a, **k: types.SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: False, execute=lambda *a, **k: types.SimpleNamespace(fetchall=lambda: [])), DB_PATH=""),
)
sys.modules.setdefault(
    "menace_sandbox.capital_management_bot", types.SimpleNamespace(CapitalManagementBot=object)
)
sys.modules.setdefault(
    "menace_sandbox.db_router",
    types.SimpleNamespace(DBRouter=object, GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None),
)
sys.modules.setdefault("menace_sandbox.menace_db", types.SimpleNamespace(MenaceDB=object))
sc_stub = types.SimpleNamespace(compress_snippets=lambda meta, **_: meta)
sys.modules.setdefault("snippet_compressor", sc_stub)

import menace_sandbox.research_aggregator_bot as rab
from menace_sandbox.research_aggregator_bot import ResearchAggregatorBot


class RecordingBuilder(_StubContextBuilder):
    def __init__(self):
        self.calls = []

    def build(self, topic, **_):
        self.calls.append(topic)
        return "CTX"

    def refresh_db_weights(self):
        pass


class RecordingChatGPTBot:
    def __init__(self):
        self.instructions = []

    def process(self, instruction, depth=1, ratio=0.2, *, context_builder):
        if context_builder is None:
            raise RuntimeError("missing builder")
        self.instructions.append(instruction)
        return types.SimpleNamespace(summary="ok")


class RecordingEnhancementBot:
    def propose(self, instruction, num_ideas=1, context="", *, context_builder):
        if context_builder is None:
            raise RuntimeError("missing builder")
        enh = types.SimpleNamespace(idea="idea", rationale="because", score=0.0)
        return [enh]


class RecordingPredictionBot:
    def evaluate_enhancement(self, idea, rationale, *, context_builder):
        if context_builder is None:
            raise RuntimeError("missing builder")
        return types.SimpleNamespace(description="d", reason="r", value=0.5)


class DummyInfoDB:
    current_model_id = 0

    def search(self, topic):
        return []

    def link_enhancement(self, *a, **k):
        pass


class DummyEnhDB:
    def fetch(self):
        return []

    def add(self, enh):
        return 1

    def link_model(self, *a, **k):
        pass

    def link_bot(self, *a, **k):
        pass

    def link_workflow(self, *a, **k):
        pass


class DummyDBRouter:
    def insert_info(self, *a, **k):
        pass


class DummyCapitalManager:
    def info_ratio(self, energy):
        return energy


def test_research_queries_include_context(tmp_path):
    builder = RecordingBuilder()
    chatgpt = RecordingChatGPTBot()
    enh = RecordingEnhancementBot()
    pred = RecordingPredictionBot()
    info = DummyInfoDB()
    enh_db = DummyEnhDB()
    router = DummyDBRouter()
    cap = DummyCapitalManager()
    bot = ResearchAggregatorBot(
        requirements=["topic"],
        info_db=info,
        enhancements_db=enh_db,
        enhancement_bot=enh,
        prediction_bot=pred,
        chatgpt_bot=chatgpt,
        capital_manager=cap,
        db_router=router,
        context_builder=builder,
    )
    bot._increment_enh_count = lambda *a, **k: None
    bot.process("topic", energy=3)
    assert builder.calls and builder.calls[0] == "topic", "context builder not invoked"
    assert chatgpt.instructions and chatgpt.instructions[0].startswith("CTX"), "context missing in prompt"


def test_research_context_compression(monkeypatch, tmp_path):
    builder = RecordingBuilder()
    chatgpt = RecordingChatGPTBot()
    enh = RecordingEnhancementBot()
    pred = RecordingPredictionBot()
    info = DummyInfoDB()
    enh_db = DummyEnhDB()
    router = DummyDBRouter()
    cap = DummyCapitalManager()

    def fake_compress(meta, **_):
        return {"snippet": "COMP-" + meta.get("snippet", "")}

    monkeypatch.setattr(rab, "compress_snippets", fake_compress)

    bot = ResearchAggregatorBot(
        requirements=["topic"],
        info_db=info,
        enhancements_db=enh_db,
        enhancement_bot=enh,
        prediction_bot=pred,
        chatgpt_bot=chatgpt,
        capital_manager=cap,
        db_router=router,
        context_builder=builder,
    )
    bot._increment_enh_count = lambda *a, **k: None
    bot.process("topic", energy=3)
    assert chatgpt.instructions and chatgpt.instructions[0].startswith("COMP-CTX")


def test_sub_bot_refuses_without_context_builder():
    chatgpt = RecordingChatGPTBot()
    with pytest.raises(RuntimeError):
        chatgpt.process("topic", context_builder=None)
