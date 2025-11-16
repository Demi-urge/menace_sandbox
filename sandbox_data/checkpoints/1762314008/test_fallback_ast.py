import sys
import types
from types import SimpleNamespace
from menace.coding_bot_interface import manager_generate_helper

# Stub heavy dependencies before importing the engine
code_db_stub = types.ModuleType("code_database")
code_db_stub.CodeDB = object
code_db_stub.CodeRecord = object
code_db_stub.PatchHistoryDB = object
code_db_stub.PatchRecord = object
sys.modules["code_database"] = code_db_stub
sys.modules["menace.code_database"] = code_db_stub

ue_stub = types.ModuleType("unified_event_bus")
ue_stub.UnifiedEventBus = object
sys.modules["unified_event_bus"] = ue_stub
sys.modules["menace.unified_event_bus"] = ue_stub

sgm_stub = types.ModuleType("shared_gpt_memory")
sgm_stub.GPT_MEMORY_MANAGER = object
sys.modules["shared_gpt_memory"] = sgm_stub
sys.modules["menace.shared_gpt_memory"] = sgm_stub

vector_stub = types.ModuleType("vector_service")
vector_stub.CognitionLayer = object
vector_stub.ContextBuilder = object
vector_stub.PatchLogger = object
vector_stub.VectorServiceError = Exception
vector_stub.SharedVectorService = object
sys.modules["vector_service"] = vector_stub
sys.modules["menace.vector_service"] = vector_stub

# Ensure fresh import context
sys.modules.pop("self_coding_engine", None)

import menace.self_coding_engine as self_coding_engine
from menace.self_coding_engine import SelfCodingEngine, MANAGER_CONTEXT


def test_fallback_compiles_and_uses_snippet_context():
    engine = object.__new__(SelfCodingEngine)
    engine.llm_client = None
    engine.prompt_engine = None
    engine.context_builder = object()
    engine.logger = types.SimpleNamespace(debug=lambda *a, **k: None)
    snippet = """total = 0\nfor item in items:\n    total += item\nif total > 0:\n    return total"""
    engine.suggest_snippets = lambda *a, **k: [SimpleNamespace(code=snippet)]
    manager = SimpleNamespace(engine=engine)
    result = manager_generate_helper(
        manager,
        "summation helper",
        context_builder=engine.context_builder,
    )
    compile(result, "<generated>", "exec")
    assert "for item in items" in result
    assert "total" in result
