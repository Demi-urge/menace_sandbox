import importlib
import sys
import types
from pathlib import Path


def test_db_path_honours_sandbox_repo_path(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    (repo / "logs").mkdir()
    db_file = repo / "enhancements.db"
    db_file.write_text("")

    monkeypatch.setenv("SANDBOX_REPO_PATH", str(repo))
    sys.modules.pop("dynamic_path_router", None)
    import dynamic_path_router as dpr
    dpr.clear_cache()

    # stub heavy dependencies
    mods = {
        "db_router": {
            "DBRouter": object,
            "GLOBAL_ROUTER": object(),
            "SHARED_TABLES": {},
            "init_db_router": lambda *a, **k: object(),
            "queue_insert": lambda *a, **k: None,
        },
        "db_dedup": {"insert_if_unique": lambda *a, **k: True},
        "menace.override_policy": {
            "OverridePolicyManager": type("OverridePolicyManager", (), {})
        },
        "menace.chatgpt_idea_bot": {
            "ChatGPTClient": type("ChatGPTClient", (), {})
        },
        "gpt_memory_interface": {
            "GPTMemoryInterface": type("GPTMemoryInterface", (), {})
        },
        "menace.memory_aware_gpt_client": {
            "ask_with_memory": lambda *a, **k: ""
        },
        "memory_aware_gpt_client": {
            "ask_with_memory": lambda *a, **k: ""
        },
        "menace.log_tags": {"IMPROVEMENT_PATH": "", "INSIGHT": ""},
        "log_tags": {"IMPROVEMENT_PATH": "", "INSIGHT": ""},
        "menace.shared_gpt_memory": {"GPT_MEMORY_MANAGER": object()},
        "shared_gpt_memory": {"GPT_MEMORY_MANAGER": object()},
        "vector_service": {"EmbeddableDBMixin": object},
        "menace.unified_event_bus": {
            "UnifiedEventBus": type("UnifiedEventBus", (), {})
        },
        "menace.scope_utils": {
            "Scope": type("Scope", (), {}),
            "build_scope_clause": lambda *a, **k: "",
        },
    }
    for name, attrs in mods.items():
        mod = types.ModuleType(name)
        for attr, obj in attrs.items():
            setattr(mod, attr, obj)
        sys.modules.setdefault(name, mod)

    # ensure package attribute
    sys.modules.setdefault("menace", types.ModuleType("menace"))
    setattr(sys.modules["menace"], "RAISE_ERRORS", False)

    sys.modules.pop("menace.chatgpt_enhancement_bot", None)
    ceb = importlib.import_module("menace.chatgpt_enhancement_bot")

    assert ceb.DEFAULT_DB_PATH == db_file
    assert ceb.DB_PATH == db_file
    assert isinstance(ceb.DB_PATH, Path)
