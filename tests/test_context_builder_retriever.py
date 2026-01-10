import sys
import types

import pytest

import universal_retriever
import vector_service.context_builder as context_builder
from vector_service.exceptions import RetrieverConfigurationError
from vector_service.retriever import Retriever


class _FakeDB:
    def __init__(self, path=None, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def encode_text(self, _text: str):
        return [0.0]


def _stub_retriever_modules(monkeypatch):
    modules = {
        "bot_database": ("BotDB", _FakeDB),
        "chatgpt_enhancement_bot": ("EnhancementDB", _FakeDB),
        "task_handoff_bot": ("WorkflowDB", _FakeDB),
        "error_bot": ("ErrorDB", _FakeDB),
        "information_db": ("InformationDB", _FakeDB),
        "code_database": ("CodeDB", _FakeDB),
    }
    for name, (class_name, db_cls) in modules.items():
        monkeypatch.setitem(sys.modules, name, types.SimpleNamespace(**{class_name: db_cls}))


def test_context_builder_retriever_registers_databases(tmp_path, monkeypatch):
    _stub_retriever_modules(monkeypatch)
    monkeypatch.setattr(context_builder, "ensure_bootstrapped", lambda: {})
    monkeypatch.setattr(universal_retriever, "_vector_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(universal_retriever, "MetricsDB", None)

    db_paths = {
        "bots": tmp_path / "bots.db",
        "enhancements": tmp_path / "enhancements.db",
        "code": tmp_path / "code.db",
        "errors": tmp_path / "errors.db",
        "workflows": tmp_path / "workflows.db",
        "information": tmp_path / "information.db",
    }
    for path in db_paths.values():
        path.write_text("")

    builder = context_builder.ContextBuilder(
        bots_db=db_paths["bots"],
        enhancements_db=db_paths["enhancements"],
        code_db=db_paths["code"],
        errors_db=db_paths["errors"],
        workflows_db=db_paths["workflows"],
        information_db=db_paths["information"],
        db_weights={"bot": 1.0},
        ranking_model=object(),
        patch_retriever=types.SimpleNamespace(roi_tag_weights={}, enhancement_weight=1.0),
    )

    retriever = builder.retriever._get_retriever()

    assert retriever._dbs


def test_missing_db_paths_raise_configuration_error(monkeypatch):
    class DummyBuilder:
        roi_tag_penalties = {}

        def __init__(self, paths):
            self._db_paths = paths

        def _retriever_db_specs(self):
            return (
                ("bots", (), "", "bots", "bot_db"),
                ("enhancements", (), "", "enhancements", "enhancement_db"),
                ("workflows", (), "", "workflows", "workflow_db"),
                ("errors", (), "", "errors", "error_db"),
                ("information", (), "", "information", "information_db"),
                ("code", (), "", "code", "code_db"),
            )

    class ExplodingUniversalRetriever:
        def __init__(self, **_kwargs):
            raise ValueError("At least one database instance is required")

    monkeypatch.setattr(
        "vector_service.retriever.UniversalRetriever",
        ExplodingUniversalRetriever,
    )

    retriever = Retriever(
        context_builder=DummyBuilder(
            {
                "bots": None,
                "enhancements": None,
                "workflows": None,
                "errors": None,
                "information": None,
                "code": None,
            }
        ),
        retriever_kwargs={},
    )

    with pytest.raises(RetrieverConfigurationError) as excinfo:
        retriever._get_retriever()

    message = str(excinfo.value)
    assert "Missing DBs" in message
    assert "ContextBuilder DB paths" in message
    assert "At least one database instance is required" not in message
