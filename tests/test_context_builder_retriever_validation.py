import pytest

from vector_service.context_builder import ContextBuilder, VectorServiceError
from vector_service.retriever import Retriever


def _builder_with_paths(paths):
    builder = ContextBuilder.__new__(ContextBuilder)
    builder._db_paths = paths
    return builder


def test_validate_retriever_config_missing_paths():
    builder = _builder_with_paths(
        {
            "bots": None,
            "code": None,
            "errors": None,
            "workflows": None,
            "information": None,
        }
    )

    with pytest.raises(VectorServiceError) as excinfo:
        builder.validate_retriever_config()

    assert "missing path" in str(excinfo.value)


def test_validate_retriever_config_unreadable_path(tmp_path):
    invalid_path = tmp_path / "code.db"
    invalid_path.mkdir()
    builder = _builder_with_paths(
        {
            "bots": None,
            "code": str(invalid_path),
            "errors": None,
            "workflows": None,
            "information": None,
        }
    )

    with pytest.raises(VectorServiceError) as excinfo:
        builder.validate_retriever_config()

    assert "code" in str(excinfo.value)


def test_validate_retriever_config_success(tmp_path):
    code_path = tmp_path / "code.db"
    code_path.write_text("")
    builder = _builder_with_paths(
        {
            "bots": None,
            "code": str(code_path),
            "errors": None,
            "workflows": None,
            "information": None,
        }
    )

    diagnostics = builder.validate_retriever_config()

    assert diagnostics["code"] == "ok"


def test_retriever_missing_db_inputs_reports_actionable_message():
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

    builder = DummyBuilder(
        {
            "bots": None,
            "enhancements": None,
            "workflows": None,
            "errors": None,
            "information": None,
            "code": None,
        }
    )
    retriever = Retriever(context_builder=builder, retriever_kwargs={})

    with pytest.raises(VectorServiceError) as excinfo:
        retriever._get_retriever()

    message = str(excinfo.value)
    assert "Missing DBs" in message
    assert "ContextBuilder DB paths" in message
    assert "retriever failure" not in message
