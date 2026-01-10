import pytest

from vector_service.context_builder import ContextBuilder, VectorServiceError


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
