import sys
import types
import importlib
import pytest

# Ensure we import the real helper rather than the lightweight test stub
sys.modules.pop("context_builder_util", None)

# Stub heavy vector_service before importing the helper
sys.modules.setdefault(
    "vector_service.context_builder", types.SimpleNamespace(ContextBuilder=object)
)

cbu = importlib.import_module("context_builder_util")  # noqa: E402


def test_create_context_builder_paths(monkeypatch, tmp_path):
    for env_var in [
        "SANDBOX_DATA_DIR",
        "BOT_DB_PATH",
        "CODE_DB_PATH",
        "ERROR_DB_PATH",
        "WORKFLOW_DB_PATH",
    ]:
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    for name in ("bots.db", "code.db", "errors.db", "workflows.db"):
        (tmp_path / name).write_text("")

    captured = {}

    class DummyBuilder:
        pass

    def fake_cb(**kwargs):
        captured['kwargs'] = kwargs
        return DummyBuilder()

    monkeypatch.setattr(cbu._create_module, 'ContextBuilder', fake_cb)
    builder = cbu.create_context_builder()
    assert captured['kwargs'] == {
        'bots_db': str(tmp_path / "bots.db"),
        'code_db': str(tmp_path / "code.db"),
        'errors_db': str(tmp_path / "errors.db"),
        'workflows_db': str(tmp_path / "workflows.db"),
    }
    assert isinstance(builder, DummyBuilder)


def test_create_context_builder_missing_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    for name in ("bots.db", "code.db", "errors.db"):
        (tmp_path / name).write_text("")

    with pytest.raises(FileNotFoundError) as excinfo:
        cbu.create_context_builder()

    assert "workflows.db" in str(excinfo.value)


def test_create_context_builder_unreadable_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    (tmp_path / "bots.db").write_text("")
    (tmp_path / "errors.db").write_text("")
    (tmp_path / "workflows.db").write_text("")
    (tmp_path / "code.db").mkdir()

    with pytest.raises(OSError) as excinfo:
        cbu.create_context_builder()

    assert "code.db" in str(excinfo.value)


def test_create_context_builder_requires_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    for name in ("bots.db", "code.db", "errors.db", "workflows.db"):
        (tmp_path / name).write_text("")

    class DummyBuilder:
        def __init__(self):
            pass

    monkeypatch.setattr(cbu._create_module, "ContextBuilder", DummyBuilder)
    with pytest.raises(ValueError):
        cbu.create_context_builder()


def test_ensure_fresh_weights_invokes_builder():
    called = False

    class DummyBuilder:
        def refresh_db_weights(self):
            nonlocal called
            called = True

    cbu.ensure_fresh_weights(DummyBuilder())
    assert called


def test_ensure_fresh_weights_propagates_exception():
    class DummyBuilder:
        def refresh_db_weights(self):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        cbu.ensure_fresh_weights(DummyBuilder())
