import importlib
from pathlib import Path


def test_tool_predictor_uses_resolve_path(monkeypatch, tmp_path):
    import micro_models.tool_predictor as tp

    called = {}

    def fake_resolve(name: str) -> Path:
        called["name"] = name
        return tmp_path / "model"

    with monkeypatch.context() as m:
        m.setattr("dynamic_path_router.resolve_path", fake_resolve)
        tp = importlib.reload(tp)
        assert called["name"] == "micro_models/tool_predictor_model"
        assert tp.DEFAULT_PATH == tmp_path / "model"
    importlib.reload(tp)


def test_diff_summarizer_uses_resolve_path(monkeypatch, tmp_path):
    import micro_models.diff_summarizer as ds

    called = {}

    def fake_resolve(name: str) -> Path:
        called["name"] = name
        return tmp_path / "model"

    with monkeypatch.context() as m:
        m.setattr("dynamic_path_router.resolve_path", fake_resolve)
        ds = importlib.reload(ds)
        assert called["name"] == "micro_models/diff_summarizer_model"
        assert ds._MODEL_PATH == tmp_path / "model"
    importlib.reload(ds)


def test_error_classifier_uses_resolve_path(monkeypatch, tmp_path):
    import micro_models.error_classifier as ec

    called = {}

    def fake_resolve(name: str) -> Path:
        called["name"] = name
        return tmp_path / "model"

    with monkeypatch.context() as m:
        m.setattr("dynamic_path_router.resolve_path", fake_resolve)
        ec = importlib.reload(ec)
        assert called["name"] == "micro_models/error_classifier_model"
        assert ec.DEFAULT_PATH == tmp_path / "model"
    importlib.reload(ec)
