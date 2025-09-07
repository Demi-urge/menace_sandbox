import importlib
import sys
import types
from pathlib import Path

pkg = types.ModuleType("menace")
pkg.__path__ = [str(Path(__file__).resolve().parents[1])]
pkg.RAISE_ERRORS = False
sys.modules["menace"] = pkg

vector_service_pkg = types.ModuleType("vector_service")
vector_service_pkg.__path__ = []
vector_service_pkg.SharedVectorService = object
vector_service_pkg.CognitionLayer = object


class _StubContextBuilder:
    def refresh_db_weights(self):
        pass


ctx_mod = types.ModuleType("vector_service.context_builder")
ctx_mod.ContextBuilder = _StubContextBuilder
sys.modules["vector_service"] = vector_service_pkg
sys.modules["vector_service.context_builder"] = ctx_mod
sys.modules["menace.shared_gpt_memory"] = types.SimpleNamespace(GPT_MEMORY_MANAGER=None)


def test_warning_on_missing_sklearn(monkeypatch, caplog, tmp_path):
    caplog.set_level("WARNING")
    # stub out optional dependencies before import
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", ctx_mod)
    monkeypatch.setitem(
        sys.modules, "menace.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
    )
    cpb = importlib.import_module("menace.chatgpt_prediction_bot")
    bot = cpb.ChatGPTPredictionBot(tmp_path / "none.joblib")
    assert "scikit-learn" in caplog.text
    assert not cpb._SKLEARN_AVAILABLE
    assert isinstance(bot.pipeline, cpb.Pipeline)
    # cleanup to avoid side effects
    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)


def test_fallback_prediction_stable(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "sklearn", types.ModuleType("sklearn"))
    monkeypatch.setitem(sys.modules, "joblib", types.ModuleType("joblib"))
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setitem(sys.modules, "vector_service", vector_service_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", ctx_mod)
    monkeypatch.setitem(
        sys.modules, "menace.shared_gpt_memory", types.SimpleNamespace(GPT_MEMORY_MANAGER=None)
    )
    cpb = importlib.import_module("menace.chatgpt_prediction_bot")

    bot = cpb.ChatGPTPredictionBot(
        tmp_path / "none.joblib", l2=0.1, iters=5, val_steps=1
    )
    idea = cpb.IdeaFeatures(
        market_type="tech",
        monetization_model="ads",
        startup_cost=1.0,
        skill=1.0,
        competition=2.0,
        uniqueness=3.0,
    )
    first = bot.predict(idea)
    second = bot.predict(idea)
    assert first == second
    assert 0.0 <= first[1] <= 1.0

    for m in list(sys.modules):
        if m.startswith("menace"):
            sys.modules.pop(m, None)
