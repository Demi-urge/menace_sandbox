from __future__ import annotations

import importlib
import sys
import types


prediction_manager_module = importlib.import_module("prediction_manager_bot")
vectorizer_module = importlib.import_module("vector_service.patch_vectorizer")
lazy_data_bot_module = importlib.import_module("menace_sandbox.shared.lazy_data_bot")

_LazyBotRegistry = prediction_manager_module._LazyBotRegistry
_RegistryGraphShim = prediction_manager_module._RegistryGraphShim
PatchVectorizer = vectorizer_module.PatchVectorizer
_PatchVectorizerWarmupDBShim = vectorizer_module._PatchVectorizerWarmupDBShim
_FallbackDB = lazy_data_bot_module._FallbackDB
_build_fallback_data_bot = lazy_data_bot_module._build_fallback_data_bot


def _import_autonomous_bootstrap_with_stubs():
    sys.modules.setdefault("sandbox_settings", types.SimpleNamespace(SandboxSettings=lambda: object()))
    sys.modules.setdefault(
        "sandbox_runner.bootstrap",
        types.SimpleNamespace(
            bootstrap_environment=lambda settings, _verify: settings,
            ensure_autonomous_launch=lambda **_k: None,
            _verify_required_dependencies=lambda: None,
        ),
    )
    sys.modules.setdefault(
        "self_improvement.api",
        types.SimpleNamespace(
            init_self_improvement=lambda _settings: None,
            start_self_improvement_cycle=lambda _cfg: types.SimpleNamespace(
                start=lambda: None,
                join=lambda: None,
                stop=lambda: None,
            ),
        ),
    )
    sys.modules.setdefault("bot_discovery", types.SimpleNamespace(discover_and_register_coding_bots=lambda *_a, **_k: None))
    return importlib.import_module("autonomous_bootstrap")


def test_self_coding_manager_shim_surface() -> None:
    module = _import_autonomous_bootstrap_with_stubs()
    manager = module._SelfCodingManagerShim(bot_registry=object())
    assert manager.bot_name == ""
    assert manager.event_bus is None
    assert manager.bot_registry is not None


def test_evolution_manager_shim_surface() -> None:
    module = _import_autonomous_bootstrap_with_stubs()
    manager = module._EvolutionManagerShim()
    assert manager.bots == []
    manager.bots.append("bot_a")
    assert list(manager.bots) == ["bot_a"]
    assert manager.run_cycle() == []


def test_registry_graph_shim_nodes_mapping() -> None:
    graph = _RegistryGraphShim()
    graph.nodes.setdefault("alpha", {})["enabled"] = True
    assert list(graph.nodes) == ["alpha"]
    assert graph.nodes["alpha"]["enabled"] is True


def test_lazy_registry_uses_graph_shim() -> None:
    registry = _LazyBotRegistry()
    assert isinstance(registry.graph, _RegistryGraphShim)
    registration_id = registry.register_bot("beta", "beta.py")
    assert isinstance(registration_id, str)
    assert "beta" in registry.graph.nodes


def test_fallback_data_bot_db_shim_surface() -> None:
    db = _FallbackDB()
    assert db.fetch(limit=50) == []

    data_bot = _build_fallback_data_bot()
    assert data_bot.db.fetch() == []
    assert data_bot.roi("any") == 0.0
    assert list(data_bot.detect_anomalies([], "cpu")) == []


def test_patch_vectorizer_warmup_db_shim_surface(monkeypatch) -> None:
    monkeypatch.setenv("MENACE_BOOTSTRAP", "1")
    vectorizer = PatchVectorizer(path="patches.db")
    assert isinstance(vectorizer.db, _PatchVectorizerWarmupDBShim)
    assert vectorizer.db.path == "patches.db"
    assert vectorizer.db.router is None
    assert vectorizer.db.get("missing") is None
    assert vectorizer.db._vec_db_enabled is False
