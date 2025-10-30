from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass

import pytest


@pytest.fixture(autouse=True)
def _cleanup_resource_allocation_module():
    module_name = "menace_sandbox.resource_allocation_bot"
    sys.modules.pop(module_name, None)
    yield
    sys.modules.pop(module_name, None)


def _install_support_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    rp_mod = types.ModuleType("menace_sandbox.resource_prediction_bot")
    rp_mod.ResourceMetrics = type("ResourceMetrics", (), {})
    rp_mod.TemplateDB = type("TemplateDB", (), {})
    monkeypatch.setitem(sys.modules, "menace_sandbox.resource_prediction_bot", rp_mod)

    retry_mod = types.ModuleType("menace_sandbox.retry_utils")

    def retry(_exc: type[BaseException], *, attempts: int = 1):
        def decorator(func):
            return func

        return decorator

    retry_mod.retry = retry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace_sandbox.retry_utils", retry_mod)

    db_mod = types.ModuleType("menace_sandbox.databases")
    db_mod.MenaceDB = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace_sandbox.databases", db_mod)

    contrarian_mod = types.ModuleType("menace_sandbox.contrarian_db")
    contrarian_mod.ContrarianDB = None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "menace_sandbox.contrarian_db", contrarian_mod)

    class _DummyConnection:
        def execute(self, *_args, **_kwargs):
            class _Result:
                lastrowid = 1

            return _Result()

        def commit(self) -> None:
            pass

    class _DummyRouter:
        menace_id = 0

        def get_connection(self, *_args, **_kwargs):
            return _DummyConnection()

    db_router_mod = types.ModuleType("db_router")
    db_router_mod.GLOBAL_ROUTER = None  # type: ignore[attr-defined]
    db_router_mod.init_db_router = lambda *_args, **_kwargs: _DummyRouter()
    monkeypatch.setitem(sys.modules, "db_router", db_router_mod)

    snippet_mod = types.ModuleType("snippet_compressor")
    snippet_mod.compress_snippets = lambda value: value
    monkeypatch.setitem(sys.modules, "snippet_compressor", snippet_mod)


def test_resource_allocation_import_is_lazy(monkeypatch):
    module_name = "menace_sandbox.resource_allocation_bot"

    _install_support_stubs(monkeypatch)
    sys.modules.pop("menace_sandbox.bot_registry", None)
    sys.modules.pop("menace_sandbox.data_bot", None)

    class StubRegistry:
        constructed = 0

        def __init__(self) -> None:
            type(self).constructed += 1

    class StubDataBot:
        constructed = 0

        def __init__(self, *args, **kwargs) -> None:
            type(self).constructed += 1

    bot_registry_mod = types.ModuleType("menace_sandbox.bot_registry")
    data_bot_mod = types.ModuleType("menace_sandbox.data_bot")
    bot_registry_mod.BotRegistry = StubRegistry  # type: ignore[attr-defined]
    data_bot_mod.DataBot = StubDataBot  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "menace_sandbox.bot_registry", bot_registry_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot_mod)

    class _TrackingModule(types.ModuleType):
        def __init__(self, name: str) -> None:
            super().__init__(name)
            self.accessed = False

        def __getattribute__(self, name: str):  # pragma: no cover - trivial access hook
            if name not in {
                "accessed",
                "__dict__",
                "__class__",
                "__name__",
                "__package__",
                "__spec__",
                "__loader__",
                "__file__",
                "__builtins__",
                "__doc__",
            }:
                object.__setattr__(self, "accessed", True)
            return super().__getattribute__(name)

    vector_pkg = types.ModuleType("vector_service")
    vector_pkg.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {})
    context_module = _TrackingModule("vector_service.context_builder")
    context_module.ContextBuilder = type("ContextBuilder", (), {})
    context_module.FallbackResult = type("FallbackResult", (), {})
    context_module.ErrorResult = type("ErrorResult", (), {})

    monkeypatch.setitem(sys.modules, "vector_service", vector_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", context_module)

    module = importlib.import_module(module_name)

    assert module._registry_instance is None
    assert module._data_bot_instance is None
    assert StubRegistry.constructed == 0
    assert StubDataBot.constructed == 0
    assert context_module.accessed is False


def test_resource_allocation_helpers_bootstrap(monkeypatch):
    module_name = "menace_sandbox.resource_allocation_bot"

    _install_support_stubs(monkeypatch)
    class DummyNodes(dict):
        pass

    class DummyGraph:
        def __init__(self) -> None:
            self.nodes: DummyNodes = DummyNodes()

    class DummyRegistry:
        def __init__(self) -> None:
            self.registered: list[dict] = []
            self.updated: list[tuple[str, str | None, dict]] = []
            self.graph = DummyGraph()
            self.modules: dict[str, str | None] = {}

        def register_bot(self, **kwargs):
            self.registered.append(kwargs)
            node = self.graph.nodes.setdefault(kwargs["name"], {})
            node.update({"module": kwargs.get("module_path")})
            if "patch_id" in kwargs:
                node["patch_id"] = kwargs.get("patch_id")
            if "commit" in kwargs:
                node["commit"] = kwargs.get("commit")
            self.modules[kwargs["name"]] = kwargs.get("module_path")

        def update_bot(self, name, module_path, **kwargs):
            self.updated.append((name, module_path, kwargs))
            node = self.graph.nodes.setdefault(name, {})
            node.update({"module": module_path})
            if "patch_id" in kwargs:
                node["patch_id"] = kwargs.get("patch_id")
            if "commit" in kwargs:
                node["commit"] = kwargs.get("commit")
            self.modules[name] = module_path

        def hot_swap_active(self) -> bool:
            return False

    @dataclass
    class DummyThresholds:
        roi_drop: float = 0.1
        error_threshold: float = 0.1
        test_failure_threshold: float = 0.1

    class DummyDataBot:
        def __init__(self, *args, **kwargs) -> None:
            self.thresholds = DummyThresholds()

        def reload_thresholds(self, _name: str) -> DummyThresholds:
            return self.thresholds

    class DummyContextBuilder:
        def refresh_db_weights(self) -> None:
            pass

    class DummyAllocationDB:
        def __init__(self) -> None:
            self.records: list = []

    class DummyTemplateDB:
        pass

    vector_pkg = types.ModuleType("vector_service")
    vector_pkg.EmbeddableDBMixin = type("EmbeddableDBMixin", (), {})
    context_mod = types.ModuleType("vector_service.context_builder")
    context_mod.ContextBuilder = DummyContextBuilder
    context_mod.FallbackResult = type("FallbackResult", (), {})
    context_mod.ErrorResult = type("ErrorResult", (), {})
    monkeypatch.setitem(sys.modules, "vector_service", vector_pkg)
    monkeypatch.setitem(sys.modules, "vector_service.context_builder", context_mod)

    sys.modules.pop("menace_sandbox.bot_registry", None)
    sys.modules.pop("menace_sandbox.data_bot", None)

    bot_registry_mod = types.ModuleType("menace_sandbox.bot_registry")
    data_bot_mod = types.ModuleType("menace_sandbox.data_bot")

    registry = DummyRegistry()
    data_bot = DummyDataBot()

    def registry_factory() -> DummyRegistry:
        return registry

    def data_bot_factory(*_args, **_kwargs) -> DummyDataBot:
        return data_bot

    bot_registry_mod.BotRegistry = lambda: registry_factory()  # type: ignore[attr-defined]
    data_bot_mod.DataBot = data_bot_factory  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "menace_sandbox.bot_registry", bot_registry_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", data_bot_mod)

    module = importlib.import_module(module_name)

    assert module._registry_instance is None
    assert module._data_bot_instance is None

    allocator = module.ResourceAllocationBot(
        db=DummyAllocationDB(),
        template_db=DummyTemplateDB(),
        context_builder=DummyContextBuilder(),
    )

    assert module._registry_instance is registry
    assert module._data_bot_instance is data_bot
    assert allocator.bot_registry is registry
    assert registry.registered, "Registry should record registration during bootstrap"
