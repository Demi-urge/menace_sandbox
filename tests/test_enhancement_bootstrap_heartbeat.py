import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


MODULE_NAME = "menace_sandbox.enhancement_bot"


def _stub_module(monkeypatch, name: str, attrs: dict[str, object]) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


@pytest.fixture(autouse=True)
def reset_enhancement_module(monkeypatch):
    sys.modules.pop(MODULE_NAME, None)
    yield
    sys.modules.pop(MODULE_NAME, None)


def test_bootstrap_heartbeat_reuses_placeholder(monkeypatch):
    package = ModuleType("menace_sandbox")
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, "menace_sandbox", package)

    broker = SimpleNamespace(active_pipeline=None, active_sentinel=None)

    def _advertise(*, pipeline=None, sentinel=None, owner=None):  # noqa: ANN001
        broker.active_pipeline = pipeline
        broker.active_sentinel = sentinel
        broker.owner = owner

    broker.advertise = _advertise

    placeholder_pipeline = SimpleNamespace(name="placeholder")
    placeholder_manager = SimpleNamespace(name="sentinel")

    coding_interface = _stub_module(
        monkeypatch,
        "menace_sandbox.coding_bot_interface",
        {
            "_bootstrap_dependency_broker": lambda: broker,
            "_current_bootstrap_context": lambda: SimpleNamespace(),
            "advertise_bootstrap_placeholder": mock.Mock(
                return_value=(placeholder_pipeline, placeholder_manager)
            ),
            "get_active_bootstrap_pipeline": lambda: (None, None),
            "prepare_pipeline_for_bootstrap": mock.Mock(
                side_effect=AssertionError(
                    "prepare_pipeline_for_bootstrap should not run"
                )
            ),
            "read_bootstrap_heartbeat": lambda: {"ts": 1},
            "self_coding_managed": lambda **_: (lambda cls: cls),
        },
    )

    context_builder_module = _stub_module(
        monkeypatch,
        "vector_service.context_builder",
        {
            "ContextBuilder": object,
            "load_failed_tags": lambda *a, **k: set(),
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.vector_service.context_builder",
        context_builder_module.__dict__,
    )
    _stub_module(
        monkeypatch,
        "vector_service",
        {"context_builder": context_builder_module, "__path__": []},
    )

    _stub_module(
        monkeypatch,
        "menace_sandbox.bot_registry",
        {"BotRegistry": lambda *_, **__: object()},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.data_bot",
        {
            "DataBot": lambda *_, **__: object(),
            "persist_sc_thresholds": lambda *_, **__: None,
        },
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.chatgpt_enhancement_bot",
        {
            "EnhancementDB": object,
            "EnhancementHistory": object,
            "Enhancement": object,
        },
    )
    _stub_module(monkeypatch, "menace_sandbox.micro_models", {})
    _stub_module(
        monkeypatch,
        "menace_sandbox.micro_models.diff_summarizer",
        {"summarize_diff": lambda *_, **__: ""},
    )
    _stub_module(monkeypatch, "snippet_compressor", {"compress_snippets": lambda value: value})
    _stub_module(monkeypatch, "billing", {})
    _stub_module(monkeypatch, "billing.prompt_notice", {"prepend_payment_notice": lambda prompt: prompt})
    _stub_module(
        monkeypatch,
        "llm_interface",
        {
            "LLMClient": object,
            "Prompt": object,
            "LLMResult": SimpleNamespace,
        },
    )
    _stub_module(monkeypatch, "menace_sandbox.self_coding_manager", {"SelfCodingManager": object, "internalize_coding_bot": lambda *_, **__: placeholder_manager})
    _stub_module(monkeypatch, "menace_sandbox.self_coding_engine", {"SelfCodingEngine": lambda *_, **__: object()})
    _stub_module(monkeypatch, "menace_sandbox.model_automation_pipeline", {"ModelAutomationPipeline": object})
    _stub_module(monkeypatch, "menace_sandbox.threshold_service", {"ThresholdService": lambda *_, **__: object()})
    _stub_module(monkeypatch, "menace_sandbox.code_database", {"CodeDB": lambda *_, **__: object()})
    _stub_module(monkeypatch, "menace_sandbox.gpt_memory", {"GPTMemoryManager": lambda *_, **__: object()})
    _stub_module(monkeypatch, "menace_sandbox.self_coding_thresholds", {"get_thresholds": lambda *_: SimpleNamespace(roi_drop=1, error_increase=1, test_failure_increase=1)})
    _stub_module(monkeypatch, "menace_sandbox.shared_evolution_orchestrator", {"get_orchestrator": lambda *_, **__: object()})
    _stub_module(
        monkeypatch,
        "menace_sandbox.context_builder_util",
        {"create_context_builder": lambda *_, **__: SimpleNamespace(refresh_db_weights=lambda: {})},
    )
    _stub_module(
        monkeypatch,
        "menace_sandbox.context_builder",
        {"handle_failure": lambda *_, **__: None, "PromptBuildError": RuntimeError},
    )
    _stub_module(
        monkeypatch,
        "context_builder_util",
        {"create_context_builder": lambda *_, **__: SimpleNamespace(refresh_db_weights=lambda: {})},
    )
    _stub_module(
        monkeypatch,
        "context_builder",
        {"handle_failure": lambda *_, **__: None, "PromptBuildError": RuntimeError},
    )

    module = importlib.import_module(MODULE_NAME)

    assert coding_interface.prepare_pipeline_for_bootstrap.call_count == 0
    assert broker.active_pipeline is placeholder_pipeline
    assert broker.active_sentinel is placeholder_manager

    runtime = module.get_runtime()
    assert runtime.pipeline is placeholder_pipeline
    assert runtime.manager is placeholder_manager


if __name__ == "__main__":
    monkeypatch = pytest.MonkeyPatch()
    try:
        test_bootstrap_heartbeat_reuses_placeholder(monkeypatch)
    finally:
        monkeypatch.undo()
