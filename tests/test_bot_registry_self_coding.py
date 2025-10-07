from __future__ import annotations

import pytest

import menace_sandbox.bot_registry as bot_registry


@pytest.fixture(autouse=True)
def disable_unmanaged_scan(monkeypatch):
    """Prevent background scanner threads during tests."""

    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "schedule_unmanaged_scan",
        lambda self, interval=3600.0: None,
    )
    yield


def _make_registry() -> bot_registry.BotRegistry:
    return bot_registry.BotRegistry(event_bus=None)


def test_register_bot_marks_self_coding_disabled_when_dependencies_missing(monkeypatch):
    monkeypatch.setattr(
        bot_registry,
        "ensure_self_coding_ready",
        lambda modules=None: (False, ("sklearn", "pydantic")),
    )

    registry = _make_registry()
    registry.register_bot("TaskValidationBot", is_coding_bot=True)

    node = registry.graph.nodes["TaskValidationBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["pydantic", "sklearn"]
    assert disabled["reason"].startswith("self-coding dependencies unavailable")
    assert node.get("pending_internalization") is False


def test_register_bot_handles_bootstrap_import_failures(monkeypatch):
    def _raise_components():
        raise bot_registry.SelfCodingUnavailableError(
            "self-coding bootstrap failed",
            missing=("torch", "numpy"),
        )

    monkeypatch.setattr(
        bot_registry,
        "_load_self_coding_components",
        lambda: _raise_components(),
    )

    registry = _make_registry()
    registry.register_bot("FutureProfitabilityBot", is_coding_bot=True)

    node = registry.graph.nodes["FutureProfitabilityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["numpy", "torch"]
    assert disabled["reason"].startswith("self-coding bootstrap failed")


def test_transient_import_errors_eventually_disable_self_coding(monkeypatch):
    registry = _make_registry()

    def _boom(*_a, **_k):  # pragma: no cover - monkeypatched in test
        raise ImportError(
            "cannot import name 'Helper' from partially initialized module 'menace.foo'"
        )

    scheduled: list[tuple[str, float | None]] = []

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        lambda *a, **k: _boom(),
    )
    monkeypatch.setattr(
        registry,
        "_schedule_internalization_retry",
        lambda name, *, delay=None: scheduled.append((name, delay)),
    )

    registry.register_bot("FutureLucrativityBot", is_coding_bot=True)

    # simulate background retries
    for _ in range(2):
        registry._retry_internalization("FutureLucrativityBot")

    node = registry.graph.nodes["FutureLucrativityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["transient_error"]["repeat_count"] == registry._max_transient_error_signature_repeats
    assert node.get("pending_internalization") is False
    assert "internalization_blocked" in node

