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


def test_internal_self_coding_modules_not_transient():
    exc = ModuleNotFoundError(
        "No module named 'menace_sandbox.quick_fix_engine'",
        name="menace_sandbox.quick_fix_engine",
    )
    assert not bot_registry._is_transient_internalization_error(exc)

    exc_top_level = ModuleNotFoundError(
        "No module named 'quick_fix_engine'",
        name="quick_fix_engine",
    )
    assert not bot_registry._is_transient_internalization_error(exc_top_level)


def test_register_bot_records_module_path_on_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(
        bot_registry.BotRegistry,
        "schedule_unmanaged_scan",
        lambda self, interval=3600.0: None,
    )
    registry = bot_registry.BotRegistry(event_bus=None)

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    def _raise(*_args, **_kwargs):
        raise bot_registry.SelfCodingUnavailableError(
            "self-coding bootstrap failed",
            missing=("quick_fix_engine",),
        )

    monkeypatch.setattr(registry, "_internalize_missing_coding_bot", _raise)

    scheduled: list[tuple[str, float | None]] = []

    def _schedule(name: str, *, delay: float | None = None) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    module_path = tmp_path / "example_bot.py"
    module_path.write_text("# stub\n", encoding="utf-8")

    registry.register_bot(
        "ExampleBot",
        module_path=module_path,
        is_coding_bot=True,
    )

    assert scheduled, "internalisation retry should be scheduled"
    registry._retry_internalization("ExampleBot")

    node = registry.graph.nodes["ExampleBot"]
    assert node["module"] == str(module_path)
    assert registry.modules["ExampleBot"] == str(module_path)
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["quick_fix_engine"]


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
    assert sorted(disabled["missing_dependencies"]) == ["pydantic", "sklearn"]
    assert disabled["reason"].startswith("self-coding dependencies unavailable")
    assert node.get("pending_internalization") is False


def test_register_bot_handles_bootstrap_import_failures(monkeypatch):
    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

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
    scheduled: list[tuple[str, float | None]] = []

    def _schedule(name: str, *, delay: float | None = None) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureProfitabilityBot", is_coding_bot=True)

    assert scheduled, "internalisation retry should be scheduled"
    registry._retry_internalization("FutureProfitabilityBot")

    node = registry.graph.nodes["FutureProfitabilityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["missing_dependencies"] == ["numpy", "torch"]
    assert disabled["reason"].startswith("self-coding bootstrap failed")


def test_transient_import_errors_eventually_disable_self_coding(monkeypatch):
    registry = _make_registry()

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

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
    def _schedule(name: str, *, delay: float | None = None) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureLucrativityBot", is_coding_bot=True)

    # simulate background retries until the registry disables self-coding
    for _ in range(5):
        if scheduled:
            registry._retry_internalization("FutureLucrativityBot")
        node = registry.graph.nodes["FutureLucrativityBot"]
        if node.get("self_coding_disabled"):
            break

    node = registry.graph.nodes["FutureLucrativityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["transient_error"]["repeat_count"] == registry._max_transient_error_signature_repeats
    assert (
        disabled["transient_error"]["total_repeat_count"]
        == registry._max_transient_error_signature_repeats
    )
    assert disabled["transient_error"]["unique_signatures"] == 1
    assert node.get("pending_internalization") is False
    assert "internalization_blocked" in node


def test_transient_import_errors_with_varying_signatures(monkeypatch):
    registry = _make_registry()
    registry._max_transient_error_signature_repeats = 99
    registry._max_transient_error_total_repeats = 3
    registry._transient_error_grace_period = 0.0

    monkeypatch.setattr(
        bot_registry, "ensure_self_coding_ready", lambda modules=None: (True, ())
    )

    attempts = {"count": 0}

    def _boom():
        attempts["count"] += 1
        raise ImportError(
            "cannot import name 'Helper' from partially initialized module 'menace.foo' "
            f"(attempt {attempts['count']})"
        )

    scheduled: list[tuple[str, float | None]] = []

    monkeypatch.setattr(
        registry,
        "_internalize_missing_coding_bot",
        lambda *a, **k: _boom(),
    )
    def _schedule(name: str, *, delay: float | None = None) -> None:
        registry._internalization_retry_attempts.setdefault(name, 0)
        scheduled.append((name, delay))

    monkeypatch.setattr(registry, "_schedule_internalization_retry", _schedule)

    registry.register_bot("FutureProfitabilityBot", is_coding_bot=True)

    for _ in range(10):
        if not scheduled:
            break
        name, _delay = scheduled.pop(0)
        registry._retry_internalization(name)
        node = registry.graph.nodes[name]
        if node.get("self_coding_disabled"):
            break

    node = registry.graph.nodes["FutureProfitabilityBot"]
    disabled = node.get("self_coding_disabled")
    assert disabled is not None, "self-coding should be disabled after repeated failures"
    error_meta = disabled["transient_error"]
    assert error_meta["total_repeat_count"] == attempts["count"]
    assert error_meta["unique_signatures"] == attempts["count"]
    assert error_meta["repeat_count"] == 1
    assert node.get("pending_internalization") is False


def test_manual_registration_clears_pending_state(tmp_path):
    registry = _make_registry()

    registry.graph.add_node("TaskValidationBot")
    node = registry.graph.nodes["TaskValidationBot"]
    node["pending_internalization"] = True
    node["self_coding_disabled"] = {
        "reason": "previous failure",
        "missing_dependencies": ["quick_fix_engine"],
    }
    registry._internalization_retry_attempts["TaskValidationBot"] = 3

    cancelled: list[str] = []

    class _Handle:
        def cancel(self) -> None:  # pragma: no cover - invoked by test
            cancelled.append("cancelled")

    registry._internalization_retry_handles["TaskValidationBot"] = _Handle()

    module_file = tmp_path / "task_validation_bot.py"
    module_file.write_text("# stub\n", encoding="utf-8")

    registry.register_bot(
        "TaskValidationBot",
        module_path=module_file,
        is_coding_bot=False,
    )

    node = registry.graph.nodes["TaskValidationBot"]
    assert "pending_internalization" not in node
    assert "internalization_blocked" not in node
    assert "internalization_errors" not in node
    assert registry._internalization_retry_attempts.get("TaskValidationBot") is None
    assert cancelled == ["cancelled"]

    disabled = node.get("self_coding_disabled")
    assert disabled is not None
    assert disabled["manual_override"] is True
    assert disabled["source"] == "manual_registration"
    assert disabled["module"].endswith("task_validation_bot.py")
    assert disabled.get("previous_reason") == "previous failure"
    assert disabled.get("missing_dependencies") == ["quick_fix_engine"]

