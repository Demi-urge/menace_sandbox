"""Unit tests for EvolutionOrchestrator degradation handling."""

from __future__ import annotations

import types
from pathlib import Path


class DummyBus:
    def __init__(self) -> None:
        self.subs: dict[str, list] = {}

    def subscribe(self, topic, fn):
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic, payload):
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DummyDataBot:
    def __init__(self, bus: DummyBus):
        self.event_bus = bus

    def subscribe_degradation(self, cb):  # pragma: no cover - simple
        self.event_bus.subscribe("bot:degraded", lambda _t, e: cb(e))

    def check_degradation(self, *args, **kwargs):  # pragma: no cover - noop
        pass


class DummySelfCodingManager:
    def __init__(self, module: Path):
        class _Graph(dict):
            def __init__(self, module: Path):
                super().__init__({"dummy": {"module": str(module)}})

            @property
            def nodes(self):  # pragma: no cover - simple mapping
                return self

        class _Reg:
            def __init__(self, module: Path):
                self.graph = _Graph(module)

        self.bot_registry = _Reg(module)
        self.bot_name = "dummy"
        self.calls: list[tuple[str, dict]] = []

    def register_patch_cycle(self, desc, ctx):  # pragma: no cover - record call
        self.calls.append((desc, ctx))

    def should_refactor(self):  # pragma: no cover - always false to skip patch
        return False


class DummyCapital:
    def energy_score(self, *a, **k):  # pragma: no cover - constant
        return 1.0


class DummyHistoryDB:
    def add(self, *a, **k):  # pragma: no cover - noop
        pass


def test_degradation_event_schedules_patch_cycle(tmp_path):
    mod = tmp_path / "dummy.py"
    mod.write_text("def foo():\n    return 1\n")

    bus = DummyBus()
    data_bot = DummyDataBot(bus)
    manager = DummySelfCodingManager(mod)

    from menace.evolution_orchestrator import EvolutionOrchestrator

    orch = EvolutionOrchestrator(
        data_bot,
        DummyCapital(),
        types.SimpleNamespace(),
        types.SimpleNamespace(),
        selfcoding_manager=manager,
        event_bus=bus,
        history_db=DummyHistoryDB(),
        roi_gain_floor=0.1,  # force prediction skip
    )

    orch.register_bot("dummy")

    event = {
        "bot": "dummy",
        "roi_baseline": 1.0,
        "delta_roi": -1.0,
        "errors_baseline": 0.0,
        "delta_errors": 1.0,
        "tests_failed_baseline": 0.0,
        "delta_tests_failed": 0.0,
    }
    bus.publish("bot:degraded", event)

    assert manager.calls, "patch cycle should be registered"
    assert "dummy" in orch._pending_patch_cycle
    desc, ctx = manager.calls[0]
    assert ctx["delta_roi"] == -1.0

