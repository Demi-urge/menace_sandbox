from __future__ import annotations

from pathlib import Path


class DummyBus:
    def __init__(self) -> None:
        self.subs: dict[str, list] = {}
        self.events: list[tuple[str, object]] = []

    def subscribe(self, topic: str, fn) -> None:
        self.subs.setdefault(topic, []).append(fn)

    def publish(self, topic: str, payload: object) -> None:
        self.events.append((topic, payload))
        for fn in self.subs.get(topic, []):
            fn(topic, payload)


class DataBot:
    """Minimal DataBot that publishes a degradation event when metrics breach."""

    def __init__(self, event_bus: DummyBus) -> None:
        self.event_bus = event_bus
        self._thresholds: dict[str, tuple[float, float]] = {}

    def check_degradation(self, bot: str, roi: float, errors: float) -> None:
        roi_thresh, err_thresh = self._thresholds.get(bot, (0.0, 0.0))
        if roi <= roi_thresh or errors >= err_thresh:
            payload = {
                "bot": bot,
                "delta_roi": roi - roi_thresh,
                "delta_errors": errors - err_thresh,
            }
            self.event_bus.publish("degradation:detected", payload)


class BotRegistry:
    def __init__(self, event_bus: DummyBus) -> None:
        self.event_bus = event_bus
        self.graph: dict[str, dict] = {}

    def register_bot(self, name: str, **meta) -> None:  # pragma: no cover - simple stub
        self.graph[name] = meta

    def update_bot(
        self, name: str, module_path: str, patch_id: int | None = None, commit: str | None = None
    ) -> None:  # pragma: no cover - simple stub
        self.graph.setdefault(name, {})
        self.graph[name].update(
            {"module": module_path, "patch_id": patch_id, "commit": commit}
        )


class DummyEngine:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_helper(self, desc: str, **kwargs) -> str:
        self.calls.append(desc)
        return "code"


class DummyQuickFix:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def apply_validated_patch(self, module_path: str, desc: str, ctx_meta: dict) -> tuple[bool, int, list]:
        self.calls.append((module_path, desc))
        return True, 1, []


class SelfCodingManager:
    def __init__(
        self,
        engine: DummyEngine,
        pipeline: object,
        *,
        bot_name: str,
        data_bot: DataBot,
        bot_registry: BotRegistry,
        event_bus: DummyBus | None = None,
        quick_fix: DummyQuickFix | None = None,
    ) -> None:
        self.engine = engine
        self.pipeline = pipeline
        self.bot_name = bot_name
        self.data_bot = data_bot
        self.bot_registry = bot_registry
        self.event_bus = event_bus
        self.quick_fix = quick_fix

    def register_patch_cycle(
        self,
        description: str,
        context_meta: dict | None = None,
        provenance_token: str | None = None,
        **kwargs,
    ) -> tuple[int, str]:
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:cycle_registered",
                {"bot": self.bot_name, "description": description},
            )
        return 1, "commit"

    def generate_and_patch(
        self,
        path: Path,
        description: str,
        *,
        context_meta: dict | None = None,
        provenance_token: str | None = None,
        **kwargs,
    ) -> tuple[None, str]:
        self.engine.generate_helper(description, path=str(path))
        if self.quick_fix:
            self.quick_fix.apply_validated_patch(str(path), description, context_meta or {})
        if self.event_bus:
            self.event_bus.publish(
                "self_coding:patch_applied",
                {"bot": self.bot_name, "path": str(path)},
            )
        return None, "commit"


def internalize_coding_bot(
    bot_name: str,
    engine: DummyEngine,
    pipeline: object,
    *,
    data_bot: DataBot,
    bot_registry: BotRegistry,
    evolution_orchestrator: EvolutionOrchestrator | None = None,
    quick_fix: DummyQuickFix | None = None,
    roi_threshold: float | None = None,
    error_threshold: float | None = None,
) -> SelfCodingManager:
    bus = getattr(evolution_orchestrator, "event_bus", None)
    manager = SelfCodingManager(
        engine,
        pipeline,
        bot_name=bot_name,
        data_bot=data_bot,
        bot_registry=bot_registry,
        event_bus=bus,
        quick_fix=quick_fix,
    )
    bot_registry.register_bot(bot_name, manager=manager, data_bot=data_bot, is_coding_bot=True)
    data_bot._thresholds[bot_name] = (roi_threshold or 0.0, error_threshold or 0.0)
    if evolution_orchestrator is not None:
        evolution_orchestrator.selfcoding_manager = manager
        if bus:
            bus.subscribe(
                "degradation:detected",
                lambda _t, e: evolution_orchestrator.register_patch_cycle(e),
            )
    return manager


class EvolutionOrchestrator:
    def __init__(self, *, event_bus: DummyBus, module_path: Path) -> None:
        self.event_bus = event_bus
        self.module_path = module_path
        self.selfcoding_manager: SelfCodingManager | None = None

    def register_patch_cycle(self, event: dict) -> None:
        desc = f"auto_patch_due_to_degradation:{event.get('bot', '')}"
        if self.selfcoding_manager:
            self.selfcoding_manager.register_patch_cycle(desc, context_meta=event)
            self.selfcoding_manager.generate_and_patch(
                self.module_path, desc, context_meta=event, provenance_token="prov"
            )


def test_internalize_metrics_breach_triggers_patch_cycle(tmp_path: Path) -> None:
    bus = DummyBus()
    data_bot = DataBot(bus)
    registry = BotRegistry(bus)
    mod_path = tmp_path / "dummy.py"
    mod_path.write_text("def foo():\n    return 1\n")
    engine = DummyEngine()
    quick_fix = DummyQuickFix()
    pipeline = object()
    orchestrator = EvolutionOrchestrator(event_bus=bus, module_path=mod_path)

    internalize_coding_bot(
        "dummy_bot",
        engine,
        pipeline,
        data_bot=data_bot,
        bot_registry=registry,
        evolution_orchestrator=orchestrator,
        quick_fix=quick_fix,
        roi_threshold=-0.5,
        error_threshold=1.0,
    )

    bus.subscribe("self_coding:cycle_registered", lambda t, e: None)
    bus.subscribe("self_coding:patch_applied", lambda t, e: None)

    data_bot.check_degradation("dummy_bot", roi=-1.0, errors=2.0)

    topics = [t for t, _ in bus.events]
    assert "self_coding:cycle_registered" in topics, "cycle registration not observed"
    assert "self_coding:patch_applied" in topics, "patch application not observed"
    assert quick_fix.calls, "QuickFixEngine did not apply a patch"
