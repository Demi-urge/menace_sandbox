"""Task Validation Bot for verifying and refining tasks."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from dataclasses import asdict
from functools import lru_cache
from typing import Callable, Iterable, List
import logging

_HELPER_NAME = "import_compat"
_PACKAGE_NAME = "menace_sandbox"

try:  # pragma: no cover - prefer package import when installed
    from menace_sandbox import import_compat  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - support flat execution
    _helper_path = Path(__file__).resolve().parent / f"{_HELPER_NAME}.py"
    _spec = importlib.util.spec_from_file_location(
        f"{_PACKAGE_NAME}.{_HELPER_NAME}",
        _helper_path,
    )
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive
        raise
    import_compat = importlib.util.module_from_spec(_spec)
    sys.modules[f"{_PACKAGE_NAME}.{_HELPER_NAME}"] = import_compat
    sys.modules[_HELPER_NAME] = import_compat
    _spec.loader.exec_module(import_compat)
else:  # pragma: no cover - ensure helper aliases exist
    sys.modules.setdefault(_HELPER_NAME, import_compat)
    sys.modules.setdefault(f"{_PACKAGE_NAME}.{_HELPER_NAME}", import_compat)

import_compat.bootstrap(__name__, __file__)
load_internal = import_compat.load_internal
sys.modules.setdefault("menace", importlib.import_module("menace_sandbox"))
sys.modules.setdefault("menace.task_validation_bot", sys.modules[__name__])

dependency_probe = load_internal("self_coding_dependency_probe")
ensure_self_coding_ready = dependency_probe.ensure_self_coding_ready

logger = logging.getLogger(__name__)


def _noop_self_coding(**_kwargs):
    def decorator(cls):
        cls.bot_registry = _kwargs.get("bot_registry")  # type: ignore[attr-defined]
        cls.data_bot = _kwargs.get("data_bot")  # type: ignore[attr-defined]
        cls.manager = _kwargs.get("manager")  # type: ignore[attr-defined]
        return cls

    return decorator


def _resolve_management() -> tuple[
    Callable[..., Callable[[type], type]],
    object | None,
    object | None,
    object | None,
]:
    """Initialise self-coding integration helpers when dependencies permit.

    The self-coding stack is extremely dependency heavy.  On Windows the
    start-up sequence regularly races the import of large modules such as
    ``self_coding_manager`` which, in turn, eagerly pull in the
    ``quick_fix_engine`` and other components.  When that happens during class
    decoration we end up registering the bot without a fully constructed
    manager which later triggers repeated internalisation attempts and stalls
    the sandbox.  To avoid that partial initialisation we only return the real
    decorator when every prerequisite can be created eagerly; otherwise we fall
    back to a no-op decorator so the bot behaves as a regular, non self-coding
    implementation.
    """

    ready, missing = ensure_self_coding_ready()
    if not ready:
        logger.warning(
            "Self-coding integration disabled for TaskValidationBot due to missing dependencies: %s",
            ", ".join(missing),
        )
        return _noop_self_coding, None, None, None

    try:
        registry_cls = load_internal("bot_registry").BotRegistry
        data_bot_cls = load_internal("data_bot").DataBot
        decorator = load_internal("coding_bot_interface").self_coding_managed
        manager_mod = load_internal("self_coding_manager")
        engine_mod = load_internal("self_coding_engine")
        pipeline_mod = load_internal("model_automation_pipeline")
        code_db_cls = load_internal("code_database").CodeDB
        memory_cls = load_internal("gpt_memory").GPTMemoryManager
        ctx_util = load_internal("context_builder_util")
    except ModuleNotFoundError as exc:  # pragma: no cover - optional deps missing
        logger.warning(
            "Self-coding integration disabled for TaskValidationBot: %s", exc
        )
        return _noop_self_coding, None, None, None
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "Failed to import self-coding dependencies for TaskValidationBot: %s",
            exc,
        )
        return _noop_self_coding, None, None, None

    try:
        registry = registry_cls()
        data_bot = data_bot_cls(start_server=False)
        context_builder = ctx_util.create_context_builder()

        def _validator_factory() -> "TaskValidationBot":
            module = sys.modules.get(__name__)
            if module is None or getattr(module, "TaskValidationBot", None) is None:
                module = load_internal("task_validation_bot")
            validator_cls = getattr(module, "TaskValidationBot")
            return validator_cls([])

        engine = engine_mod.SelfCodingEngine(
            code_db_cls(),
            memory_cls(),
            context_builder=context_builder,
        )
        pipeline = pipeline_mod.ModelAutomationPipeline(
            context_builder=context_builder,
            bot_registry=registry,
            validator_factory=_validator_factory,
        )
        manager = manager_mod.SelfCodingManager(
            engine,
            pipeline,
            data_bot=data_bot,
            bot_registry=registry,
        )
    except Exception as exc:  # pragma: no cover - bootstrap degraded
        logger.warning(
            "Self-coding services unavailable for TaskValidationBot: %s",
            exc,
        )
        return _noop_self_coding, None, None, None

    return decorator, registry, data_bot, manager


@lru_cache(maxsize=1)
def _cached_management() -> tuple[
    Callable[..., Callable[[type], type]], object | None, object | None, object | None
]:
    """Resolve and cache self-coding helpers on first use.

    Windows command prompt environments frequently observe slow module imports
    which caused the previous eager initialisation to race the
    :class:`TaskValidationBot` definition.  By deferring the resolution until
    the decorator is actually applied we guarantee the class has been defined
    and avoid the circular-import induced internalisation stalls reported by
    users.
    """

    return _resolve_management()


def _apply_self_coding(cls: type) -> type:
    """Decorate ``cls`` using the lazily resolved self-coding helpers."""

    decorator, registry, data_bot, manager = _cached_management()
    return decorator(bot_registry=registry, data_bot=data_bot, manager=manager)(cls)


SynthesisTask = load_internal("synthesis_models").SynthesisTask

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
try:
    from celery import Celery
except Exception:  # pragma: no cover - optional
    Celery = None  # type: ignore

try:
    from marshmallow import Schema, fields, ValidationError  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    from .simple_validation import (
        SimpleSchema as Schema,
        fields,
        ValidationError,
    )
try:
    import nltk
    from nltk.tokenize import wordpunct_tokenize
except Exception:  # noqa: E722 - optional dependency
    nltk = None  # type: ignore
    wordpunct_tokenize = lambda text: text.split()  # type: ignore
try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None  # type: ignore
import uuid

class TaskSchema(Schema):
    """Schema ensuring task structure."""

    description = fields.Str(required=True)
    urgency = fields.Int(required=True)
    complexity = fields.Int(required=True)
    category = fields.Str(required=True)


@_apply_self_coding
class TaskValidationBot:
    """Validate tasks against goals and structure."""

    def __init__(self, goals: List[str], broker: str = "memory://") -> None:
        self.goals = [g.lower() for g in goals]
        self.schema = TaskSchema()
        if Celery:
            self.app = Celery("validation", broker=broker, backend="rpc://")
        else:  # pragma: no cover - celery unavailable
            from .in_memory_queue import InMemoryQueue

            self.app = InMemoryQueue()
        if zmq:
            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.PAIR)
            addr = f"inproc://validation-{uuid.uuid4()}"
            self.socket.bind(addr)
            self._noblock = zmq.NOBLOCK
        else:  # pragma: no cover - zmq unavailable
            class _StubSocket:
                def __init__(self) -> None:
                    self.sent: list[dict] = []

                def send_json(self, data, *a, **k):
                    self.sent.append(data)

            self.context = None
            self.socket = _StubSocket()
            self._noblock = 0

    def _aligned(self, task: SynthesisTask) -> bool:
        desc = task.description.lower()
        return any(goal in desc for goal in self.goals)

    def _adjust_granularity(self, task: SynthesisTask) -> List[SynthesisTask]:
        tokens = wordpunct_tokenize(task.description)
        if len(tokens) > 8 and " and " in task.description:
            parts = [p.strip() for p in task.description.split(" and ") if p.strip()]
            return [
                SynthesisTask(description=p, urgency=task.urgency, complexity=task.complexity, category=task.category)
                for p in parts
            ]
        if len(tokens) < 3:
            return []
        return [task]

    def _remove_duplicates(self, tasks: List[SynthesisTask]) -> List[SynthesisTask]:
        if pd is None:
            seen = set()
            unique = []
            for t in tasks:
                if t.description not in seen:
                    seen.add(t.description)
                    unique.append(t)
            return unique
        df = pd.DataFrame([asdict(t) for t in tasks])
        df = df.drop_duplicates(subset=["description"])
        return [SynthesisTask(**row) for row in df.to_dict("records")]

    def validate_tasks(self, tasks: Iterable[SynthesisTask]) -> List[SynthesisTask]:
        valid: List[SynthesisTask] = []
        for t in tasks:
            try:
                self.schema.load(asdict(t))
            except ValidationError:
                self.app.send_task("research.refine", kwargs={"task": t.description})
                continue
            if not self._aligned(t):
                self.app.send_task("research.refine", kwargs={"task": t.description})
                continue
            refined = self._adjust_granularity(t)
            if not refined:
                self.app.send_task("research.refine", kwargs={"task": t.description})
                continue
            valid.extend(refined)
        valid = self._remove_duplicates(valid)
        for t in valid:
            try:
                self.socket.send_json(asdict(t), flags=self._noblock)
            except Exception as exc:
                logger.warning("failed sending task over socket: %s", exc)
        return valid


__all__ = ["TaskValidationBot", "TaskSchema"]