"""Task Validation Bot for verifying and refining tasks."""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from dataclasses import asdict
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
]:
    try:
        registry_cls = load_internal("bot_registry").BotRegistry
        data_bot_cls = load_internal("data_bot").DataBot
        decorator = load_internal("coding_bot_interface").self_coding_managed
    except ModuleNotFoundError as exc:  # pragma: no cover - optional deps missing
        logger.warning(
            "Self-coding integration disabled for TaskValidationBot: %s", exc
        )
        return _noop_self_coding, None, None

    try:
        registry = registry_cls()
        data_bot = data_bot_cls(start_server=False)
    except Exception as exc:  # pragma: no cover - bootstrap degraded
        logger.warning(
            "Failed to initialise self-coding services for TaskValidationBot: %s",
            exc,
        )
        return _noop_self_coding, None, None

    return decorator, registry, data_bot


_self_coding_managed, registry, data_bot = _resolve_management()
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


@_self_coding_managed(bot_registry=registry, data_bot=data_bot)
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