"""Task Validation Bot for verifying and refining tasks."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
from dataclasses import asdict
from typing import Iterable, List

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
import logging

logger = logging.getLogger(__name__)

from menace.information_synthesis_bot import SynthesisTask



class TaskSchema(Schema):
    """Schema ensuring task structure."""

    description = fields.Str(required=True)
    urgency = fields.Int(required=True)
    complexity = fields.Int(required=True)
    category = fields.Str(required=True)


@self_coding_managed
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
