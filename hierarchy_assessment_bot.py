"""Hierarchy Assessment Bot for overseeing bot hierarchy and preventing duplication."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import json
import logging
from dataclasses import dataclass
from typing import List, Dict

registry = BotRegistry()
data_bot = DataBot(start_server=False)

logger = logging.getLogger(__name__)

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None  # type: ignore
import psutil
import uuid
try:
    import pika  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pika = None  # type: ignore
try:
    import risky  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    risky = None  # type: ignore


@dataclass
class BotTaskRecord:
    """Track a task executed by a bot."""

    bot: str
    task: str
    completed: bool = False


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class HierarchyAssessmentBot:
    """Coordinate bots, monitor redundancy and system health."""

    def __init__(
        self,
        planning_api: str = "http://localhost:8000/plan",
        mq_url: str = "amqp://guest:guest@localhost:5672/%2F",
    ) -> None:
        self.planning_api = planning_api
        if zmq:
            self.context = zmq.Context.instance()
            self.socket = self.context.socket(zmq.PAIR)
            addr = f"inproc://hierarchy-{uuid.uuid4()}"
            self.socket.bind(addr)
            self._noblock = zmq.NOBLOCK
            self._error = zmq.ZMQError
        else:  # pragma: no cover - zmq unavailable
            class _StubSocket:
                def send_json(self, *a, **k):
                    logger.debug("stub socket send_json")

            self.context = None
            self.socket = _StubSocket()
            self._noblock = 0
            self._error = Exception
        self.tasks: List[BotTaskRecord] = []
        self.channel = None
        if pika:
            try:
                params = pika.URLParameters(mq_url)
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="oversight", durable=True)
                self.channel = ch
            except Exception:  # pragma: no cover - external service
                self.channel = None

    def register(self, bot: str, task: str) -> None:
        self.tasks.append(BotTaskRecord(bot=bot, task=task))

    def complete(self, bot: str, task: str) -> None:
        for rec in self.tasks:
            if rec.bot == bot and rec.task == task:
                rec.completed = True
                self._trigger_oversight(bot, task)
                break

    def _trigger_oversight(self, bot: str, task: str) -> None:
        msg = {"bot": bot, "task": task}
        try:
            self.socket.send_json(msg, flags=self._noblock)
        except self._error:  # pragma: no cover - if no listener
            logger.debug("no oversight listener")
        if self.channel:
            try:
                self.channel.basic_publish("", "oversight", json.dumps(msg))
            except Exception:  # pragma: no cover - external
                logger.exception("failed publishing oversight event")

    def redundancy_analysis(self) -> List[str]:
        seen = set()
        overlaps = []
        for rec in self.tasks:
            if rec.task in seen and not rec.completed:
                overlaps.append(rec.task)
            seen.add(rec.task)
        if overlaps:
            try:
                requests.post(self.planning_api, json={"overlaps": overlaps}, timeout=3)
            except Exception:  # pragma: no cover - network failure
                logger.exception("redundancy analysis notification failed")
        return overlaps

    def assess_risk(self) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for rec in self.tasks:
            risk = 0.0
            if risky:
                try:
                    risk_func = getattr(risky, "risk", None)
                    if callable(risk_func):
                        risk = float(risk_func(rec.task))
                except Exception:
                    logger.exception("risk function failed")
                    risk = 0.0
            scores[rec.task] = risk
        return scores

    def monitor_system(self, limit: float = 90.0) -> bool:
        if psutil.cpu_percent() > limit:
            self.trigger_contingency()
            return True
        return False

    def trigger_contingency(self) -> None:
        if self.channel:
            try:
                self.channel.basic_publish("", "oversight", "contingency")
            except Exception:  # pragma: no cover - external
                logger.exception("contingency publish failed")


__all__ = ["BotTaskRecord", "HierarchyAssessmentBot"]