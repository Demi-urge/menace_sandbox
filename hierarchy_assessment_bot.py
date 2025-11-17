"""Hierarchy Assessment Bot for overseeing bot hierarchy and preventing duplication."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import json
import logging
import os
import socket
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _noop_self_coding_managed(**_kwargs: object) -> Callable[[type], type]:
    def decorator(cls: type) -> type:
        return cls

    return decorator


def _self_coding_disabled() -> bool:
    if _truthy_env("SANDBOX_DISABLE_SELF_CODING"):
        return True
    if _truthy_env("SANDBOX_DISABLE_HIERARCHY_SELF_CODING"):
        return True
    if _truthy_env("SANDBOX_ENABLE_HIERARCHY_SELF_CODING"):
        return False
    if sys.platform.startswith("win") and not _truthy_env(
        "SANDBOX_ENABLE_WINDOWS_SELF_CODING"
    ):
        return True
    return True


def _can_reach_broker(mq_url: str, timeout: float = 0.25) -> bool:
    """Return True when the RabbitMQ broker is reachable."""

    try:
        parsed = urlparse(mq_url)
        if not parsed.hostname:
            return False
        port = parsed.port or 5672
        with socket.create_connection((parsed.hostname, port), timeout=timeout):
            return True
    except OSError as exc:
        logger.debug("RabbitMQ broker not reachable at %s: %s", mq_url, exc)
        return False
    except Exception as exc:
        logger.debug("RabbitMQ broker check failed for %s: %s", mq_url, exc)
        return False


if _self_coding_disabled():
    _self_coding_decorator = _noop_self_coding_managed
    logger.info(
        "HierarchyAssessmentBot self-coding disabled; using passive decorator"
    )
else:
    _self_coding_decorator = self_coding_managed


@lru_cache(maxsize=1)
def _registry_singleton() -> BotRegistry:
    return BotRegistry()


def _get_registry() -> BotRegistry:
    return _registry_singleton()


_get_registry.__self_coding_lazy__ = True  # type: ignore[attr-defined]


@lru_cache(maxsize=1)
def _data_bot_singleton() -> DataBot:
    return DataBot(start_server=False)


def _get_data_bot() -> DataBot:
    return _data_bot_singleton()


_get_data_bot.__self_coding_lazy__ = True  # type: ignore[attr-defined]

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore
try:
    import zmq  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    zmq = None  # type: ignore
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore
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


@(_self_coding_decorator(bot_registry=_get_registry, data_bot=_get_data_bot))
class HierarchyAssessmentBot:
    """Coordinate bots, monitor redundancy and system health."""

    def __init__(
        self,
        planning_api: str = "http://localhost:8000/plan",
        mq_url: str = "amqp://guest:guest@localhost:5672/%2F",
        *,
        manager: "SelfCodingManager | None" = None,
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
        if pika and _can_reach_broker(mq_url):
            try:
                params = pika.URLParameters(mq_url)
                conn = pika.BlockingConnection(params)
                ch = conn.channel()
                ch.queue_declare(queue="oversight", durable=True)
                self.channel = ch
            except Exception:  # pragma: no cover - external service
                logger.info(
                    "RabbitMQ unreachable at %s; using local oversight queue",
                    mq_url,
                )
                self.channel = None
        elif mq_url:
            logger.info(
                "RabbitMQ not available at %s; using in-process oversight queue",
                mq_url,
            )

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
        if psutil is None:
            logger.info(
                "psutil not available; skipping system monitoring safety check"
            )
            return False
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
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]