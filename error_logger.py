"""Middleware and utilities for high-resolution error telemetry."""

from __future__ import annotations

import logging
import os
import re
import traceback
import json
import hashlib
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Iterable, Optional

from .sentry_client import SentryClient

try:
    from .advanced_error_management import TelemetryReplicator  # type: ignore
except Exception:  # pragma: no cover - optional
    TelemetryReplicator = None  # type: ignore

from pydantic import BaseModel

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .error_bot import ErrorDB

try:
    from sentence_transformers import SentenceTransformer, util  # type: ignore
except Exception:  # pragma: no cover - optional
    SentenceTransformer = None  # type: ignore
    util = None  # type: ignore


class TelemetryEvent(BaseModel):
    """Single error telemetry record."""

    task_id: str | None = None
    bot_id: str | None = None
    error_type: str = ""
    stack_trace: str = ""
    root_module: str = ""
    timestamp: str = datetime.utcnow().isoformat()
    resolution_status: str = "unresolved"
    patch_id: int | None = None
    deploy_id: int | None = None
    checksum: str = ""


class ErrorClassifier:
    """Categorise errors via regex and semantic matching."""

    regex_map = {
        r"KeyError": "Runtime/Reference",
        r"IndexError": "Runtime/Index",
        r"FileNotFoundError": "Runtime/IO",
    }
    semantic_map = {
        "missing key": "Runtime/Reference",
        "key not found": "Runtime/Reference",
        "not in index": "Runtime/Index",
    }

    def __init__(self) -> None:
        self.logger = logging.getLogger("ErrorClassifier")
        if SentenceTransformer and util:
            try:
                self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except Exception:  # pragma: no cover - runtime download issues
                self.model = None
        else:
            self.model = None

    def classify(self, stack: str) -> str:
        for pattern, label in self.regex_map.items():
            if re.search(pattern, stack, re.IGNORECASE):
                return label
        low = stack.lower()
        for phrase, label in self.semantic_map.items():
            if phrase in low:
                return label
        if self.model:
            try:
                emb = self.model.encode([low])[0]
                best_score = 0.0
                best_label = "unknown"
                for phrase, label in self.semantic_map.items():
                    sim = float(util.cos_sim(emb, self.model.encode([phrase])[0]))
                    if sim > best_score:
                        best_score = sim
                        best_label = label
                if best_score > 0.5:
                    return best_label
            except Exception as e:  # pragma: no cover - runtime issues
                self.logger.warning("semantic classification failed: %s", e)
        return "unknown"


class ErrorLogger:
    """Wrap functions to capture exceptions and log telemetry."""

    def __init__(
        self, db: "ErrorDB" | None = None, *, sentry: "SentryClient" | None = None
    ) -> None:
        if db is None:
            from .error_bot import ErrorDB as _ErrorDB
            db = _ErrorDB()
        self.db = db
        self.classifier = ErrorClassifier()
        self.logger = logging.getLogger("ErrorLogger")
        self.sentry = sentry
        self.replicator = None
        if TelemetryReplicator:
            hosts = os.getenv("KAFKA_HOSTS")
            disk = os.getenv("TELEMETRY_LOG", "telemetry.log")
            if hosts:
                try:
                    self.replicator = TelemetryReplicator(
                        hosts=hosts,
                        sentry=self.sentry,
                        disk_path=disk,
                    )
                    self.replicator.flush()
                except Exception as e:  # pragma: no cover - optional
                    self.logger.error("failed to init TelemetryReplicator: %s", e)

    def log(
        self,
        exc: Exception,
        task_id: Optional[str],
        bot_id: Optional[str],
        *,
        patch_id: Optional[int] = None,
        deploy_id: Optional[int] = None,
    ) -> None:
        stack = traceback.format_exc()
        if patch_id is None:
            env = os.getenv("PATCH_ID")
            if env and env.isdigit():
                patch_id = int(env)
        if deploy_id is None:
            env = os.getenv("DEPLOY_ID")
            if env and env.isdigit():
                deploy_id = int(env)
        event = TelemetryEvent(
            task_id=task_id,
            bot_id=bot_id,
            error_type=self.classifier.classify(stack),
            stack_trace=stack,
            root_module=exc.__class__.__module__.split(".")[0],
            timestamp=datetime.utcnow().isoformat(),
            resolution_status="unresolved",
            patch_id=patch_id,
            deploy_id=deploy_id,
        )
        payload = json.dumps(event.dict(exclude={"checksum"}), sort_keys=True).encode("utf-8")
        event.checksum = hashlib.sha256(payload).hexdigest()
        try:
            self.db.add_telemetry(event)
        except Exception as e:  # pragma: no cover - db issues
            self.logger.error("failed to record telemetry: %s", e)
        if self.replicator:
            try:
                self.replicator.replicate(event)
            except Exception as e:  # pragma: no cover - network issues
                self.logger.error("failed to replicate telemetry: %s", e)
        if self.sentry:
            try:
                self.sentry.capture_exception(exc)
            except Exception as e:
                self.logger.warning("failed to send exception to Sentry: %s", e)

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            task_id = kwargs.get("task_id")
            bot_id = getattr(args[0], "name", None) if args else None
            patch_id = kwargs.get("patch_id")
            deploy_id = kwargs.get("deploy_id")
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - runtime
                self.log(exc, task_id, bot_id, patch_id=patch_id, deploy_id=deploy_id)
                raise
        return wrapper


__all__ = ["TelemetryEvent", "ErrorLogger", "ErrorClassifier"]
