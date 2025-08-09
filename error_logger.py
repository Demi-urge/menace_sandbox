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
from typing import Any, Callable, Optional

from .sentry_client import SentryClient

try:
    from .advanced_error_management import TelemetryReplicator  # type: ignore
except Exception:  # pragma: no cover - optional
    TelemetryReplicator = None  # type: ignore

from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

from .error_ontology import ErrorType

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
    error_type: ErrorType = ErrorType.UNKNOWN
    stack_trace: str = ""
    root_module: str = ""
    module: str = ""
    module_counts: dict[str, int] = Field(default_factory=dict)
    inferred_cause: str = ""
    timestamp: str = datetime.utcnow().isoformat()
    resolution_status: str = "unresolved"
    patch_id: int | None = None
    deploy_id: int | None = None
    checksum: str = ""


DEFAULT_CLASSIFICATION_RULES = {
    "RuntimeFault": {
        "regex": [r"KeyError", r"IndexError", r"FileNotFoundError"],
        "semantic": ["missing key", "key not found", "not in index"],
    },
    "DependencyMismatch": {
        "regex": [r"ModuleNotFoundError|ImportError"],
        "semantic": ["module not found", "dependency missing"],
    },
    "LogicMisfire": {
        "regex": [r"AssertionError"],
        "semantic": ["assertion failed", "not implemented"],
    },
    "SemanticBug": {
        "regex": [r"TypeError"],
        "semantic": ["unexpected type", "wrong type"],
    },
}


class ErrorClassifier:
    """Categorise errors via regex and semantic matching."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: str | None = None,
    ) -> None:
        self.logger = logging.getLogger("ErrorClassifier")
        if config is None:
            if config_path is None:
                config_path = os.getenv("ERROR_CLASSIFIER_CONFIG")
            if config_path and os.path.exists(config_path):
                try:
                    with open(config_path, "r", encoding="utf-8") as fh:
                        config = json.load(fh)
                except Exception as e:  # pragma: no cover - configuration errors
                    self.logger.warning("failed to load config %s: %s", config_path, e)
                    config = DEFAULT_CLASSIFICATION_RULES
            else:
                config = DEFAULT_CLASSIFICATION_RULES

        self.regex_map: dict[str, ErrorType] = {}
        self.semantic_map: dict[str, ErrorType] = {}

        for name, rules in config.items():
            err_type = self._parse_type(name)
            if not err_type:
                continue
            for pattern in rules.get("regex", []):
                self.regex_map[pattern] = err_type
            for phrase in rules.get("semantic", []):
                self.semantic_map[phrase.lower()] = err_type

        if SentenceTransformer and util:
            try:
                self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
            except Exception:  # pragma: no cover - runtime download issues
                self.model = None
        else:
            self.model = None

    @staticmethod
    def _parse_type(name: str) -> ErrorType | None:
        import re

        key = re.sub(r"(?<!^)(?=[A-Z])", "_", name)
        key = key.replace("-", "_").replace(" ", "_").upper()
        try:
            return ErrorType[key]
        except KeyError:
            try:
                return ErrorType(name.lower())
            except Exception:  # pragma: no cover - unknown type
                return None

    def classify(self, stack: str) -> ErrorType:
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
                best_label: ErrorType | None = None
                for phrase, label in self.semantic_map.items():
                    sim = float(util.cos_sim(emb, self.model.encode([phrase])[0]))
                    if sim > best_score:
                        best_score = sim
                        best_label = label
                if best_label and best_score > 0.5:
                    return best_label
            except Exception as e:  # pragma: no cover - runtime issues
                self.logger.warning("semantic classification failed: %s", e)
        return ErrorType.UNKNOWN


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
        module = ""
        module_counts: dict[str, int] = {}
        tb = traceback.extract_tb(exc.__traceback__)
        for frame in tb:
            mod = os.path.splitext(os.path.basename(frame.filename))[0]
            if mod:
                module_counts[mod] = module_counts.get(mod, 0) + 1
        if tb:
            module = os.path.splitext(os.path.basename(tb[-1].filename))[0]
        root_module = max(module_counts, key=module_counts.get, default="")
        cause = str(exc)
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
            root_module=root_module,
            module=module,
            module_counts=module_counts,
            inferred_cause=cause,
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
