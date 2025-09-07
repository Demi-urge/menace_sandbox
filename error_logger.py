"""Middleware and utilities for high-resolution error telemetry."""

from __future__ import annotations

import logging
import os
import re
import traceback
import json
import hashlib
import math
from pathlib import Path
try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Optional

from .db_router import GLOBAL_ROUTER, init_db_router
from .scope_utils import Scope, build_scope_clause, apply_scope

try:
    from .sentry_client import SentryClient
except ImportError:  # pragma: no cover - package fallback
    from sentry_client import SentryClient  # type: ignore

try:
    from .advanced_error_management import TelemetryReplicator  # type: ignore
except Exception:  # pragma: no cover - optional
    try:
        from advanced_error_management import TelemetryReplicator  # type: ignore
    except Exception:
        TelemetryReplicator = None  # type: ignore

from pydantic import BaseModel, Field

from typing import TYPE_CHECKING

try:
    from .error_ontology import ErrorCategory, classify_exception
except ImportError:  # pragma: no cover - package fallback
    from error_ontology import ErrorCategory, classify_exception  # type: ignore

try:  # pragma: no cover - optional dependency
    from . import codex_db_helpers as cdh
except Exception:  # pragma: no cover - optional dependency
    cdh = None  # type: ignore

try:
    from .knowledge_graph import KnowledgeGraph
except ImportError:  # pragma: no cover - package fallback
    from knowledge_graph import KnowledgeGraph  # type: ignore

try:
    from .roi_calculator import propose_fix
except Exception:  # pragma: no cover - package fallback
    from roi_calculator import propose_fix  # type: ignore

try:  # pragma: no cover - optional dependency
    from .quick_fix_engine import generate_patch
    from vector_service import ContextBuilder
except Exception:
    try:
        from quick_fix_engine import generate_patch  # type: ignore
        from vector_service import ContextBuilder  # type: ignore
    except Exception:
        generate_patch = None  # type: ignore
        ContextBuilder = None  # type: ignore

from governed_embeddings import governed_embed, get_embedder
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path, path_for_prompt
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path, path_for_prompt  # type: ignore

# Backwards compatibility with legacy imports
ErrorType = ErrorCategory

if TYPE_CHECKING:  # pragma: no cover - type hints only
    try:
        from .error_bot import ErrorDB
    except ImportError:
        from error_bot import ErrorDB  # type: ignore

try:
    from sentence_transformers import util  # type: ignore
except Exception:  # pragma: no cover - optional
    util = None  # type: ignore


def _cosine(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


try:  # pragma: no cover - optional micro model
    from .micro_models.error_classifier import classify_error
    from .micro_models.prefix_injector import inject_prefix
except Exception:  # pragma: no cover - fallback when package layout differs
    try:
        from micro_models.error_classifier import classify_error  # type: ignore
        from micro_models.prefix_injector import inject_prefix  # type: ignore
    except Exception:  # pragma: no cover - micro model unavailable
        classify_error = None  # type: ignore
        inject_prefix = (
            lambda prompt, prefix, conf, role="system": prompt
        )  # type: ignore


class TelemetryEvent(BaseModel):
    """Single error telemetry record."""

    task_id: str | None = None
    bot_id: str | None = None
    # Normalised ontology identifier for the error category
    error_type: ErrorCategory = ErrorCategory.Unknown
    # Legacy alias retained for backward compatibility
    category: ErrorCategory = ErrorCategory.Unknown
    root_cause: str = ""
    stack_trace: str = ""
    root_module: str = ""
    module: str = ""
    module_counts: dict[str, int] = Field(default_factory=dict)
    inferred_cause: str = ""
    timestamp: str = datetime.utcnow().isoformat()
    resolution_status: str = "unresolved"
    patch_id: int | None = None
    deploy_id: int | None = None
    fix_suggestions: list[str] | None = None
    bottlenecks: list[str] | None = None
    checksum: str = ""


DEFAULT_CLASSIFICATION_RULES = {
    "RuntimeFault": {
        "regex": [
            r"KeyError",
            r"IndexError",
            r"FileNotFoundError",
            r"ZeroDivisionError",
            r"AttributeError",
        ],
        "semantic": [
            "missing key",
            "key not found",
            "not in index",
            "division by zero",
            "attribute not found",
        ],
    },
    "DependencyMismatch": {
        "regex": [
            r"ModuleNotFoundError|ImportError",
            r"PackageNotFoundError",
            r"OSError",
        ],
        "semantic": [
            "module not found",
            "dependency missing",
            "missing package",
            "no module named",
            "cannot import name",
            "dependency conflict",
            "version conflict",
        ],
    },
    "LogicMisfire": {
        "regex": [r"AssertionError", r"NotImplementedError"],
        "semantic": ["assertion failed", "not implemented"],
    },
    "SemanticBug": {
        "regex": [r"TypeError", r"ValueError"],
        "semantic": ["unexpected type", "wrong type", "invalid value"],
    },
    "ResourceLimit": {
        "regex": [r"MemoryError"],
        "semantic": ["out of memory", "memory limit"],
    },
    "Timeout": {
        "regex": [r"TimeoutError"],
        "semantic": ["timed out", "timeout"],
    },
    "ExternalAPI": {
        "regex": [r"ConnectionError", r"HTTPError"],
        "semantic": ["external api", "service unavailable", "connection refused"],
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
        self.config_path = config_path or os.getenv("ERROR_CLASSIFIER_CONFIG")
        self.config_mtime = 0.0
        if config is None:
            config = self._load_config()
        else:
            self.config_path = None
        self._build_maps(config)

        if util:
            self.model = get_embedder()
        else:
            self.model = None

    @staticmethod
    def _parse_type(name: str) -> ErrorCategory | None:
        import re

        key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        key = key.replace("-", "_").replace(" ", "_").upper()
        try:
            return ErrorCategory[key]
        except KeyError:
            try:
                return ErrorCategory(name.lower())
            except Exception:  # pragma: no cover - unknown type
                return None

    def _build_maps(self, config: dict[str, Any]) -> None:
        self.regex_map = {}
        self.semantic_map = {}
        for name, rules in config.items():
            err_type = self._parse_type(name)
            if not err_type:
                continue
            for pattern in rules.get("regex", []):
                self.regex_map[pattern] = err_type
            for phrase in rules.get("semantic", []):
                self.semantic_map[phrase.lower()] = err_type

    def _load_config(self) -> dict[str, Any]:
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    if self.config_path.endswith((".yml", ".yaml")) and yaml:
                        data = yaml.safe_load(fh) or {}
                    else:
                        data = json.load(fh)
                self.config_mtime = os.path.getmtime(self.config_path)
                return data
            except Exception as e:  # pragma: no cover - configuration errors
                self.logger.warning(
                    "failed to load config %s: %s", self.config_path, e
                )
        self.config_mtime = 0.0
        return DEFAULT_CLASSIFICATION_RULES

    def _maybe_reload(self) -> None:
        if not self.config_path:
            return
        try:
            mtime = os.path.getmtime(self.config_path)
        except OSError:  # pragma: no cover - file removed
            return
        if mtime != self.config_mtime:
            config = self._load_config()
            self._build_maps(config)

    def learn_error_phrases(
        self,
        db: "ErrorDB",
        *,
        min_count: int = 5,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> None:
        """Learn frequent error phrases from telemetry and persist rules.

        Parameters:
            db: ``ErrorDB`` instance providing a ``conn`` attribute.
            min_count: minimum occurrences required before adding a rule.
            scope: menace visibility for telemetry selection.
            source_menace_id: override menace identifier used for scoping.
        """

        menace_id = db._menace_id(source_menace_id)
        clause, scope_params = build_scope_clause("telemetry", Scope(scope), menace_id)

        # First, allow the micro-model to annotate telemetry with high confidence
        # predictions so that subsequent heuristic learning has richer data.
        if classify_error is not None:
            try:
                query = (
                    "SELECT id, stack_trace FROM telemetry "
                    "WHERE (category IS NULL OR TRIM(category) = '' OR category = 'Unknown') "
                    "AND stack_trace IS NOT NULL AND TRIM(stack_trace) != ''"
                )
                query = apply_scope(query, clause)
                cur = db.conn.execute(query, scope_params)
                rows = cur.fetchall()
                for row in rows:
                    category, fix, conf = classify_error(
                        row["stack_trace"]
                    )  # type: ignore[arg-type]
                    prompt = inject_prefix(
                        row["stack_trace"],
                        f"Error Category: {category}\nSuggested Fix: {fix}",
                        conf,
                        role="system",
                    )
                    self.logger.debug("codex prompt prepared: %s", prompt)
                    if conf > 0.8 and category:
                        db.conn.execute(
                            "UPDATE telemetry SET category = ?, inferred_cause = ? WHERE id = ?",
                            (category, fix, row["id"]),
                        )
                if rows:
                    db.conn.commit()
            except Exception as exc:  # pragma: no cover - micro model failures
                self.logger.debug("micro-model classification skipped: %s", exc)

        try:
            base = (
                "SELECT category, COALESCE(cause, inferred_cause) AS phrase, COUNT(*) AS c "
                "FROM telemetry WHERE COALESCE(cause, inferred_cause) IS NOT NULL "
                "AND TRIM(COALESCE(cause, inferred_cause)) != ''"
            )
            base = apply_scope(base, clause)
            query = f"{base} GROUP BY category, phrase HAVING c >= ?"
            cur = db.conn.execute(query, [*scope_params, min_count])
            rows = cur.fetchall()
        except Exception as exc:  # pragma: no cover - db failures
            self.logger.warning("rule learning failed: %s", exc)
            return

        if not rows:
            return

        config: dict[str, Any] = (
            self._load_config() if self.config_path else DEFAULT_CLASSIFICATION_RULES.copy()
        )
        updated = False
        for category, phrase, _cnt in rows:
            err_type = self._parse_type(category or "")
            if not err_type:
                continue
            if not phrase:
                continue
            phrase = str(phrase)
            lower = phrase.lower()

            # Add as regex if it looks like an Error subclass name
            if re.match(r"^[A-Za-z_]*Error$", phrase) and phrase not in self.regex_map:
                self.regex_map[phrase] = err_type
                rules = config.setdefault(err_type.name, {})
                regex_list = rules.setdefault("regex", [])
                if phrase not in regex_list:
                    regex_list.append(phrase)
                    updated = True
            elif lower not in self.semantic_map:
                self.semantic_map[lower] = err_type
                rules = config.setdefault(err_type.name, {})
                sem_list = rules.setdefault("semantic", [])
                if phrase not in sem_list:
                    sem_list.append(phrase)
                    updated = True

        if updated and self.config_path:
            try:
                with open(self.config_path, "w", encoding="utf-8") as fh:
                    if self.config_path.endswith((".yml", ".yaml")) and yaml:
                        yaml.safe_dump(config, fh, sort_keys=False)
                    else:
                        json.dump(config, fh, indent=2)
                self.config_mtime = os.path.getmtime(self.config_path)
            except Exception as exc:  # pragma: no cover - write errors
                self.logger.warning("failed to persist learned rules: %s", exc)

        if updated:
            self.logger.info("learned %d new error rules", len(rows))

    def update_rules_from_db(
        self,
        db: "ErrorDB",
        *,
        min_count: int = 5,
        scope: Scope | str = "local",
        source_menace_id: str | None = None,
    ) -> None:  # pragma: no cover - backward compatibility
        self.learn_error_phrases(
            db,
            min_count=min_count,
            scope=scope,
            source_menace_id=source_menace_id,
        )

    def classify(self, stack: str, exc: Exception | None = None) -> ErrorCategory:
        """Classify a stack trace, prioritising ontology categories."""
        self._maybe_reload()
        if exc is not None:
            ont = classify_exception(exc, stack)
            if ont is not ErrorCategory.Unknown:
                return ont
        for pattern, label in self.regex_map.items():
            if re.search(pattern, stack, re.IGNORECASE):
                return label
        low = stack.lower()
        for phrase, label in self.semantic_map.items():
            if phrase in low:
                return label
        if self.model:
            try:
                emb = governed_embed(low, self.model)
                if emb is not None:
                    best_score = 0.0
                    best_label: ErrorCategory | None = None
                    for phrase, label in self.semantic_map.items():
                        p_emb = governed_embed(phrase, self.model)
                        if p_emb is None:
                            continue
                        if util:
                            sim = float(util.cos_sim(emb, p_emb))
                        else:
                            sim = _cosine(emb, p_emb)
                        if sim > best_score:
                            best_score = sim
                            best_label = label
                    if best_label and best_score > 0.5:
                        return best_label
            except Exception as e:  # pragma: no cover - runtime issues
                self.logger.warning("semantic classification failed: %s", e)
        return ErrorCategory.Unknown

    def classify_details(self, exc: Exception, stack: str) -> tuple[ErrorCategory, str]:
        """Return ontology category and root cause."""
        category = self.classify(stack, exc)
        root_cause = exc.__class__.__name__ if category is not ErrorCategory.Unknown else ""
        return category, root_cause


class ErrorLogger:
    """Wrap functions to capture exceptions and log telemetry."""

    def __init__(
        self,
        db: "ErrorDB" | None = None,
        *,
        sentry: "SentryClient" | None = None,
        knowledge_graph: "KnowledgeGraph" | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        if db is None:
            try:
                from .error_bot import ErrorDB as _ErrorDB
            except ImportError:  # pragma: no cover - package fallback
                try:
                    from error_bot import ErrorDB as _ErrorDB  # type: ignore
                except Exception:  # pragma: no cover - optional
                    _ErrorDB = None
            if _ErrorDB is None:
                class _StubDB:
                    def add_telemetry(self, *a: Any, **k: Any) -> None:  # pragma: no cover - no-op
                        pass

                db = _StubDB()
            else:
                if GLOBAL_ROUTER is None:
                    init_db_router("errors")
                db = _ErrorDB()
        self.db = db
        self.classifier = ErrorClassifier()
        self.logger = logging.getLogger("ErrorLogger")
        self.sentry = sentry
        self.graph = knowledge_graph
        self.context_builder = context_builder
        self.replicator = None
        self._unknown_counter = 0
        self._update_threshold = int(
            os.getenv("ERROR_RULE_UPDATE_THRESHOLD", "10")
        )
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
    ) -> TelemetryEvent:
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
        category, root_cause = self.classifier.classify_details(exc, stack)
        event = TelemetryEvent(
            task_id=task_id,
            bot_id=bot_id,
            error_type=category,
            category=category,
            root_cause=root_cause,
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
        if self.graph:
            try:
                self.graph.add_telemetry_event(
                    bot_id,
                    category.value,
                    root_module,
                    module_counts,
                    patch_id=patch_id,
                    deploy_id=deploy_id,
                )
                self.graph.add_error_instance(
                    category.value, module, root_cause
                )
            except Exception as e:  # pragma: no cover - graph issues
                self.logger.error("failed to update knowledge graph: %s", e)
        if self.replicator:
            try:
                self.replicator.replicate(event)
            except Exception as e:  # pragma: no cover - network issues
                self.logger.error("failed to replicate telemetry: %s", e)
        if category is ErrorCategory.Unknown:
            self._unknown_counter += 1
            if self._unknown_counter >= self._update_threshold:
                try:
                    self.classifier.learn_error_phrases(self.db)
                except Exception as e:
                    self.logger.warning("rule update failed: %s", e)
                self._unknown_counter = 0
        if self.sentry:
            try:
                self.sentry.capture_exception(exc)
            except Exception as e:
                self.logger.warning("failed to send exception to Sentry: %s", e)
        return event

    def log_roi_cap(
        self,
        workflow_id: str,
        metrics: dict[str, float],
        profile: dict[str, Any],
    ) -> list[tuple[str, str]]:
        """Emit telemetry for ROI bottlenecks and suggest fixes."""
        suggestions = propose_fix(metrics, profile)
        payload = {
            "workflow_id": workflow_id,
            "metrics": metrics,
            "suggestions": suggestions,
        }
        message = f"ROIBottleneck: {json.dumps(payload, sort_keys=True)}"
        if self.replicator:
            try:
                event = TelemetryEvent(
                    task_id=workflow_id,
                    error_type=ErrorCategory.Unknown,
                    category=ErrorCategory.Unknown,
                    root_cause="ROIBottleneck",
                    stack_trace=message,
                    root_module="roi_calculator",
                    module="roi_calculator",
                    module_counts={},
                    inferred_cause="",
                    timestamp=datetime.utcnow().isoformat(),
                    resolution_status="unresolved",
                    patch_id=None,
                    deploy_id=None,
                )
                self.replicator.replicate(event)
            except Exception as e:  # pragma: no cover - network issues
                self.logger.error("failed to replicate ROI telemetry: %s", e)
        elif self.sentry:
            try:
                self.sentry.capture_exception(Exception(message))
            except Exception as e:  # pragma: no cover - telemetry issues
                self.logger.warning("failed to send ROI bottleneck to Sentry: %s", e)
        else:
            self.logger.info(message)
        return suggestions

    def log_fix_suggestions(
        self,
        metrics: dict[str, float],
        profile: dict[str, Any] | str,
        task_id: str | None = None,
        bot_id: str | None = None,
        *,
        sample_limit: int = 5,
        sample_sort_by: str = "confidence",
        with_vectors: bool = True,
    ) -> list[TelemetryEvent]:
        """Derive and record fix suggestions for metric bottlenecks.

        Training examples are pulled via :func:`codex_db_helpers.aggregate_samples`
        and discrepancy samples via :func:`codex_db_helpers.fetch_discrepancies`
        to enrich Codex prompts.
        """

        suggestions = propose_fix(metrics, profile)
        events: list[TelemetryEvent] = []

        samples = []
        discrepancies = []
        if cdh is not None:
            try:
                samples = cdh.aggregate_samples(
                    sort_by=sample_sort_by,
                    limit=sample_limit,
                    include_embeddings=with_vectors,
                    scope=Scope.ALL,
                )
            except Exception:  # pragma: no cover - helper failures
                samples = []
            try:
                discrepancies = cdh.fetch_discrepancies(
                    sort_by="confidence",
                    limit=sample_limit,
                    include_embeddings=False,
                    scope=Scope.ALL,
                )
            except Exception:  # pragma: no cover - helper failures
                discrepancies = []

        prompt_context = "\n".join(
            s.content for s in samples if getattr(s, "content", "")
        )
        discrepancy_context = "\n".join(
            d.content for d in discrepancies if getattr(d, "content", "")
        )

        for module, hint in suggestions:
            resolved_module = path_for_prompt(module) if module else None
            payload = {
                "task_id": task_id,
                "bot_id": bot_id,
                "module": resolved_module,
                "suggestion": hint,
            }
            event = TelemetryEvent(
                task_id=task_id,
                bot_id=bot_id,
                error_type=ErrorCategory.MetricBottleneck,
                category=ErrorCategory.MetricBottleneck,
                root_cause=hint,
                stack_trace=json.dumps(payload, sort_keys=True),
                root_module=resolved_module or "fix_suggestions",
                module=resolved_module or "fix_suggestions",
                module_counts={resolved_module: 1} if resolved_module else {},
                inferred_cause="",
                timestamp=datetime.utcnow().isoformat(),
                resolution_status="unresolved",
                patch_id=None,
                deploy_id=None,
                fix_suggestions=[hint],
                bottlenecks=[resolved_module] if resolved_module else [],
            )
            try:
                self.db.add_telemetry(event)
            except Exception as e:  # pragma: no cover - db issues
                self.logger.error("failed to record telemetry: %s", e)

            if self.graph and bot_id:
                try:
                    self.graph.add_telemetry_event(
                        bot_id,
                        ErrorCategory.MetricBottleneck.value,
                        resolved_module or None,
                        {resolved_module: 1} if resolved_module else None,
                    )
                except Exception as e:  # pragma: no cover - graph issues
                    self.logger.error("failed to update knowledge graph: %s", e)

            if self.replicator:
                try:
                    self.replicator.replicate(event)
                except Exception as e:  # pragma: no cover - network issues
                    self.logger.error("failed to replicate telemetry: %s", e)
            if self.sentry:
                try:
                    self.sentry.capture_exception(
                        Exception(json.dumps(payload, sort_keys=True))
                    )
                except Exception as e:
                    self.logger.warning(
                        "failed to send fix suggestion to Sentry: %s", e
                    )

            ticket_file = os.getenv("FIX_TICKET_FILE")
            if ticket_file:
                try:
                    with open(resolve_path(ticket_file), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(payload, sort_keys=True) + "\n")
                except Exception as e:  # pragma: no cover - I/O issues
                    self.logger.error("failed to open fix ticket: %s", e)

            if generate_patch is not None and resolved_module:
                if self.context_builder is None:
                    self.logger.error(
                        "ContextBuilder not provided; cannot generate patch for %s",
                        resolved_module,
                    )
                else:
                    try:
                        self.context_builder.refresh_db_weights()
                    except Exception as e:  # pragma: no cover - validation issues
                        self.logger.error(
                            "ContextBuilder validation failed: %s", e
                        )
                    else:
                        try:
                            patch_id = generate_patch(
                                resolved_module, context_builder=self.context_builder
                            )
                            if patch_id is not None:
                                try:
                                    from sandbox_runner import integrate_new_orphans

                                    integrate_new_orphans(
                                        Path.cwd(), router=GLOBAL_ROUTER
                                    )
                                except Exception as e2:  # pragma: no cover - integration issues
                                    self.logger.error(
                                        "integrate_new_orphans after patch for %s failed: %s",
                                        resolved_module,
                                        e2,
                                    )
                        except Exception as e:  # pragma: no cover - patch failures
                            self.logger.error(
                                "quick fix generation failed for %s: %s",
                                resolved_module,
                                e,
                            )
            else:
                prompt = f"Fix bottleneck in {resolved_module or 'unknown module'}: {hint}"
                if prompt_context:
                    prompt += "\n\n### Training Examples\n" + prompt_context
                if discrepancy_context:
                    prompt += "\n\n### Discrepancy Examples\n" + discrepancy_context
                self.logger.info("Codex prompt: %s", prompt)

            events.append(event)

        return events

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


__all__ = ["TelemetryEvent", "ErrorLogger", "ErrorClassifier", "ErrorType"]
