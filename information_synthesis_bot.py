"""Information Synthesis Bot for consolidating Stage 2 data.

If Celery is unavailable, tasks are queued in a local in-process queue and
executed by a background worker thread.  This provides basic asynchronous
behaviour without requiring the Celery dependency.
"""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
from dataclasses import asdict
from functools import lru_cache
from typing import TYPE_CHECKING, Callable, Iterable, List
import os
import sys
import logging


def _truthy_env(name: str) -> bool:
    """Return ``True`` when environment variable *name* is truthy."""

    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _noop_self_coding_managed(**_kwargs: object) -> Callable[[type], type]:
    """Fallback decorator used when self-coding must be disabled."""

    def decorator(cls: type) -> type:
        return cls

    return decorator


def _self_coding_disabled() -> bool:
    """Return ``True`` when InformationSynthesisBot should skip self-coding."""

    if _truthy_env("SANDBOX_DISABLE_SELF_CODING"):
        return True
    if _truthy_env("SANDBOX_DISABLE_INFORMATION_SYNTHESIS_SELF_CODING"):
        return True
    if _truthy_env("SANDBOX_ENABLE_INFORMATION_SYNTHESIS_SELF_CODING"):
        return False
    # Windows historically struggles with the heavy dependency footprint.
    if sys.platform.startswith("win") and not _truthy_env(
        "SANDBOX_ENABLE_WINDOWS_SELF_CODING"
    ):
        return True
    # Default to a passive decorator to avoid recursive bootstrap loops when
    # the self-coding runtime is unavailable during sandbox start-up.  Explicit
    # opt-in is required to enable the heavy dependency chain.
    return True


if _self_coding_disabled():
    _self_coding_decorator = _noop_self_coding_managed
    logging.getLogger(__name__).info(
        "InformationSynthesisBot self-coding disabled; using passive decorator"
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
except Exception:  # pragma: no cover - optional
    requests = None  # type: ignore

from .research_aggregator_bot import ResearchAggregatorBot, ResearchItem
try:  # pragma: no cover - optional dependency
    from .task_handoff_bot import WorkflowDB
except Exception:  # pragma: no cover - import may fail in trimmed environments
    WorkflowDB = None  # type: ignore
from .unified_event_bus import UnifiedEventBus
from vector_service.context_builder import ContextBuilder

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore
try:
    from sqlalchemy import create_engine, MetaData, Table, select  # type: ignore
    _SQLALCHEMY_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - optional dependency
    create_engine = MetaData = Table = select = None  # type: ignore
    _SQLALCHEMY_IMPORT_ERROR = exc


def _format_sqlalchemy_import_error(exc: Exception) -> str:
    return (
        "SQLAlchemy helpers are unavailable: "
        f"{type(exc).__name__}: {exc}"
    )


def _current_sqlalchemy_status() -> tuple[bool, str | None]:
    """Return availability of SQLAlchemy helpers and any diagnostic message."""

    if _SQLALCHEMY_IMPORT_ERROR is not None:
        return False, _format_sqlalchemy_import_error(_SQLALCHEMY_IMPORT_ERROR)

    helper_map = {
        "create_engine": create_engine,
        "MetaData": MetaData,
        "Table": Table,
        "select": select,
    }
    missing = []
    for name, helper in helper_map.items():
        if helper is None:
            missing.append(name)
            continue
        if name in {"create_engine", "MetaData", "Table", "select"} and not callable(helper):
            missing.append(name)

    if missing:
        helper_list = ", ".join(sorted(missing))
        return False, f"SQLAlchemy helpers are unavailable: missing helper(s): {helper_list}"

    return True, None


_sqlalchemy_available, _sqlalchemy_unavailable_message = _current_sqlalchemy_status()
try:
    from celery import Celery
except Exception:  # pragma: no cover - optional
    Celery = None  # type: ignore
try:
    from fuzzywuzzy import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None  # type: ignore

from .simple_validation import (
    SimpleField,
    SimpleSchema,
    fields as simple_fields,
    ValidationError as SimpleValidationError,
)


try:
    from marshmallow import (
        Schema as MM_Schema,
        fields as mm_fields,
        ValidationError as MMValidationError,
    )  # type: ignore
    Schema = MM_Schema
    fields = mm_fields
    ValidationError = MMValidationError
except Exception:  # pragma: no cover - optional dependency
    Schema = SimpleSchema
    fields = simple_fields
    ValidationError = SimpleValidationError


from .synthesis_models import DataRequest, SynthesisTask


def send_to_task_manager(task: SynthesisTask) -> None:
    """Send a task description to an external task manager via HTTP."""

    if not requests:
        return
    url = os.getenv("TASK_MANAGER_URL", "http://localhost:8000/task")
    try:
        requests.post(url, json=asdict(task), timeout=5)
    except Exception:  # pragma: no cover - network
        logging.getLogger(__name__).warning("Failed to send task to manager")


@_self_coding_decorator(bot_registry=_get_registry, data_bot=_get_data_bot)
class InformationSynthesisBot:
    """Retrieve, analyse and synthesise data into tasks."""

    def __init__(
        self,
        db_url: str = "sqlite:///synthesis.db",
        broker: str = "memory://",
        aggregator: ResearchAggregatorBot | None = None,
        workflow_db: WorkflowDB | None = None,
        *,
        event_bus: UnifiedEventBus | None = None,
        context_builder: ContextBuilder,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        available, diagnostic = _current_sqlalchemy_status()
        # update module-level flags so other consumers observe the latest status
        global _sqlalchemy_available, _sqlalchemy_unavailable_message
        _sqlalchemy_available = available
        _sqlalchemy_unavailable_message = diagnostic

        self._sqlalchemy_available = available
        self._sqlalchemy_error_message = diagnostic

        if available:
            self.engine = create_engine(db_url)
            self.meta = MetaData()
            self.meta.reflect(bind=self.engine)
        else:  # pragma: no cover - dependency missing
            self.engine = None
            self.meta = None
        try:
            context_builder.refresh_db_weights()
        except Exception:
            pass
        self.context_builder = context_builder
        self.aggregator = aggregator or ResearchAggregatorBot(
            [], context_builder=context_builder, manager=manager
        )
        self.workflow_db = workflow_db or WorkflowDB(event_bus=event_bus)
        if Celery:
            self.app = Celery("synthesis", broker=broker, backend="rpc://")
        else:  # pragma: no cover - celery unavailable
            from .in_memory_queue import InMemoryQueue

            self.app = InMemoryQueue()

    def load_data(self, table: str) -> pd.DataFrame:
        if not self._sqlalchemy_available or not self.engine or not self.meta:
            message = self._sqlalchemy_error_message or (
                "SQLAlchemy helpers are unavailable; cannot load data."
            )
            raise RuntimeError(message)

        tbl = Table(table, self.meta, autoload_with=self.engine)
        with self.engine.connect() as conn:
            df = pd.read_sql(select(tbl), conn)
        return df

    def analyse(self, df: pd.DataFrame, schema: Schema, table: str) -> List[DataRequest]:
        requests: List[DataRequest] = []
        for _, row in df.iterrows():
            try:
                schema.load(row.to_dict())
            except (ValidationError, SimpleValidationError) as e:  # missing or invalid fields
                messages = getattr(e, "messages", {})
                for field in messages:
                    requests.append(DataRequest(table=table, field=field, reason="invalid"))
        # detect duplicates via fuzzy name matching
        if fuzz and "name" in df.columns:
            names = df["name"].tolist()
            for i, n1 in enumerate(names):
                for n2 in names[i + 1:]:
                    if fuzz.token_set_ratio(str(n1), str(n2)) > 90:
                        requests.append(
                            DataRequest(table=table, field="name", reason="duplicate")
                        )
                        break
                else:
                    continue
                break
        return requests

    def dispatch_requests(self, requests: Iterable[DataRequest]) -> None:
        for req in requests:
            self.app.send_task("stage2.fetch", kwargs=asdict(req))

    def create_tasks(self, df: pd.DataFrame) -> List[SynthesisTask]:
        tasks: List[SynthesisTask] = []
        for _, row in df.iterrows():
            desc = f"Process {row.get('name', 'item')}"
            tasks.append(
                SynthesisTask(
                    description=desc, urgency=1, complexity=1, category="analysis"
                )
            )
        return tasks

    def send_tasks(self, tasks: Iterable[SynthesisTask]) -> None:
        for t in tasks:
            send_to_task_manager(t)

    def flagged_workflows(self, items: Iterable[ResearchItem]) -> List[str]:
        names = []
        for it in items:
            if it.category.lower() == "workflow" or "workflow" in (
                t.lower() for t in it.tags
            ):
                names.append(it.title or it.topic)
        return names

    def fetch_reusable(self, items: Iterable[ResearchItem]) -> List[str]:
        """Return names of workflows that could be reused."""
        names = self.flagged_workflows(items)
        if not names:
            return []
        try:
            records = self.workflow_db.fetch()
        except Exception:
            return []

        matches: List[tuple[float, str]] = []
        for rec in records:
            text = f"{rec.title} {rec.description} {' '.join(rec.tags)}".lower()
            best = 0.0
            for name in names:
                low = name.lower()
                if low in text:
                    best = 1.0
                    break
                try:
                    from difflib import SequenceMatcher

                    score = SequenceMatcher(None, low, text).ratio()
                except Exception:
                    score = 0.0
                if score > best:
                    best = score
            if best >= 0.75:
                matches.append((best, rec.title or rec.description or ""))

        matches.sort(reverse=True, key=lambda x: x[0])
        return [m[1] for m in matches]

    def synthesise(
        self,
        table: str,
        schema: Schema,
        topic: str | None = None,
        energy: int = 1,
    ) -> List[SynthesisTask]:
        df = self.load_data(table)
        reqs = self.analyse(df, schema, table)
        if reqs:
            self.dispatch_requests(reqs)
        tasks = self.create_tasks(df)
        if not tasks:
            topic = topic or table
            try:
                items = self.aggregator.process(topic, energy=energy)
            except Exception:
                items = []
            data = [
                {
                    "id": it.item_id or 0,
                    "name": it.title or it.topic,
                    "content": it.content,
                }
                for it in items
            ]
            if pd and data:
                df = pd.DataFrame(data)
                tasks = self.create_tasks(df)
            # do not reuse entire workflows
        self.send_tasks(tasks)
        return tasks


__all__ = [
    "DataRequest",
    "SynthesisTask",
    "InformationSynthesisBot",
    "send_to_task_manager",
    "Schema",
    "fields",
    "SimpleSchema",
    "SimpleField",
    "ValidationError",
]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]