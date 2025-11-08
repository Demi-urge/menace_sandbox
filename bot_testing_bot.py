"""Bot Testing Bot for automated test execution and logging."""

from __future__ import annotations

from .coding_bot_interface import self_coding_managed
import importlib
import inspect
import doctest
import json
import logging
import os
import random
import threading
import uuid
import traceback
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Union, get_args, get_origin
import hashlib
import time

from .bot_testing_config import BotTestingSettings
from .db_router import DBRouter, GLOBAL_ROUTER, init_db_router
from .data_bot import DataBot, persist_sc_thresholds
from context_builder_util import create_context_builder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .bot_registry import BotRegistry
    from .self_coding_manager import SelfCodingManager
    from .self_coding_engine import SelfCodingEngine
    from .model_automation_pipeline import ModelAutomationPipeline
    from .shared_evolution_orchestrator import EvolutionOrchestrator
    from vector_service.context_builder import ContextBuilder

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .evolution_orchestrator import EvolutionOrchestrator

logger = logging.getLogger("BotTester")

_MANAGER_LOCK = threading.RLock()


def _import_relative(name: str):
    """Import ``name`` relative to this package without eager evaluation."""

    base = __package__ or __name__.rpartition(".")[0]
    module_name = f"{base}.{name}" if base else name
    return importlib.import_module(module_name)


def _get_module_attr(module: Any, attr: str, module_name: str) -> Any:
    """Return ``attr`` from ``module`` re-importing fallbacks if required."""

    target = getattr(module, attr, None)
    if target is not None:
        return target

    for prefix in ("menace_sandbox", "menace"):
        try:
            candidate = importlib.import_module(f"{prefix}.{module_name}")
        except Exception:  # pragma: no cover - fallback best effort
            continue
        fallback = getattr(candidate, attr, None)
        if fallback is not None:
            return fallback

    raise AttributeError(f"module {module_name!r} has no attribute {attr!r}")


@lru_cache(maxsize=1)
def _get_registry() -> "BotRegistry":
    from .bot_registry import BotRegistry

    return BotRegistry()


@lru_cache(maxsize=1)
def _get_data_bot() -> DataBot:
    return DataBot(start_server=False)


@lru_cache(maxsize=1)
def _get_context_builder() -> "ContextBuilder":
    return create_context_builder()


@lru_cache(maxsize=1)
def _get_engine() -> "SelfCodingEngine":
    code_db_module = _import_relative("code_database")
    memory_module = _import_relative("gpt_memory")
    engine_module = _import_relative("self_coding_engine")

    engine_cls = _get_module_attr(engine_module, "SelfCodingEngine", "self_coding_engine")
    code_db_cls = _get_module_attr(code_db_module, "CodeDB", "code_database")
    memory_cls = _get_module_attr(memory_module, "GPTMemoryManager", "gpt_memory")

    return engine_cls(  # type: ignore[call-arg]
        code_db_cls(),
        memory_cls(),
        context_builder=_get_context_builder(),
    )


@lru_cache(maxsize=1)
def _get_pipeline() -> "ModelAutomationPipeline":
    pipeline_module = _import_relative("model_automation_pipeline")

    pipeline_cls = _get_module_attr(
        pipeline_module,
        "ModelAutomationPipeline",
        "model_automation_pipeline",
    )

    return pipeline_cls(context_builder=_get_context_builder())  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def _get_evolution_orchestrator() -> "EvolutionOrchestrator":
    orchestrator_module = _import_relative("shared_evolution_orchestrator")

    get_orchestrator = _get_module_attr(
        orchestrator_module,
        "get_orchestrator",
        "shared_evolution_orchestrator",
    )

    return get_orchestrator(  # type: ignore[call-arg]
        "BotTestingBot",
        _get_data_bot(),
        _get_engine(),
    )


def _ensure_real_manager() -> "SelfCodingManager":
    with _MANAGER_LOCK:
        if getattr(_ensure_real_manager, "_instance", None) is not None:
            return _ensure_real_manager._instance  # type: ignore[attr-defined]

        manager_module = _import_relative("self_coding_manager")
        thresholds_module = _import_relative("self_coding_thresholds")
        threshold_service_module = _import_relative("threshold_service")

        get_thresholds = _get_module_attr(
            thresholds_module,
            "get_thresholds",
            "self_coding_thresholds",
        )
        thresholds = get_thresholds("BotTestingBot")
        persist_sc_thresholds(
            "BotTestingBot",
            roi_drop=thresholds.roi_drop,
            error_increase=thresholds.error_increase,
            test_failure_increase=thresholds.test_failure_increase,
        )
        internalize_coding_bot = _get_module_attr(
            manager_module,
            "internalize_coding_bot",
            "self_coding_manager",
        )
        threshold_service_cls = _get_module_attr(
            threshold_service_module,
            "ThresholdService",
            "threshold_service",
        )
        manager = internalize_coding_bot(  # type: ignore[call-arg]
            "BotTestingBot",
            _get_engine(),
            _get_pipeline(),
            data_bot=_get_data_bot(),
            bot_registry=_get_registry(),
            evolution_orchestrator=_get_evolution_orchestrator(),
            roi_threshold=thresholds.roi_drop,
            error_threshold=thresholds.error_increase,
            test_failure_threshold=thresholds.test_failure_increase,
            threshold_service=threshold_service_cls(),
        )
        _ensure_real_manager._instance = manager  # type: ignore[attr-defined]
        return manager


_get_registry.__self_coding_lazy__ = True  # type: ignore[attr-defined]
_get_data_bot.__self_coding_lazy__ = True  # type: ignore[attr-defined]


class _DeferredManager:
    """Lightweight proxy deferring heavy manager bootstrap until needed."""

    def __init__(self, factory: Callable[[], "SelfCodingManager"]) -> None:
        self._factory = factory
        self._manager: "SelfCodingManager | None" = None
        self._pending_register: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self._lock = threading.RLock()
        self.bot_registry = _get_registry()
        self.data_bot = _get_data_bot()

    def _ensure(self) -> "SelfCodingManager":
        manager = self._manager
        if manager is not None:
            return manager

        pending: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        with self._lock:
            manager = self._manager
            if manager is None:
                manager = self._factory()
                self._manager = manager
                pending = self._pending_register[:]
                self._pending_register.clear()

        if pending:
            for args, kwargs in pending:
                try:
                    manager.register(*args, **kwargs)
                except Exception:  # pragma: no cover - registration best effort
                    manager.logger.exception(
                        "deferred manager registration failed for %s", args[0] if args else ""
                    )

        return manager

    def register(self, *args: Any, **kwargs: Any) -> Any:
        if self._manager is None:
            with self._lock:
                if self._manager is None:
                    self._pending_register.append((args, kwargs))
                    return None
        return self._ensure().register(*args, **kwargs)

    def ensure(self) -> "SelfCodingManager":
        return self._ensure()

    def __getattr__(self, item: str) -> Any:
        return getattr(self._ensure(), item)


_MANAGER_PROXY = _DeferredManager(_ensure_real_manager)

# Allow users to register custom randomizers for specific types
CUSTOM_GENERATORS: dict[type[Any], Callable[[], Any]] = {}


def register_generator(tp: type[Any], generator: Callable[[], Any]) -> None:
    """Register a custom random value generator for a specific type."""
    CUSTOM_GENERATORS[tp] = generator


try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Faker = None  # type: ignore
    logger.warning("Faker not installed; randomized testing disabled")

try:
    from hypothesis import strategies as st  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    st = None  # type: ignore


def test_category(category: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark a function with a test category."""

    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        setattr(func, "_test_category", category)
        return func

    return wrapper


@dataclass
class TestResult:
    """Result of a single test case."""

    id: str
    bot: str
    version: str
    passed: bool
    error: str | None
    timestamp: str
    code_hash: str | None = None


@dataclass
class IntegrationTask:
    """Schema for an integration test step."""

    module: str
    function: str | None = None
    pre_conditions: dict[str, Any] | None = None
    post_conditions: dict[str, Any] | None = None


class TestingLogDB:
    """DB for storing test results. Supports SQLite and PostgreSQL."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: Path | str | None = None,
        connection_factory: Callable[[], Any] | None = None,
        *,
        backend: str = "sqlite",
        settings: BotTestingSettings | None = None,
        router: DBRouter | None = None,
    ) -> None:
        self.settings = settings or BotTestingSettings()
        self.lock = threading.Lock()
        backend = os.environ.get("BOT_TESTING_DB_BACKEND", self.settings.db_backend)
        if backend == "sqlite":
            if connection_factory is not None:
                self.conn = connection_factory()
            else:
                router = router or GLOBAL_ROUTER or init_db_router("bot_testing_bot")
                self.conn = router.get_connection("results")
        elif backend == "postgres":  # pragma: no cover - optional backend
            import psycopg2  # type: ignore

            dsn = path or os.environ.get("BOT_TESTING_DB_DSN", self.settings.db_dsn)
            if connection_factory is not None:
                self.conn = connection_factory()
            else:
                self.conn = psycopg2.connect(dsn)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        schema_path = os.environ.get("BOT_TESTING_DB_SCHEMA")
        if schema_path:
            sql = Path(schema_path).read_text()
            self.conn.executescript(sql)
        else:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS results(
                    id TEXT,
                    bot TEXT,
                    version TEXT,
                    passed INTEGER,
                    error TEXT,
                    ts TEXT,
                    code_hash TEXT
                )
                """
            )
        try:
            cur = self.conn.execute("PRAGMA table_info(results)")
            cols = [r[1] for r in cur.fetchall()]
            if "code_hash" not in cols:
                self.conn.execute("ALTER TABLE results ADD COLUMN code_hash TEXT")
        except Exception as exc:
            logger.exception("failed to adjust test results schema: %s", exc)
        self.conn.commit()

    def log(self, result: TestResult) -> None:
        delay = self.settings.db_write_delay
        for attempt in range(self.settings.db_write_attempts):
            try:
                with self.lock:
                    self.conn.execute(
                        "INSERT INTO results VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            result.id,
                            result.bot,
                            result.version,
                            int(result.passed),
                            result.error or "",
                            result.timestamp,
                            result.code_hash or "",
                        ),
                    )
                    self.conn.commit()
                break
            except Exception:  # pragma: no cover - best effort
                if attempt == self.settings.db_write_attempts - 1:
                    raise
                time.sleep(delay)
                delay *= 2

    def all(self) -> List[TestResult]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, bot, version, passed, error, ts, code_hash FROM results"
            )
            rows = cur.fetchall()
        return [
            TestResult(
                id=r[0],
                bot=r[1],
                version=r[2],
                passed=bool(r[3]),
                error=r[4] or None,
                timestamp=r[5],
                code_hash=r[6] or None,
            )
            for r in rows
        ]


@self_coding_managed(
    bot_registry=_get_registry,
    data_bot=_get_data_bot,
    manager=_MANAGER_PROXY,
)
class BotTestingBot:
    """Run unit and basic integration tests for bots."""

    def __init__(
        self,
        db: TestingLogDB | None = None,
        *,
        settings: BotTestingSettings | None = None,
        manager: SelfCodingManager | None = None,
    ) -> None:
        self.settings = settings or BotTestingSettings()
        self.db = db or TestingLogDB(settings=self.settings)
        self.version = self.settings.version
        self.random_runs = self.settings.random_runs
        if os.environ.get("BOT_TESTING_LOG_JSON"):

            class JsonFormatter(logging.Formatter):
                def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
                    data = {
                        "time": self.formatTime(record),
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage(),
                    }
                    return json.dumps(data)

            handler = logging.StreamHandler()
            handler.setFormatter(JsonFormatter())
            self.logger = logging.getLogger("BotTester")
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        else:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger("BotTester")
        if Faker:
            self.fake = Faker()
        else:
            self.fake = None
            if not self.settings.allow_unrandomized:
                raise ImportError("Faker not installed and allow_unrandomized is False")
            self.logger.warning("Faker not installed; randomized testing disabled")
        self.name = getattr(self, "name", self.__class__.__name__)
        self.data_bot = DataBot()
        _MANAGER_PROXY.ensure()

    def _random_arg(self, param: inspect.Parameter) -> Any:
        ann = param.annotation
        fake = self.fake
        gen_int = fake.pyint if fake else lambda: random.randint(0, 100)
        gen_float = fake.pyfloat if fake else random.random
        gen_str = fake.pystr if fake else lambda: "x"
        gen_bool = (
            (lambda: bool(fake.pyint() % 2))
            if fake
            else lambda: bool(random.randint(0, 1))
        )
        if ann in CUSTOM_GENERATORS:
            try:
                return CUSTOM_GENERATORS[ann]()
            except Exception as exc:
                logger.exception("custom generator failed for %s: %s", ann, exc)
        if ann is inspect._empty:
            return gen_str()
        if not fake and param.default is not inspect._empty:
            # Use provided default when no faker is available
            default_val = param.default
        else:
            default_val = None
        if st is not None:
            try:
                return st.from_type(ann).example()
            except Exception as exc:
                logger.exception("hypothesis example failed for %s: %s", ann, exc)
        origin = get_origin(ann)
        args = get_args(ann)
        if origin is list and args:
            return [
                self._random_arg(
                    inspect.Parameter(
                        "x", inspect.Parameter.POSITIONAL_ONLY, annotation=args[0]
                    )
                )
            ]
        if origin is dict and len(args) == 2:
            key = self._random_arg(
                inspect.Parameter(
                    "k", inspect.Parameter.POSITIONAL_ONLY, annotation=args[0]
                )
            )
            val = self._random_arg(
                inspect.Parameter(
                    "v", inspect.Parameter.POSITIONAL_ONLY, annotation=args[1]
                )
            )
            return {key: val}
        if origin is tuple and args:
            return tuple(
                self._random_arg(
                    inspect.Parameter(
                        "x", inspect.Parameter.POSITIONAL_ONLY, annotation=args[0]
                    )
                )
                for _ in range(1)
            )
        if origin is not None:
            if origin is Union:
                non_none = [a for a in args if a is not type(None)]
                if not non_none:
                    return None
                return self._random_arg(
                    inspect.Parameter(
                        param.name,
                        inspect.Parameter.POSITIONAL_ONLY,
                        annotation=non_none[0],
                    )
                )
            return self._random_arg(
                inspect.Parameter(
                    param.name, inspect.Parameter.POSITIONAL_ONLY, annotation=origin
                )
            )
        if is_dataclass(ann):
            values = {}
            for f in fields(ann):
                fake_param = inspect.Parameter(
                    f.name, inspect.Parameter.POSITIONAL_ONLY, annotation=f.type
                )
                values[f.name] = self._random_arg(fake_param)
            return ann(**values)
        if ann is int:
            return gen_int()
        if ann is float:
            return gen_float()
        if ann is bool:
            return gen_bool()
        return default_val if default_val is not None else gen_str()

    def _execute_function(
        self, func, *, runs: Optional[int] = None
    ) -> tuple[bool, str | None]:
        try:
            if func.__doc__:
                finder = doctest.DocTestFinder()
                runner = doctest.DocTestRunner(verbose=False)
                for test in finder.find(func, globs={}):
                    runner.run(test)
                if runner.failures:
                    raise AssertionError(f"{runner.failures} doctest failure(s)")

            module = importlib.import_module(func.__module__)
            test_func = getattr(module, f"test_{func.__name__}", None)
            if callable(test_func):
                test_func()

            params = inspect.signature(func).parameters
            run_count = runs or self.random_runs
            for _ in range(run_count):
                args_list = []
                kwargs: dict[str, Any] = {}
                for p in params.values():
                    if p.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    value = self._random_arg(p)
                    if p.kind in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    ):
                        args_list.append(value)
                    else:
                        kwargs[p.name] = value
                result = func(*args_list, **kwargs)
                if (
                    inspect.signature(func).return_annotation
                    is not inspect.Signature.empty
                    and result is None
                ):
                    raise AssertionError("Function returned None")
            return True, None
        except Exception:  # pragma: no cover - runtime failures
            return False, traceback.format_exc()

    def _run_function(
        self, func, *, runs: Optional[int] = None
    ) -> tuple[bool, str | None]:
        if self.settings.test_timeout:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._execute_function, func, runs=runs)
                try:
                    return future.result(timeout=self.settings.test_timeout)
                except Exception as exc:  # pragma: no cover - runtime failures
                    future.cancel()
                    if isinstance(exc, TimeoutError):
                        return False, "Timeout"
                    return False, traceback.format_exc()
        else:
            return self._execute_function(func, runs=runs)

    def run_unit_tests(
        self,
        modules: List[str],
        *,
        categories: Optional[List[str]] | None = None,
        name_prefix: str | None = None,
        parallel: bool | None = None,
        random_runs: Optional[int] = None,
    ) -> List[TestResult]:
        start_time = time.time()
        results: List[TestResult] = []
        if parallel is None:
            parallel = self.settings.parallel
        if random_runs is None:
            random_runs = self.random_runs
        tasks: List[Callable[[], TestResult]] = []
        for mod_name in modules:
            try:
                module = importlib.import_module(mod_name)
            except Exception:  # pragma: no cover - import failures
                res = TestResult(
                    id=f"import::{mod_name}",
                    bot=mod_name,
                    version=self.version,
                    passed=False,
                    error=traceback.format_exc(),
                    timestamp=datetime.utcnow().isoformat(),
                )
                self.db.log(res)
                results.append(res)
                continue
            for name in dir(module):
                obj = getattr(module, name)
                funcs: List[Any] = []
                if (
                    inspect.isfunction(obj)
                    and not name.startswith("_")
                    and getattr(obj, "__module__", mod_name) == mod_name
                ):
                    funcs.append(obj)
                elif (
                    inspect.isclass(obj)
                    and not name.startswith("_")
                    and getattr(obj, "__module__", mod_name) == mod_name
                ):
                    for method_name, method in inspect.getmembers(
                        obj, predicate=inspect.isfunction
                    ):
                        if (
                            method_name.startswith("_")
                            or getattr(method, "__module__", mod_name) != mod_name
                        ):
                            continue
                        funcs.append(method)
                for func in funcs:
                    if name_prefix and not func.__name__.startswith(name_prefix):
                        continue
                    if categories is not None:
                        cat = getattr(func, "_test_category", "unit")
                        if cat not in categories:
                            continue

                    def run_and_log(f=func, module_name=mod_name) -> TestResult:
                        try:
                            src = inspect.getsource(f)
                        except OSError:
                            try:
                                path = inspect.getfile(f)
                                src = Path(path).read_text()
                            except Exception:
                                src = ""
                        code_hash = (
                            hashlib.sha256(src.encode()).hexdigest() if src else None
                        )
                        passed, err = self._run_function(f, runs=random_runs)
                        res = TestResult(
                            id=str(uuid.uuid4()),
                            bot=module_name,
                            version=self.version,
                            passed=passed,
                            error=err,
                            timestamp=datetime.utcnow().isoformat(),
                            code_hash=code_hash,
                        )
                        self.db.log(res)
                        return res

                    if parallel:
                        tasks.append(run_and_log)
                    else:
                        results.append(run_and_log())

        if parallel and tasks:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                for res in executor.map(lambda fn: fn(), tasks):
                    results.append(res)
        tests_run = len(results)
        tests_failed = sum(1 for r in results if not r.passed)
        self.data_bot.collect(
            bot=self.name,
            response_time=time.time() - start_time,
            errors=tests_failed,
            tests_failed=tests_failed,
            tests_run=tests_run,
            revenue=0.0,
            expense=0.0,
        )
        return results

    def run_integration_tests(self, blueprint: str) -> List[TestResult]:
        start_time = time.time()
        try:
            data = json.loads(blueprint)
            tasks_data = data.get("tasks", [])
        except Exception:
            tasks_data = []
        results: List[TestResult] = []
        for t in tasks_data:
            try:
                task = IntegrationTask(**t)
            except Exception:
                continue
            mods = [task.module] if task.module else []
            res = self.run_unit_tests(
                mods, name_prefix=task.function, random_runs=self.random_runs
            )
            results.extend(res)
        tests_run = len(results)
        tests_failed = sum(1 for r in results if not r.passed)
        self.data_bot.collect(
            bot=self.name,
            response_time=time.time() - start_time,
            errors=tests_failed,
            tests_failed=tests_failed,
            tests_run=tests_run,
            revenue=0.0,
            expense=0.0,
        )
        return results


__all__ = [
    "TestResult",
    "IntegrationTask",
    "TestingLogDB",
    "BotTestingBot",
    "register_generator",
]
