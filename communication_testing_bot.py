"""Communication Testing Bot for verifying communication components and benchmarking Mirror Bot."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
from difflib import SequenceMatcher
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Tuple, Literal, TYPE_CHECKING, cast
import asyncio
import tempfile

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .mirror_bot import MirrorBot
from .comm_testing_config import SETTINGS
from .logging_utils import get_logger
from .db_router import DBRouter, GLOBAL_ROUTER, LOCAL_TABLES, init_db_router
from .scope_utils import Scope, build_scope_clause, apply_scope


if TYPE_CHECKING:  # pragma: no cover - typing only import
    from .bot_testing_bot import BotTestingBot


def _default_db_path() -> Path:
    return SETTINGS.resolved_db_path


# ---------------------------------------------------------------------------
# Module registry for validation during functional tests
REGISTERED_MODULES: set[str] = set()


def register_module(name: str) -> None:
    """Add *name* to the set of valid modules for testing."""
    REGISTERED_MODULES.add(name)


def unregister_module(name: str) -> None:
    """Remove *name* from the registry of valid modules."""
    REGISTERED_MODULES.discard(name)


def is_valid_module(name: str) -> bool:
    """Return ``True`` if *name* has been registered for testing."""
    return name in REGISTERED_MODULES


@dataclass
class CommTestResult:
    """Result of a communication test."""

    name: str
    passed: bool
    details: str
    timestamp: str = datetime.now(timezone.utc).isoformat()
    source_menace_id: str = ""


class CommTestDB:
    """SQLite-backed storage for communication test results."""

    MIGRATIONS: list[tuple[int, list[str]]] = [
        (
            1,
            [
                """
                CREATE TABLE IF NOT EXISTS results(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    passed INTEGER,
                    details TEXT,
                    ts TEXT
                )
                """,
                "CREATE INDEX IF NOT EXISTS idx_results_name ON results(name)",
            ],
        ),
        (
            2,
            [
                "ALTER TABLE results ADD COLUMN source_menace_id TEXT NOT NULL DEFAULT ''",
                "CREATE INDEX IF NOT EXISTS idx_results_source_menace_id "
                "ON results(source_menace_id)",
            ],
        ),
    ]
    SCHEMA_VERSION = MIGRATIONS[-1][0]

    def __init__(
        self, path: Path | str | None = None, *, router: DBRouter | None = None
    ) -> None:
        p = Path(path or _default_db_path())
        if str(p) == ":memory:" and router is None:
            tmp = tempfile.NamedTemporaryFile(delete=False)
            tmp_path = tmp.name
            tmp.close()
            self.router = init_db_router("comm_test", tmp_path, tmp_path)
        else:
            self.router = router or GLOBAL_ROUTER or init_db_router(
                "comm_test", str(p), str(p)
            )
        LOCAL_TABLES.add("results")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = self.router.get_connection("results")
        cur = conn.cursor()
        version = int(cur.execute("PRAGMA user_version").fetchone()[0])
        for target, stmts in self.MIGRATIONS:
            if version < target:
                for stmt in stmts:
                    cur.execute(stmt)
                version = target
                cur.execute(f"PRAGMA user_version = {version}")
        conn.commit()

    def log(self, result: CommTestResult, *, source_menace_id: str | None = None) -> None:
        conn = self.router.get_connection("results")
        menace_id = source_menace_id or result.source_menace_id or self.router.menace_id
        conn.execute(
            "INSERT INTO results(name, passed, details, ts, source_menace_id) "
            "VALUES (?, ?, ?, ?, ?)",
            (result.name, int(result.passed), result.details, result.timestamp, menace_id),
        )
        conn.commit()
        result.source_menace_id = menace_id

    def fetch(
        self,
        *,
        scope: Literal["local", "global", "all"] = "local",
        source_menace_id: str | None = None,
    ) -> List[CommTestResult]:
        conn = self.router.get_connection("results")
        menace_id = source_menace_id or self.router.menace_id
        clause, params = build_scope_clause("results", Scope(scope), menace_id)
        query = apply_scope(
            "SELECT name, passed, details, ts, source_menace_id FROM results", clause
        )
        query += " ORDER BY id"
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            CommTestResult(
                name=r[0],
                passed=bool(r[1]),
                details=r[2],
                timestamp=r[3],
                source_menace_id=r[4],
            )
            for r in rows
        ]

    def fetch_failed(
        self,
        *,
        scope: Scope | str = Scope.LOCAL,
        source_menace_id: str | None = None,
    ) -> List[CommTestResult]:
        conn = self.router.get_connection("results")
        menace_id = source_menace_id or self.router.menace_id
        clause, params = build_scope_clause("results", scope, menace_id)
        query = "SELECT name, passed, details, ts, source_menace_id FROM results WHERE passed=0"
        if clause:
            query += f" AND {clause}"
        query += " ORDER BY id"
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            CommTestResult(
                name=r[0],
                passed=False,
                details=r[2],
                timestamp=r[3],
                source_menace_id=r[4],
            )
            for r in rows
        ]

    def fetch_by_name(
        self,
        name: str,
        *,
        scope: Scope | str = Scope.LOCAL,
        source_menace_id: str | None = None,
    ) -> List[CommTestResult]:
        conn = self.router.get_connection("results")
        menace_id = source_menace_id or self.router.menace_id
        clause, scope_params = build_scope_clause("results", scope, menace_id)
        query = "SELECT name, passed, details, ts, source_menace_id FROM results WHERE name=?"
        params: list[object] = [name]
        if clause:
            query += f" AND {clause}"
            params.extend(scope_params)
        query += " ORDER BY id"
        cur = conn.execute(query, params)
        rows = cur.fetchall()
        return [
            CommTestResult(
                name=r[0],
                passed=bool(r[1]),
                details=r[2],
                timestamp=r[3],
                source_menace_id=r[4],
            )
            for r in rows
        ]


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class CommunicationTestingBot:
    """Bot that runs communication tests and mirror benchmarks."""

    def __init__(
        self,
        db: CommTestDB | None = None,
        tester: "BotTestingBot | None" = None,
    ) -> None:
        self.db = db or CommTestDB()
        if tester is None:
            from .bot_testing_bot import BotTestingBot as _BotTestingBot

            tester = _BotTestingBot()
        assert tester is not None
        self.tester = cast("BotTestingBot", tester)
        self.logger = get_logger("CommTester")

    def functional_tests(self, modules: Iterable[str]) -> List[CommTestResult]:
        for mod in modules:
            if not is_valid_module(mod):
                raise ValueError(f"unregistered module: {mod}")

        results = []
        unit = self.tester.run_unit_tests(list(modules))
        for res in unit:
            c = CommTestResult(name=res.id, passed=res.passed, details=res.error or "")
            self.db.log(c)
            results.append(c)
        return results

    def integration_test(
        self,
        send: Callable[[str], None],
        receive: Callable[[], str],
        message: str,
        *,
        expected: str | Callable[[str], bool] | None = None,
        retries: int = 0,
        delay: float = 0.1,
        max_delay: float | None = None,
    ) -> CommTestResult:
        attempt = 0
        reply = ""
        failure = ""
        cur_delay = delay
        while True:
            try:
                send(message)
                try:
                    reply = receive()
                    break
                except Exception as exc:  # pragma: no cover - receive failures
                    failure = f"receive_error:{exc}"
            except Exception as exc:  # pragma: no cover - send failures
                failure = f"send_error:{exc}"

            if attempt >= retries:
                reply = failure or ""
                break
            attempt += 1
            time.sleep(cur_delay)
            cur_delay = cur_delay * 2
            if max_delay is not None:
                cur_delay = min(cur_delay, max_delay)

        if callable(expected):
            passed = expected(reply)
            details = f"reply={reply}"
        elif isinstance(expected, str):
            passed = reply == expected
            details = f"expected={expected} actual={reply}"
        else:
            passed = bool(reply)
            details = reply

        result = CommTestResult(name="integration", passed=passed, details=details)
        self.db.log(result)
        return result

    def benchmark_mirror(
        self,
        bot: MirrorBot,
        samples: List[Tuple[str, str]],
        *,
        threshold: float | None = None,
        fuzzy: bool = True,
    ) -> pd.DataFrame:
        if pd is None:
            raise ImportError("pandas is required for benchmark_mirror")

        threshold = threshold if threshold is not None else SETTINGS.benchmark_threshold
        data = []
        for text, expected in samples:
            resp = bot.generate_response(text)
            if expected in resp:
                acc = 1.0
            elif fuzzy:
                acc = SequenceMatcher(None, expected, resp).ratio()
            else:
                acc = 0.0
            data.append({"prompt": text, "expected": expected, "response": resp, "accuracy": acc})
        df = pd.DataFrame(data)
        avg = df["accuracy"].mean() if not df.empty else 0.0
        summary = CommTestResult(
            name="mirror_benchmark", passed=avg >= threshold, details=f"avg_accuracy={avg:.2f}"
        )
        self.db.log(summary)
        return df

    async def functional_tests_async(self, modules: Iterable[str]) -> List[CommTestResult]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.functional_tests, modules)

    async def integration_test_async(
        self,
        send: Callable[[str], None],
        receive: Callable[[], str],
        message: str,
        *,
        expected: str | Callable[[str], bool] | None = None,
        retries: int = 0,
        delay: float = 0.1,
        max_delay: float | None = None,
    ) -> CommTestResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.integration_test,
            send,
            receive,
            message,
            expected,
            retries,
            delay,
            max_delay,
        )

    async def benchmark_mirror_async(
        self,
        bot: MirrorBot,
        samples: List[Tuple[str, str]],
        *,
        threshold: float | None = None,
        fuzzy: bool = True,
    ) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self.benchmark_mirror,
            bot,
            samples,
            threshold,
            fuzzy,
        )

    def report(self, *, output: str = "csv") -> str | list[dict[str, object]]:
        if pd is None:
            raise ImportError("pandas is required for report")

        entries = self.db.fetch()
        df = pd.DataFrame([e.__dict__ for e in entries])
        if df.empty:
            return "No tests run"
        summary = df.groupby("name")["passed"].mean().reset_index()
        if output == "csv":
            return summary.to_csv(index=False)
        if output == "json":
            return summary.to_json(orient="records")
        if output == "dict":
            return summary.to_dict(orient="records")
        raise ValueError(f"unsupported output format: {output}")


__all__ = [
    "CommTestResult",
    "CommTestDB",
    "CommunicationTestingBot",
    "register_module",
    "unregister_module",
    "is_valid_module",
]