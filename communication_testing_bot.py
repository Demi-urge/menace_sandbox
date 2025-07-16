"""Communication Testing Bot for verifying communication components and benchmarking Mirror Bot."""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Tuple
import asyncio

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

from .bot_testing_bot import BotTestingBot
from .mirror_bot import MirrorBot
from .comm_testing_config import SETTINGS
from .logging_utils import get_logger


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
        )
    ]
    SCHEMA_VERSION = MIGRATIONS[-1][0]

    def __init__(self, path: Path | str | None = None) -> None:
        db_path = Path(path or _default_db_path())
        # allow connection reuse across threads for async tests
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cur = self.conn.cursor()
        version = int(cur.execute("PRAGMA user_version").fetchone()[0])
        for target, stmts in self.MIGRATIONS:
            if version < target:
                for stmt in stmts:
                    cur.execute(stmt)
                version = target
                cur.execute(f"PRAGMA user_version = {version}")
        self.conn.commit()

    def log(self, result: CommTestResult) -> None:
        self.conn.execute(
            "INSERT INTO results(name, passed, details, ts) VALUES (?, ?, ?, ?)",
            (result.name, int(result.passed), result.details, result.timestamp),
        )
        self.conn.commit()

    def fetch(self) -> List[CommTestResult]:
        cur = self.conn.execute(
            "SELECT name, passed, details, ts FROM results ORDER BY id"
        )
        rows = cur.fetchall()
        return [
            CommTestResult(name=r[0], passed=bool(r[1]), details=r[2], timestamp=r[3])
            for r in rows
        ]

    def fetch_failed(self) -> List[CommTestResult]:
        cur = self.conn.execute(
            "SELECT name, passed, details, ts FROM results WHERE passed=0 ORDER BY id"
        )
        rows = cur.fetchall()
        return [
            CommTestResult(name=r[0], passed=False, details=r[2], timestamp=r[3])
            for r in rows
        ]

    def fetch_by_name(self, name: str) -> List[CommTestResult]:
        cur = self.conn.execute(
            "SELECT name, passed, details, ts FROM results WHERE name=? ORDER BY id",
            (name,),
        )
        rows = cur.fetchall()
        return [
            CommTestResult(name=r[0], passed=bool(r[1]), details=r[2], timestamp=r[3])
            for r in rows
        ]


class CommunicationTestingBot:
    """Bot that runs communication tests and mirror benchmarks."""

    def __init__(self, db: CommTestDB | None = None) -> None:
        self.db = db or CommTestDB()
        self.tester = BotTestingBot()
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
