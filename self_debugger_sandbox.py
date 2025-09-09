from __future__ import annotations

"""Self-debugging workflow with sandboxed patch testing."""

from .logging_utils import log_record
from .retry_utils import with_retry
import os
import shutil
import subprocess
import asyncio
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
import json
import io
import math
from statistics import pstdev
import sqlite3
import threading
import importlib
from types import SimpleNamespace
from contextlib import contextmanager
from typing import Callable, Mapping
from collections import deque
from coverage import Coverage
from .error_logger import ErrorLogger, TelemetryEvent
from target_region import TargetRegion, extract_target_region
from .knowledge_graph import KnowledgeGraph
from .quick_fix_engine import generate_patch
from .human_alignment_agent import HumanAlignmentAgent
from .human_alignment_flagger import _collect_diff_data
from .violation_logger import log_violation
from .sandbox_runner.scoring import record_run
from db_router import GLOBAL_ROUTER, init_db_router
from .automated_debugger import AutomatedDebugger
from .self_coding_engine import SelfCodingEngine
from .audit_trail import AuditTrail
from patch_attempt_tracker import PatchAttemptTracker
try:
    from .code_database import PatchHistoryDB, _hash_code
except Exception:  # pragma: no cover - test fallback
    from code_database import PatchHistoryDB  # type: ignore

    def _hash_code(data: bytes) -> str:
        return "x"
try:  # pragma: no cover - allow flat imports
    from .dynamic_path_router import resolve_path
except Exception:  # pragma: no cover - fallback for flat layout
    from dynamic_path_router import resolve_path  # type: ignore
from .self_improvement_policy import SelfImprovementPolicy
from .roi_tracker import ROITracker
from .error_cluster_predictor import ErrorClusterPredictor
from .error_parser import ErrorReport, FailureCache, parse_failure
try:
    from self_improvement.baseline_tracker import BaselineTracker
except Exception:  # pragma: no cover - test fallback
    class BaselineTracker:
        def __init__(self, window: int) -> None:
            self.window = int(window)
            self._history = deque(maxlen=self.window)

        def update(self, *, score: float) -> None:
            self._history.append(float(score))

        def get(self, _metric: str) -> float:
            if not self._history:
                return 0.0
            return sum(self._history) / len(self._history)

        def std(self, _metric: str) -> float:
            if len(self._history) < 2:
                return 0.0
            return pstdev(self._history)

try:
    from self_improvement.metrics import (
        compute_entropy_metrics,
        compute_entropy_delta,
    )
except Exception:  # pragma: no cover - test fallback
    def compute_entropy_metrics(files):
        return 0.0, 0.0, 0.0

    def compute_entropy_delta(code_diversity, token_complexity):
        return 0.0, 0.0
try:
    from sandbox_runner.environment import create_ephemeral_env, generate_edge_cases
except Exception:  # pragma: no cover - test fallback
    def create_ephemeral_env(*a, **k):
        raise RuntimeError("sandbox_runner unavailable")

    def generate_edge_cases() -> dict[str, object]:  # type: ignore
        return {}
try:
    from .sandbox_settings import SandboxSettings
except Exception:  # pragma: no cover - fallback for flat layout
    from sandbox_settings import SandboxSettings  # type: ignore
try:
    from .sandbox_runner import post_round_orphan_scan
except Exception:  # pragma: no cover - fallback for flat layout
    from sandbox_runner import post_round_orphan_scan  # type: ignore
try:
    from vector_service.context_builder import ContextBuilder
    from vector_service.context_builder import record_failed_tags
except Exception:  # pragma: no cover - optional dependency
    ContextBuilder = None  # type: ignore

    def record_failed_tags(_tags):  # type: ignore
        return None


class CoverageSubprocessError(RuntimeError):
    """Raised when a test subprocess exits with a non-zero code."""

    def __init__(self, output: str) -> None:
        super().__init__(output)
        self.output = output


class RollbackError(RuntimeError):
    """Raised when a rollback operation fails during sandbox evaluation."""


class CandidateEvaluationError(RuntimeError):
    """Raised when evaluating a candidate patch fails but is recoverable."""


router = GLOBAL_ROUTER or init_db_router("self_debugger_sandbox")


class SelfDebuggerSandbox(AutomatedDebugger):
    """Extend AutomatedDebugger with sandbox verification.

    Parameters
    ----------
    context_builder:
        Preconfigured :class:`~vector_service.context_builder.ContextBuilder` used
        for context retrieval and patch scoring.
    flakiness_runs:
        Number of test executions used when estimating flakiness.
    weight_update_interval:
        Minimum seconds between score weight recalculations. Can also be set
        via the ``WEIGHT_UPDATE_INTERVAL`` environment variable.
    baseline_window:
        Number of recent composite scores used when computing the dynamic
        baseline.
    stagnation_iters:
        Iterations without improvement before the baseline resets to the
        current average.
    delta_margin:
        Minimum positive delta over the moving baseline required for patch
        acceptance.
    """

    def __init__(
        self,
        telemetry_db: object,
        engine: SelfCodingEngine,
        *,
        context_builder: ContextBuilder,
        audit_trail: AuditTrail | None = None,
        policy: SelfImprovementPolicy | None = None,
        state_getter: Callable[[], tuple[int, ...]] | None = None,
        error_predictor: ErrorClusterPredictor | None = None,
        score_weights: tuple[float, float, float, float, float, float] | None = None,
        flakiness_runs: int | None = None,
        smoothing_factor: float = 0.5,
        weight_update_interval: float | None = None,
        baseline_window: int | None = None,
        stagnation_iters: int | None = None,
        delta_margin: float | None = None,
        merge_threshold: float | None = None,
        settings: SandboxSettings | None = None,
    ) -> None:
        if context_builder is None:
            raise ValueError("SelfDebuggerSandbox requires a ContextBuilder instance")
        super().__init__(telemetry_db, engine, context_builder)
        self.audit_trail = audit_trail or getattr(engine, "audit_trail", None)
        self.policy = policy
        self.state_getter = state_getter
        self.error_predictor = error_predictor
        self._bad_hashes: set[str] = set()
        self._settings = settings or SandboxSettings()
        if baseline_window is None:
            baseline_window = getattr(self._settings, "baseline_window", 5)
        if stagnation_iters is None:
            stagnation_iters = getattr(self._settings, "stagnation_iters", 10)
        if delta_margin is None:
            delta_margin = getattr(self._settings, "delta_margin", 0.0)
        if score_weights is None:
            score_weights = tuple(
                getattr(
                    self._settings,
                    "score_weights",
                    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                )
            )
        if merge_threshold is None:
            merge_threshold = delta_margin
        self.delta_margin = float(delta_margin)
        self.merge_threshold = float(merge_threshold)
        self.score_weights = tuple(score_weights)
        self.stagnation_iters = int(stagnation_iters)
        self._baseline_tracker = BaselineTracker(int(baseline_window))
        if flakiness_runs is None:
            flakiness_runs = getattr(self._settings, "flakiness_runs", 1)
        self.flakiness_runs = max(1, int(flakiness_runs))
        self.smoothing_factor = max(0.0, min(1.0, float(smoothing_factor))) or 0.5
        if weight_update_interval is None:
            weight_update_interval = getattr(
                self._settings, "weight_update_interval", 0.0
            )
        self._last_weights_update = 0.0
        self._weight_update_interval = max(0.0, float(weight_update_interval))
        self._score_db: PatchHistoryDB | None = None
        self._db_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._history_conn: sqlite3.Connection | None = None
        self._history_records: list[tuple[float, float, float, float, float, float]] = []
        self._recent_outcomes: deque[int] = deque(
            maxlen=getattr(self._settings, "success_window", 20)
        )
        self._total_attempts = 0
        self._total_successes = 0
        self.recent_success_rate = 0.0
        self.long_term_success_rate = 0.0
        self.momentum = 0.0
        try:
            self._history_conn = router.get_connection("flakiness_history")
            with self._history_lock:
                self._history_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS flakiness_history(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        flakiness REAL,
                        ts TEXT
                    )
                    """
                )
                self._history_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS composite_history(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        coverage_delta REAL,
                        error_delta REAL,
                        roi_delta REAL,
                        complexity REAL,
                        synergy_roi REAL,
                        synergy_efficiency REAL,
                        synergy_resilience REAL,
                        synergy_antifragility REAL,
                        flakiness REAL,
                        score REAL,
                        ts TEXT
                    )
                    """
                )
                self._history_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patch_scores(
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        description TEXT,
                        result TEXT,
                        coverage_delta REAL,
                        error_delta REAL,
                        roi_delta REAL,
                        complexity REAL,
                        synergy_roi REAL,
                        synergy_efficiency REAL,
                        synergy_resilience REAL,
                        synergy_antifragility REAL,
                        flakiness REAL,
                        runtime_impact REAL,
                        score REAL,
                        ts TEXT
                    )
                    """
                )
                self._history_conn.commit()
            self._load_history_stats()
            try:
                cur = self._history_conn.execute(
                    "SELECT score FROM composite_history ORDER BY id DESC LIMIT ?",
                    (self._baseline_tracker.window,),
                )
                for (s,) in reversed(cur.fetchall()):
                    self._baseline_tracker.update(score=float(s))
            except Exception:
                self.logger.exception("baseline history fetch failed")
        except Exception:
            self.logger.exception("score history init failed")
            self._history_conn = None
        self._test_timeout = float(getattr(self._settings, "test_run_timeout", 1))
        self._test_retries = int(getattr(self._settings, "test_run_retries", 0))
        self._score_backend = None
        backend_spec = (
            getattr(self._settings, "patch_score_backend", None)
            or getattr(self._settings, "patch_score_backend_url", None)
        )
        if backend_spec:
            try:
                self._score_backend = self._load_score_backend(backend_spec)
                # fail fast check
                check = getattr(self._score_backend, "ping", None)
                if callable(check):
                    check()
                else:
                    self._score_backend.fetch_recent(1)
            except Exception:
                self.logger.exception("patch score backend init failed")
                raise
        self._metric_stats: dict[str, tuple[float, float]] = {
            "coverage": (0.0, 1.0),
            "error": (0.0, 1.0),
            "roi": (0.0, 1.0),
            "complexity": (0.0, 1.0),
            "synergy_roi": (0.0, 1.0),
            "synergy_efficiency": (0.0, 1.0),
            "synergy_resilience": (0.0, 1.0),
            "synergy_antifragility": (0.0, 1.0),
        }
        self._failure_cache = FailureCache()
        self._last_test_log: Path | None = None
        self.graph = KnowledgeGraph()
        self.error_logger = ErrorLogger(
            knowledge_graph=self.graph, context_builder=self.context_builder
        )
        self._attempt_tracker = PatchAttemptTracker(self.logger)
        self._last_region: TargetRegion | None = None

    # ------------------------------------------------------------------
    def _load_score_backend(self, spec: str):
        """Instantiate a scoring backend from *spec*."""
        from .patch_score_backend import backend_from_url, PatchScoreBackend

        if "://" in spec or spec.startswith("file:"):
            return backend_from_url(spec)
        mod_name, _, cls_name = spec.partition(":")
        if not cls_name:
            mod_name, _, cls_name = spec.rpartition(".")
        module = importlib.import_module(mod_name)
        backend_cls = getattr(module, cls_name)
        backend = backend_cls()
        assert isinstance(backend, PatchScoreBackend)
        return backend

    # ------------------------------------------------------------------
    @contextmanager
    def _history_db(self):
        """Yield the history connection under a thread-safe lock."""
        if not self._history_conn:
            yield None
            return
        with self._history_lock:
            try:
                yield self._history_conn
                self._history_conn.commit()
            except sqlite3.DatabaseError:
                try:
                    self._history_conn.rollback()
                except sqlite3.DatabaseError:
                    self.logger.exception("history rollback failed")
                self.logger.exception("history commit failed")
                raise

    # ------------------------------------------------------------------
    def _record_exception(self, exc: Exception) -> TelemetryEvent:
        """Log ``exc`` and record telemetry in the knowledge graph."""
        return self.error_logger.log(
            exc,
            None,
            getattr(self.engine, "name", None),
        )

    def _record_failed_strategy(self, report: ErrorReport) -> None:
        """Persist canonical tags from ``report`` when available."""
        db = getattr(self.engine, "patch_suggestion_db", None)
        for tag in report.tags:
            try:
                if db:
                    db.add_failed_strategy(tag)
            except Exception:
                self.logger.exception("failed to store failed strategy tag")
        try:
            record_failed_tags(list(report.tags))
        except Exception:
            self.logger.exception("failed to record failed tags")

    # ------------------------------------------------------------------
    def attempt_count(self, region: TargetRegion) -> int:
        """Return number of attempts made for ``region``."""
        return self._attempt_tracker.attempts_for(region)

    def _update_success_metrics(self, result: str) -> None:
        """Update success rate statistics and momentum from ``result``."""
        success = 1 if result == "success" else 0
        self._recent_outcomes.append(success)
        self._total_attempts += 1
        self._total_successes += success
        self.recent_success_rate = sum(self._recent_outcomes) / len(self._recent_outcomes)
        self.long_term_success_rate = (
            self._total_successes / self._total_attempts if self._total_attempts else 0.0
        )
        self.momentum = self.recent_success_rate - self.long_term_success_rate
        if self.policy is not None:
            try:
                self.policy.adjust_for_momentum(self.momentum)
            except Exception:
                self.logger.exception("momentum policy adjustment failed")

    # ------------------------------------------------------------------
    def preemptive_fix_high_risk_modules(self, limit: int = 5) -> None:
        """Apply fixes for modules predicted to be high risk."""
        if not self.error_predictor:
            return
        try:
            modules = self.error_predictor.predict_high_risk_modules(top_n=limit)
        except Exception:
            self.logger.exception("high risk prediction failed")
            return
        for mod in modules:
            try:
                with (
                    tempfile.TemporaryDirectory() as before_dir,
                    tempfile.TemporaryDirectory() as after_dir,
                ):
                    orig = Path(mod)
                    rel = orig.name if orig.is_absolute() else orig
                    src = resolve_path(
                        f"{orig}.py" if orig.suffix == "" else str(orig)
                    )
                    before_target = Path(before_dir) / rel
                    before_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, before_target)
                    patch_id = generate_patch(
                        mod, self.engine, context_builder=self.context_builder
                    )
                    if patch_id is not None:
                        try:
                            post_round_orphan_scan(
                                Path.cwd(), logger=self.logger, router=router
                            )
                        except Exception:
                            self.logger.exception(
                                "post_round_orphan_scan after preemptive patch failed",
                                extra=log_record(module=mod),
                            )
                        after_target = Path(after_dir) / rel
                        after_target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, after_target)
                        diff_data = _collect_diff_data(Path(before_dir), Path(after_dir))
                        workflow_changes = [
                            {"file": f, "code": "\n".join(d["added"])}
                            for f, d in diff_data.items()
                            if d["added"]
                        ]
                        if workflow_changes:
                            agent = HumanAlignmentAgent()
                            warnings = agent.evaluate_changes(workflow_changes, None, [])
                            if any(warnings.values()):
                                log_violation(
                                    str(patch_id),
                                    "alignment_warning",
                                    1,
                                    {"warnings": warnings},
                                    alignment_warning=True,
                                )
                self.logger.info("preemptive patch applied", extra=log_record(module=mod))
            except Exception:
                self.logger.exception("preemptive fix failed", extra=log_record(module=mod))

    # ------------------------------------------------------------------
    async def _coverage_percent(
        self,
        paths: list[Path],
        env: dict[str, str] | None = None,
        *,
        python: str | Path | None = None,
        cwd: Path | None = None,
    ) -> tuple[float, dict[str, dict[str, float]]]:
        """Run tests for *paths* under coverage using parallel workers asynchronously.

        Returns
        -------
        tuple
            A pair of ``(percent, function_coverage)`` where ``function_coverage``
            maps filenames to functions and their line coverage ratio.
        """

        async def run_one(idx: int, set_paths: list[Path]) -> tuple[Path, int, str]:
            data_file = tmp_dir / f".coverage.{idx}"
            p_env = dict(env or os.environ)
            p_env["COVERAGE_FILE"] = str(data_file)
            py_bin = str(python or sys.executable)
            cmd = [
                py_bin,
                "-m",
                "coverage",
                "run",
                "--parallel-mode",
                "-m",
                "pytest",
                "-q",
                "-n",
                "auto",
                "-p",
                "sandbox_runner.edge_case_plugin",
                *[str(p) for p in set_paths],
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                env=p_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd) if cwd else None,
            )
            try:
                out_b, err_b = await asyncio.wait_for(
                    proc.communicate(), timeout=self._test_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                out_b, err_b = await proc.communicate()
                output = (out_b or b"") + (err_b or b"")
                out_text = output.decode("utf-8", "replace")
                self.logger.error(
                    "test run timeout",
                    extra=log_record(cmd=cmd, output=out_text),
                )
                raise CoverageSubprocessError(out_text)
            output = (out_b or b"") + (err_b or b"")
            out_text = output.decode("utf-8", "replace")
            if proc.returncode != 0:
                self.logger.error(
                    "test run failed",
                    extra=log_record(cmd=cmd, rc=proc.returncode, output=out_text),
                )
                raise CoverageSubprocessError(out_text)
            self.logger.debug(
                "test run output", extra=log_record(cmd=cmd, output=out_text)
            )
            return data_file, 0, out_text

        # Support a list of paths treated as individual test sets
        if paths and isinstance(paths[0], (list, tuple)):
            sets = [list(p) for p in paths]  # type: ignore[list-item]
        else:
            sets = [[p] for p in paths]

        tmp_dir = Path(tempfile.mkdtemp())
        start = time.perf_counter()
        try:
            results = await asyncio.gather(
                *(run_one(idx, sp) for idx, sp in enumerate(sets))
            )
            coverage_files = [r[0] for r in results]
            returncodes = [r[1] for r in results]
            outputs = [r[2] for r in results]
            if any(rc != 0 for rc in returncodes):
                msg = "test subprocess failed" + "\n" + "\n".join(outputs)
                raise CoverageSubprocessError(msg)
            cov = Coverage(data_file=str(tmp_dir / ".coverage"))
            cov.combine([str(f) for f in coverage_files])
            buf = io.StringIO()
            xml_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
            xml_tmp.close()
            percent = 0.0
            func_cov: dict[str, dict[str, float]] = {}
            executed_funcs: list[str] = []
            try:
                cov.xml_report(
                    outfile=xml_tmp.name,
                    include=[str(p) for paths in sets for p in paths],
                )
                percent = cov.report(
                    include=[str(p) for paths in sets for p in paths],
                    file=buf,
                )
                try:
                    import xml.etree.ElementTree as ET

                    tree = ET.parse(xml_tmp.name)
                    for cls in tree.findall(".//class"):
                        fname = cls.get("filename")
                        methods: dict[str, float] = {}
                        for meth in cls.findall("methods/method"):
                            name = meth.get("name")
                            lines = meth.findall("lines/line")
                            total = len(lines)
                            covered = sum(
                                1 for ln in lines if int(ln.get("hits", "0")) > 0
                            )
                            if name and total:
                                methods[name] = covered / total
                                if covered > 0 and fname:
                                    executed_funcs.append(f"{fname}:{name}")
                        if fname and methods:
                            func_cov[fname] = methods
                except Exception:
                    func_cov = {}
                    executed_funcs = []
                self.logger.debug(
                    "coverage report", extra=log_record(output=buf.getvalue())
                )
            except Exception as exc:
                self.logger.exception(
                    "coverage generation failed",
                    extra=log_record(err=str(exc), output=buf.getvalue()),
                )
            finally:
                try:
                    os.unlink(xml_tmp.name)
                except Exception:
                    self.logger.exception("coverage cleanup failed")
            runtime = time.perf_counter() - start
            metrics = {
                "success": True,
                "entropy_delta": 0.0,
                "runtime": runtime,
                "error": None,
                "coverage": func_cov,
                "executed_functions": executed_funcs,
            }
            record_run(
                SimpleNamespace(
                    success=metrics.get("success"),
                    duration=metrics.get("runtime"),
                    failure=metrics.get("error"),
                ),
                metrics,
            )
            return float(percent or 0.0), func_cov
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _run_tests(
        self, path: Path, env: dict[str, str] | None = None
    ) -> tuple[float, float] | tuple[float, float, ErrorReport]:
        """Return coverage, runtime and optionally an :class:`ErrorReport`."""
        test_paths = [path]
        tmp: Path | None = None
        self._last_test_log = None
        sandbox_dir = resolve_path(self._settings.sandbox_data_dir)
        sandbox_dir.mkdir(parents=True, exist_ok=True)
        try:
            logs = list(self._recent_logs())
            tests = self._generate_tests(logs)
            if tests:
                tf = tempfile.NamedTemporaryFile(
                    delete=False, suffix=f"_telemetry{os.extsep}py"
                )
                tf.write("\n\n".join(tests).encode("utf-8"))
                tf.close()
                tmp = Path(tf.name)
                test_paths.append(tmp)
        except Exception:
            self.logger.exception("failed to create telemetry tests")

        repo_src = resolve_path(self._settings.sandbox_repo_path or ".")
        failure: ErrorReport | None = None
        cov_details: dict[str, dict[str, float]] = {}

        with create_ephemeral_env(
            repo_src, context_builder=self.context_builder
        ) as (repo, run):
            repo = resolve_path(repo)
            paths_in_repo: list[Path] = []
            for p in test_paths:
                try:
                    rel = p.relative_to(repo_src)
                    target = repo / rel
                except ValueError:
                    target = repo / p.name
                    try:
                        shutil.copy2(p, target)
                    except Exception:
                        self.logger.exception("failed to copy telemetry test")
                paths_in_repo.append(resolve_path(target))

            env_local = dict(env or os.environ)
            env_local["PYTHONPATH"] = str(repo)
            try:
                env_local["SANDBOX_EDGE_CASES"] = json.dumps(generate_edge_cases())
            except Exception:
                env_local["SANDBOX_EDGE_CASES"] = "{}"

            proc = run(
                ["python", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
            )
            python_bin = proc.stdout.strip() or "python"

            start = time.perf_counter()
            try:
                percent, cov_details = with_retry(
                    lambda: asyncio.run(
                        self._coverage_percent(
                            paths_in_repo,
                            env_local,
                            python=python_bin,
                            cwd=repo,
                        )
                    ),
                    attempts=self._test_retries,
                    logger=self.logger,
                    exc=CoverageSubprocessError,
                )
            except CoverageSubprocessError as exc:
                runtime = time.perf_counter() - start
                output = str(exc.output)
                log_file = sandbox_dir / f"fail_{int(time.time()*1000)}.log"
                try:
                    log_file.write_text(output)
                    self._last_test_log = log_file
                except Exception:
                    self.logger.exception("failed to write test log")
                failure = parse_failure(output)
                region = extract_target_region(failure.trace)
                if region:
                    failure.target_region = region
                    level, _ = self._attempt_tracker.level_for(region, region)
                    self._attempt_tracker.record_failure(level, region, region)
                    failure.attempts = self._attempt_tracker.attempts_for(region)
                    self._last_region = region
                if not self._failure_cache.seen(failure):
                    self._failure_cache.add(failure)
                self._record_failed_strategy(failure)
                try:
                    self.error_logger.log(
                        TelemetryEvent(
                            stack_trace=failure.trace,
                            root_cause=",".join(failure.tags),
                        )
                    )
                except Exception:
                    self.logger.exception("failed to log parsed failure")
                self._record_exception(exc)
                percent = 0.0
                entropy_delta = 0.0
                try:
                    code_div, complexity, _ = compute_entropy_metrics([path])
                    entropy_delta, _ = compute_entropy_delta(code_div, complexity)
                except Exception:
                    self.logger.exception("failed to compute entropy metrics")
                metrics = {
                    "success": False,
                    "entropy_delta": entropy_delta,
                    "runtime": runtime,
                    "error": failure.trace if failure else output,
                    "coverage": {},
                    "executed_functions": [],
                }
                record_run(
                    SimpleNamespace(
                        success=metrics.get("success"),
                        duration=metrics.get("runtime"),
                        failure=metrics.get("error"),
                    ),
                    metrics,
                )
            except Exception as exc:
                runtime = time.perf_counter() - start
                output = str(exc)
                failure = parse_failure(output)
                region = extract_target_region(failure.trace)
                if region:
                    failure.target_region = region
                    level, _ = self._attempt_tracker.level_for(region, region)
                    self._attempt_tracker.record_failure(level, region, region)
                    failure.attempts = self._attempt_tracker.attempts_for(region)
                    self._last_region = region
                if not self._failure_cache.seen(failure):
                    self._failure_cache.add(failure)
                self._record_failed_strategy(failure)
                try:
                    self.error_logger.log(
                        TelemetryEvent(
                            stack_trace=failure.trace,
                            root_cause=",".join(failure.tags),
                        )
                    )
                except Exception:
                    self.logger.exception("failed to log parsed failure")
                percent = 0.0
                self._record_exception(exc)
                self.logger.exception("coverage generation failed")
                entropy_delta = 0.0
                try:
                    code_div, complexity, _ = compute_entropy_metrics([path])
                    entropy_delta, _ = compute_entropy_delta(code_div, complexity)
                except Exception:
                    self.logger.exception("failed to compute entropy metrics")
                metrics = {
                    "success": False,
                    "entropy_delta": entropy_delta,
                    "runtime": runtime,
                    "error": failure.trace if failure else output,
                    "coverage": {},
                    "executed_functions": [],
                }
                record_run(
                    SimpleNamespace(
                        success=metrics.get("success"),
                        duration=metrics.get("runtime"),
                        failure=metrics.get("error"),
                    ),
                    metrics,
                )
            else:
                runtime = time.perf_counter() - start
                percent = float(percent or 0.0)

        if tmp:
            tmp.unlink(missing_ok=True)

        result = (float(percent or 0.0), runtime)
        if failure is not None:
            result = result + (failure,)
        return result

    # ------------------------------------------------------------------
    def _test_flakiness(
        self,
        path: Path,
        env: dict[str, str] | None = None,
        *,
        runs: int | None = None,
    ) -> float:
        """Return the standard error of coverage across multiple test runs."""
        n = max(1, int(runs if runs is not None else self.flakiness_runs))

        coverages: list[float] = []

        for i in range(n):
            try:
                res = self._run_tests(path, env)
                cov = res[0]
                self.logger.info(
                    "flakiness run",
                    extra=log_record(
                        run=i,
                        coverage=cov,
                        log=str(self._last_test_log) if self._last_test_log else None,
                    ),
                )
            except Exception as exc:
                self._record_exception(exc)
                self.logger.exception(
                    "flakiness run failed", extra=log_record(run=i)
                )
                cov = 0.0
            coverages.append(float(cov))

        if len(coverages) <= 1:
            flakiness = 0.0
        else:
            stdev = pstdev(coverages)
            flakiness = stdev / math.sqrt(len(coverages))

        if self._score_db:
            try:
                with self._db_lock:
                    self._score_db.record_flakiness(str(path), flakiness)
            except Exception:
                self.logger.exception("flakiness history update failed")

        try:
            with self._history_db() as conn:
                if conn:
                    conn.execute(
                        "INSERT INTO flakiness_history(filename, flakiness, ts) VALUES(?,?,?)",
                        (str(path), float(flakiness), datetime.utcnow().isoformat()),
                    )
        except Exception:
            self.logger.exception("local flakiness history update failed")

        return flakiness

    # ------------------------------------------------------------------
    def _code_complexity(self, path: Path) -> float:
        """Estimate code complexity for *path* using radon when available."""
        try:
            from radon.complexity import cc_visit  # type: ignore
        except Exception:
            return 0.0
        try:
            code = path.read_text()
            blocks = cc_visit(code)
            scores = [b.complexity for b in blocks if hasattr(b, "complexity")]
            return float(sum(scores) / len(scores)) if scores else 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    def _recent_synergy_metrics(
        self, tracker: ROITracker | None
    ) -> tuple[float, float, float, float]:
        """Return the latest synergy metrics from ``tracker``."""
        if not tracker:
            return 0.0, 0.0, 0.0, 0.0
        try:
            roi_hist = tracker.synergy_metrics_history.get("synergy_roi")
            if not roi_hist:
                roi_hist = tracker.metrics_history.get("synergy_roi")
            s_roi = float(roi_hist[-1]) if roi_hist else 0.0
        except Exception:
            s_roi = 0.0
        try:
            eff_hist = tracker.synergy_metrics_history.get("synergy_efficiency")
            if not eff_hist:
                eff_hist = tracker.metrics_history.get("synergy_efficiency")
            s_eff = float(eff_hist[-1]) if eff_hist else 0.0
        except Exception:
            s_eff = 0.0
        try:
            res_hist = tracker.synergy_metrics_history.get("synergy_resilience")
            if not res_hist:
                res_hist = tracker.metrics_history.get("synergy_resilience")
            s_res = float(res_hist[-1]) if res_hist else 0.0
        except Exception:
            s_res = 0.0
        try:
            af_hist = tracker.synergy_metrics_history.get("synergy_antifragility")
            if not af_hist:
                af_hist = tracker.metrics_history.get("synergy_antifragility")
            s_af = float(af_hist[-1]) if af_hist else 0.0
        except Exception:
            s_af = 0.0
        return s_roi, s_eff, s_res, s_af

    # ------------------------------------------------------------------
    def _context_feedback(self, report: ErrorReport) -> list[str]:
        """Fetch new examples excluding previously failed strategies."""

        if ContextBuilder is None:
            return []

        exclude: list[str] = []
        if self._score_db:
            try:
                with self._db_lock:
                    exclude = list(self._score_db.failed_strategy_hashes())
            except Exception:
                self.logger.exception("failed strategy lookup failed")
        failed_tags: list[str] = []
        db = getattr(self.engine, "patch_suggestion_db", None)
        if db is not None:
            try:
                failed_tags = db.failed_strategy_tags()
            except Exception:
                self.logger.exception("failed strategy tag lookup failed")
        builder = self.context_builder
        if builder is None:
            return []
        if failed_tags:
            try:
                builder.exclude_failed_strategies(failed_tags)
            except Exception:
                return []
        try:
            ctx, meta = builder.query(
                report.trace,
                top_k=5,
                return_metadata=True,
                failure=report,
                exclude_strategies=exclude,
            )
        except Exception:
            self.logger.exception("context feedback query failed")
            return []
        examples: list[str] = []
        if isinstance(meta, dict):
            for bucket in meta.values():
                for entry in bucket:
                    try:
                        ex = entry.get("example") or entry.get("code")
                    except Exception:
                        self.logger.exception(
                            "malformed context metadata: %r", entry
                        )
                        continue
                    if isinstance(ex, str):
                        examples.append(ex)
        return examples

    # ------------------------------------------------------------------
    def _load_history_stats(self, limit: int = 50) -> None:
        """Load recent score records from the local history database."""
        if not self._history_conn:
            self._history_records = []
            return
        try:
            with self._history_db() as conn:
                cur = conn.execute(
                    (
                        "SELECT coverage_delta, error_delta, roi_delta, complexity, "
                        "synergy_roi, synergy_efficiency FROM composite_history "
                        "ORDER BY id DESC LIMIT ?"
                    ),
                    (limit,),
                )
                rows = cur.fetchall()
            self._history_records = [
                (
                    float(r[0] or 0.0),
                    float(r[1] or 0.0),
                    float(r[2] or 0.0),
                    float(r[3] or 0.0),
                    float(r[4] or 0.0),
                    float(r[5] or 0.0),
                )
                for r in rows
            ]
        except Exception:
            self.logger.exception("score history load failed")
            self._history_records = []

    # ------------------------------------------------------------------
    def _update_score_weights(
        self, patch_db: PatchHistoryDB | None = None, limit: int = 50
    ) -> None:
        """Adjust ``score_weights`` using patch history statistics."""

        now = time.monotonic()
        if now - getattr(self, "_last_weights_update", 0.0) < self._weight_update_interval:
            return
        self._last_weights_update = now

        records: list[tuple[float, float, float, float, float, float]] = []
        weights_path: Path | None = None

        self._load_history_stats(limit)
        if self._history_records:
            records.extend(self._history_records[-limit:])

        if patch_db:
            weights_path = Path(getattr(patch_db, "path", "weights.db")).with_suffix(
                ".weights.json"
            )
            try:
                with self._db_lock:
                    patches = patch_db.filter()
                    patches.sort(key=lambda r: r.ts)
                    for rec in patches[-limit:]:
                        records.append(
                            (
                                float(getattr(rec, "coverage_delta", 0.0)),
                                float(rec.errors_before - rec.errors_after),
                                float(rec.roi_delta),
                                float(rec.complexity_delta),
                                float(getattr(rec, "synergy_roi", 0.0)),
                                float(getattr(rec, "synergy_efficiency", 0.0)),
                            )
                        )
                    db_weights = patch_db.get_weights()
                if db_weights:
                    self.score_weights = db_weights
            except Exception:
                self.logger.exception("score weight update failed from patch DB")

        if not records and self.audit_trail:
            try:
                if hasattr(self.audit_trail, "records"):
                    lines = list(self.audit_trail.records)[-limit:]
                else:
                    with open(self.audit_trail.path, "r", encoding="utf-8") as fh:
                        lines = fh.readlines()[-limit:]
                for line in lines:
                    try:
                        msg = line
                        if msg and msg[0] != "{":
                            msg = msg.split(" ", 1)[1]
                        rec = json.loads(msg)
                    except Exception:
                        continue
                    records.append(
                        (
                            float(rec.get("coverage_delta") or 0.0),
                            float(rec.get("error_delta") or 0.0),
                            float(rec.get("roi_delta") or 0.0),
                            float(
                                rec.get("complexity")
                                or rec.get("complexity_delta")
                                or 0.0
                            ),
                            float(rec.get("synergy_roi") or 0.0),
                            float(rec.get("synergy_efficiency") or 0.0),
                        )
                    )
            except Exception:
                self.logger.exception("score weight update failed from audit trail")

        if weights_path and weights_path.is_file():
            try:
                with open(weights_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict) and "weights" in data:
                    vals = data.get("weights")
                    if isinstance(vals, list) and len(vals) == 6:
                        self.score_weights = tuple(float(v) for v in vals)
                stats = data.get("stats") if isinstance(data, dict) else None
                if isinstance(stats, dict):
                    self._metric_stats = {
                        k: (float(v[0]), float(v[1])) for k, v in stats.items()
                    }
            except Exception:
                self.logger.exception("score weight load failed")

        if not records:
            return

        data = {
            "coverage": [r[0] for r in records][-limit:],
            "error": [r[1] for r in records][-limit:],
            "roi": [r[2] for r in records][-limit:],
            "complexity": [r[3] for r in records][-limit:],
            "synergy_roi": [r[4] for r in records][-limit:],
            "synergy_efficiency": [r[5] for r in records][-limit:],
        }

        means: dict[str, float] = {}
        vars_: dict[str, float] = {}
        for key, values in data.items():
            if not values:
                means[key] = 0.0
                vars_[key] = 1.0
                continue
            m = sum(values) / len(values)
            v = sum((x - m) ** 2 for x in values) / len(values)
            means[key] = m
            vars_[key] = v if v > 0 else 1e-6

        self._metric_stats = {k: (means[k], math.sqrt(vars_[k])) for k in data.keys()}

        def _corr(xs: list[float], ys: list[float]) -> float:
            if len(xs) < 2 or len(ys) < 2:
                return 0.0
            mx = sum(xs) / len(xs)
            my = sum(ys) / len(ys)
            num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
            den1 = sum((a - mx) ** 2 for a in xs)
            den2 = sum((b - my) ** 2 for b in ys)
            if den1 <= 0 or den2 <= 0:
                return 0.0
            return num / math.sqrt(den1 * den2)

        roi_vals = data["roi"]
        correlations = {
            "coverage": _corr(data["coverage"], roi_vals),
            "error": _corr(data["error"], roi_vals),
            "complexity": _corr(data["complexity"], roi_vals),
            "synergy_roi": _corr(data["synergy_roi"], roi_vals),
            "synergy_efficiency": _corr(data["synergy_efficiency"], roi_vals),
        }

        weights = [
            1.0 / (vars_["coverage"] + 1e-6),
            1.0 / (vars_["error"] + 1e-6),
            1.0 / (vars_["roi"] + 1e-6),
            1.0 / (vars_["complexity"] + 1e-6),
            1.0 / (vars_["synergy_roi"] + 1e-6),
            1.0 / (vars_["synergy_efficiency"] + 1e-6),
        ]

        keys = [
            "coverage",
            "error",
            "roi",
            "complexity",
            "synergy_roi",
            "synergy_efficiency",
        ]
        for i, key in enumerate(keys):
            if key in correlations:
                weights[i] *= 1.0 + max(0.0, correlations[key])
        total = sum(weights)
        if total > 0:
            new_weights = [w / total * 6.0 for w in weights]
            norm = sum(new_weights)
            if norm > 0:
                new_weights = [w / norm * 6.0 for w in new_weights]
            if self.score_weights and new_weights[4] <= self.score_weights[4]:
                increase = self.score_weights[4] + 1e-6 - new_weights[4]
                new_weights[4] += increase
                other_total = sum(new_weights) - new_weights[4]
                if other_total > 0:
                    scale = (6.0 - new_weights[4]) / other_total
                    for i in range(len(new_weights)):
                        if i != 4:
                            new_weights[i] *= scale
            self.score_weights = tuple(new_weights)
        if patch_db:
            try:
                with self._db_lock:
                    patch_db.store_weights(self.score_weights)
            except Exception:
                self.logger.exception("score weight DB persistence failed")
        if weights_path:
            try:
                with open(weights_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"weights": self.score_weights, "stats": self._metric_stats}, fh
                    )
            except Exception:
                self.logger.exception("score weight persistence failed")

    # ------------------------------------------------------------------
    def recent_scores(self, limit: int = 20) -> list[tuple]:
        """Return the most recent patch scores."""
        if self._score_backend:
            try:
                rows = with_retry(
                    lambda: self._score_backend.fetch_recent(limit),
                    logger=self.logger,
                )
                if rows:
                    return rows
            except Exception as exc:
                self.logger.warning(
                    "patch score backend unreachable",
                    extra=log_record(action="fetch", error=str(exc)),
                )
                self.logger.exception("patch score backend fetch failed")
        if not self._history_conn:
            return []
        try:
            with self._history_db() as conn:
                cur = conn.execute(
                    """
                    SELECT
                        description,
                        result,
                        coverage_delta,
                        error_delta,
                        roi_delta,
                        complexity,
                        synergy_roi,
                        synergy_efficiency,
                        synergy_resilience,
                        synergy_antifragility,
                        flakiness,
                        runtime_impact,
                        score,
                        ts
                    FROM patch_scores
                    ORDER BY id DESC LIMIT ?
                    """,
                    (int(limit),),
                )
                rows = cur.fetchall()
            return [tuple(r) for r in rows]
        except Exception:
            self.logger.exception("recent scores fetch failed")
            return []

    # ------------------------------------------------------------------
    def _composite_score(
        self,
        coverage_delta: float,
        error_delta: float,
        roi_delta: float,
        flakiness: float,
        runtime_delta: float,
        complexity: float,
        entropy_delta: float,
        synergy_roi: float | None = None,
        synergy_efficiency: float | None = None,
        synergy_resilience: float | None = None,
        synergy_antifragility: float | None = None,
        filename: str | None = None,
        *,
        tracker: ROITracker | None = None,
        weights: Mapping[str, float] | None = None,
    ) -> tuple[float, float, float]:
        """Return a composite score from multiple metrics.

        If ``weights`` is ``None`` and the attached engine provides a
        ``synergy_learner`` with a ``weights`` mapping, those values are used to
        scale the synergy metrics before applying the logistic transform.
        """
        if tracker is not None:
            s_roi, s_eff, s_res, s_af = self._recent_synergy_metrics(tracker)
            if synergy_roi is None:
                synergy_roi = s_roi
            if synergy_efficiency is None:
                synergy_efficiency = s_eff
            if synergy_resilience is None:
                synergy_resilience = s_res
            if synergy_antifragility is None:
                synergy_antifragility = s_af

        synergy_roi = float(synergy_roi or 0.0)
        synergy_efficiency = float(synergy_efficiency or 0.0)
        synergy_resilience = float(synergy_resilience or 0.0)
        synergy_antifragility = float(synergy_antifragility or 0.0)

        if weights is None and hasattr(self.engine, "synergy_learner"):
            try:
                weights = getattr(self.engine.synergy_learner, "weights", None)
            except Exception:
                weights = None
        weights = weights or {}

        if self._score_db:
            with self._db_lock:
                self._update_score_weights(self._score_db)
        else:
            self._update_score_weights(None)
        mc = {k: s for k, s in self._metric_stats.items()}
        cov_mean, cov_sd = mc.get("coverage", (0.0, 1.0))
        err_mean, err_sd = mc.get("error", (0.0, 1.0))
        roi_mean, roi_sd = mc.get("roi", (0.0, 1.0))
        comp_mean, comp_sd = mc.get("complexity", (0.0, 1.0))

        cov = (coverage_delta - cov_mean) / (cov_sd + 1e-6)
        err = (error_delta - err_mean) / (err_sd + 1e-6)
        roi = (roi_delta - roi_mean) / (roi_sd + 1e-6)
        comp = (complexity - comp_mean) / (comp_sd + 1e-6)
        syn_r = synergy_roi / (mc.get("synergy_roi", (0.0, 1.0))[1] + 1e-6)
        syn_e = synergy_efficiency / (
            mc.get("synergy_efficiency", (0.0, 1.0))[1] + 1e-6
        )
        syn_res = synergy_resilience / (
            mc.get("synergy_resilience", (0.0, 1.0))[1] + 1e-6
        )
        syn_af = synergy_antifragility / (
            mc.get("synergy_antifragility", (0.0, 1.0))[1] + 1e-6
        )
        syn_r *= float(weights.get("roi", 1.0))
        syn_e *= float(weights.get("efficiency", 1.0))
        syn_res *= float(weights.get("resilience", 1.0))
        syn_af *= float(weights.get("antifragility", 1.0))
        hist_flak = 0.0
        if filename and self._score_db:
            try:
                with self._db_lock:
                    hist_flak = self._score_db.average_flakiness(filename)
            except Exception:
                self.logger.exception("flakiness history fetch failed")
                hist_flak = 0.0
        x = (
            self.score_weights[0] * cov
            + self.score_weights[1] * err
            + self.score_weights[2] * roi
            - self.score_weights[3] * comp
            + self.score_weights[4] * syn_r
            + self.score_weights[5] * syn_e
            - flakiness
            - runtime_delta
            - hist_flak
            - abs(entropy_delta)
        )
        try:
            score = 1.0 / (1.0 + math.exp(-x))
        except Exception:
            score = 0.0

        moving_avg = self._baseline_tracker.get("score")
        moving_dev = self._baseline_tracker.std("score")

        try:
            with self._history_db() as conn:
                if conn:
                    conn.execute(
                        """
                        INSERT INTO composite_history(
                            coverage_delta,
                            error_delta,
                            roi_delta,
                            complexity,
                            synergy_roi,
                            synergy_efficiency,
                            synergy_resilience,
                            synergy_antifragility,
                            flakiness,
                            score,
                            ts
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            float(coverage_delta),
                            float(error_delta),
                            float(roi_delta),
                            float(complexity),
                            float(synergy_roi),
                            float(synergy_efficiency),
                            float(synergy_resilience),
                            float(synergy_antifragility),
                            float(flakiness),
                            float(score),
                            datetime.utcnow().isoformat(),
                        ),
                    )
        except Exception:
            self.logger.exception("score history persistence failed")

        self._baseline_tracker.update(score=score)

        return score, moving_avg, moving_dev

    # ------------------------------------------------------------------
    def _log_patch(
        self,
        description: str,
        result: str,
        before_cov: float | None = None,
        after_cov: float | None = None,
        *,
        coverage_delta: float | None = None,
        error_delta: float | None = None,
        roi_delta: float | None = None,
        score: float | None = None,
        reason: str | None = None,
        flakiness: float | None = None,
        runtime_impact: float | None = None,
        complexity: float | None = None,
        synergy_roi: float | None = None,
        synergy_efficiency: float | None = None,
        synergy_resilience: float | None = None,
        synergy_antifragility: float | None = None,
        log_path: str | None = None,
    ) -> None:
        if not self.audit_trail:
            return
        try:
            payload = json.dumps(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "action": "sandbox_patch",
                    "description": description,
                    "result": result,
                    "coverage_before": before_cov,
                    "coverage_after": after_cov,
                    "coverage_delta": coverage_delta,
                    "error_delta": error_delta,
                    "roi_delta": roi_delta,
                    "synergy_roi": synergy_roi,
                    "synergy_efficiency": synergy_efficiency,
                    "synergy_resilience": synergy_resilience,
                    "synergy_antifragility": synergy_antifragility,
                    "score": score,
                    "reason": reason,
                    "flakiness": flakiness,
                    "runtime_impact": runtime_impact,
                    "complexity": complexity,
                    "log_path": log_path,
                },
                sort_keys=True,
            )
            self.audit_trail.record(payload)
            if self._score_backend:
                try:
                    with_retry(
                        lambda: self._score_backend.store(
                            {
                                "description": description,
                                "result": result,
                                "coverage_delta": coverage_delta,
                                "error_delta": error_delta,
                                "roi_delta": roi_delta,
                                "complexity": complexity,
                                "synergy_roi": synergy_roi,
                                "synergy_efficiency": synergy_efficiency,
                                "synergy_resilience": synergy_resilience,
                                "synergy_antifragility": synergy_antifragility,
                                "flakiness": flakiness,
                                "runtime_impact": runtime_impact,
                                "score": score,
                            }
                        ),
                        logger=self.logger,
                    )
                except Exception as exc:
                    self.logger.warning(
                        "patch score backend unreachable",
                        extra=log_record(action="store", error=str(exc)),
                    )
                    self.logger.exception("patch score backend store failed")
            try:
                with self._history_db() as conn:
                    if conn:
                        conn.execute(
                            """
                            INSERT INTO patch_scores(
                                description,
                                result,
                                coverage_delta,
                                error_delta,
                                roi_delta,
                                complexity,
                                synergy_roi,
                                synergy_efficiency,
                                synergy_resilience,
                                synergy_antifragility,
                                flakiness,
                                runtime_impact,
                                score,
                                ts
                            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                            """,
                            (
                                description,
                                result,
                                None if coverage_delta is None else float(coverage_delta),
                                None if error_delta is None else float(error_delta),
                                None if roi_delta is None else float(roi_delta),
                                None if complexity is None else float(complexity),
                                None if synergy_roi is None else float(synergy_roi),
                                None if synergy_efficiency is None else float(synergy_efficiency),
                                None if synergy_resilience is None else float(synergy_resilience),
                                None
                                if synergy_antifragility is None
                                else float(synergy_antifragility),
                                None if flakiness is None else float(flakiness),
                                None if runtime_impact is None else float(runtime_impact),
                                None if score is None else float(score),
                                datetime.utcnow().isoformat(),
                            ),
                        )
            except Exception:
                self.logger.exception("patch score persistence failed")
        except Exception:
            self.logger.exception("audit trail logging failed")
        finally:
            self._update_success_metrics(result)

    # ------------------------------------------------------------------
    def analyse_and_fix(
        self,
        patch_db: PatchHistoryDB | None = None,
        limit: int = 1,
        tracker: ROITracker | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> None:  # type: ignore[override]
        """Analyse telemetry and attempt fixes with retries.

        If ``progress_cb`` is provided it will be invoked with the current
        candidate index (1-based) and the total number of candidates after each
        evaluation.
        """

        self._score_db = patch_db
        self.preemptive_fix_high_risk_modules()
        for _ in range(max(1, int(limit))):
            logs = list(self._recent_logs())
            if not logs:
                return
            tests = self._generate_tests(logs)
            attempt = 0
            patched = False
            while attempt < self._test_retries and not patched:
                total_candidates = len(tests)
                best: dict[str, object] | None = None

                async def _eval_candidate(idx: int, code: str) -> dict[str, object] | None:
                    repo_src = resolve_path(self._settings.sandbox_repo_path or ".")
                    with create_ephemeral_env(
                        repo_src, context_builder=self.context_builder
                    ) as (repo, run):
                        repo = resolve_path(repo)
                        test_path = repo / f"test_auto{os.extsep}py"
                        test_path.write_text(code)
                        test_path = resolve_path(test_path)
                        env = os.environ.copy()
                        env["PYTHONPATH"] = str(repo)
                        try:
                            env["SANDBOX_EDGE_CASES"] = json.dumps(generate_edge_cases())
                        except Exception:
                            env["SANDBOX_EDGE_CASES"] = "{}"
                        try:
                            self.engine.patch_file(test_path, "auto_debug")

                            code_hash: str | None = None
                            try:
                                with open(test_path, "rb") as fh:
                                    code_hash = _hash_code(fh.read())
                            except Exception:
                                code_hash = None

                            if code_hash:
                                if code_hash in self._bad_hashes:
                                    self.logger.info(
                                        "skipping known bad patch",
                                        extra=log_record(hash=code_hash),
                                    )
                                    return None
                                if patch_db:
                                    try:
                                        with self._db_lock:
                                            if patch_db.has_failed_strategy(code_hash):
                                                self.logger.info(
                                                    "skipping failed strategy",
                                                    extra=log_record(hash=code_hash),
                                                )
                                                self._bad_hashes.add(code_hash)
                                                return None
                                            records = patch_db.by_hash(code_hash)
                                    except Exception:
                                        self.logger.exception("patch history lookup failed")
                                        records = []
                                    if any(r.reverted or r.roi_delta <= 0 for r in records):
                                        self.logger.info(
                                            "skipping patch due to negative history",
                                            extra=log_record(hash=code_hash),
                                        )
                                        self._bad_hashes.add(code_hash)
                                        return None

                            run(
                                [
                                    "pytest",
                                    "-q",
                                    "-p",
                                    "sandbox_runner.edge_case_plugin",
                                ],
                                env=env,
                                check=True,
                                timeout=self._test_timeout,
                                capture_output=True,
                                text=True,
                            )
                        except subprocess.CalledProcessError as exc:
                            self._record_exception(exc)
                            failure = parse_failure(exc.stderr or str(exc))
                            region = extract_target_region(failure.trace)
                            if region:
                                failure.target_region = region
                                level, _ = self._attempt_tracker.level_for(region, region)
                                self._attempt_tracker.record_failure(level, region, region)
                                failure.attempts = self._attempt_tracker.attempts_for(region)
                                self._last_region = region
                            self._record_failed_strategy(failure)
                            try:
                                self.error_logger.log(
                                    TelemetryEvent(
                                        stack_trace=failure.trace,
                                        root_cause=",".join(failure.tags),
                                    )
                                )
                            except Exception:
                                self.logger.exception("failed to log parsed failure")
                            self.logger.error(
                                "sandbox tests failed",
                                extra=log_record(cmd=exc.cmd, rc=exc.returncode, output=exc.stderr),
                            )
                            if code_hash and patch_db:
                                try:
                                    with self._db_lock:
                                        patch_db.record_failed_strategy(code_hash)
                                except Exception:
                                    self.logger.exception("record failed strategy failed")
                            return None
                        except subprocess.TimeoutExpired as exc:
                            self._record_exception(exc)
                            failure = parse_failure(exc.stderr or str(exc))
                            region = extract_target_region(failure.trace)
                            if region:
                                failure.target_region = region
                                level, _ = self._attempt_tracker.level_for(region, region)
                                self._attempt_tracker.record_failure(level, region, region)
                                failure.attempts = self._attempt_tracker.attempts_for(region)
                                self._last_region = region
                            self._record_failed_strategy(failure)
                            try:
                                self.error_logger.log(
                                    TelemetryEvent(
                                        stack_trace=failure.trace,
                                        root_cause=",".join(failure.tags),
                                    )
                                )
                            except Exception:
                                self.logger.exception("failed to log parsed failure")
                            self.logger.error(
                                "sandbox tests timed out",
                                extra=log_record(
                                    cmd=exc.cmd,
                                    timeout=exc.timeout,
                                    output=exc.stderr,
                                ),
                            )
                            if code_hash and patch_db:
                                try:
                                    with self._db_lock:
                                        patch_db.record_failed_strategy(code_hash)
                                except Exception:
                                    self.logger.exception("record failed strategy failed")
                            return None
                        finally:
                            test_path.unlink(missing_ok=True)

                    root_dir = resolve_path(self._settings.sandbox_repo_path or ".")
                    root_test = root_dir / f"test_auto_{idx}.py"
                    root_test.write_text(code)
                    root_test = resolve_path(root_test)
                    result = "failed"
                    before_cov = after_cov = None
                    before_runtime = after_runtime = 0.0
                    coverage_delta = 0.0
                    error_delta = 0.0
                    roi_delta = 0.0
                    flakiness = 0.0
                    complexity = 0.0
                    pid = None
                    runtime_delta = 0.0
                    reason = None
                    try:
                        res = await asyncio.to_thread(self._run_tests, root_test)
                        before_cov, before_runtime = res[:2]
                        if len(res) == 3:
                            return None
                        before_err = getattr(
                            self.engine, "_current_errors", lambda: 0
                        )()
                        roi_before = (
                            tracker.roi_history[-1]
                            if tracker and getattr(tracker, "roi_history", None)
                            else 0.0
                        )
                        pid, reverted, roi_delta = await asyncio.to_thread(
                            self.engine.apply_patch, root_test, "auto_debug"
                        )
                        try:
                            await asyncio.to_thread(
                                post_round_orphan_scan,
                                Path.cwd(),
                                logger=self.logger,
                                router=router,
                            )
                        except Exception:
                            self.logger.exception(
                                "post_round_orphan_scan after apply_patch failed"
                            )
                        res = await asyncio.to_thread(self._run_tests, root_test)
                        after_cov, after_runtime = res[:2]
                        if len(res) == 3:
                            return None
                        after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        coverage_delta = (
                            (after_cov - before_cov)
                            if after_cov is not None and before_cov is not None
                            else 0.0
                        )
                        error_delta = before_err - after_err
                        flakiness = await asyncio.to_thread(
                            self._test_flakiness, root_test, runs=self.flakiness_runs
                        )
                        code_div, complexity, _ = compute_entropy_metrics([root_test])
                        entropy_delta, _ = compute_entropy_delta(code_div, complexity)
                        runtime_delta = after_runtime - before_runtime
                        roi_after = roi_before + roi_delta
                        if tracker is not None:
                            try:
                                tracker.update(roi_before, roi_after)
                            except Exception:
                                self.logger.exception("ROITracker update failed")
                        result = "reverted" if reverted else "success"
                        if (
                            not reverted
                            and pid is not None
                            and getattr(self.engine, "rollback_mgr", None)
                            and (coverage_delta < 0 or error_delta < 0)
                        ):
                            try:
                                self.engine.rollback_mgr.rollback(str(pid))
                            except Exception as exc:
                                reason = f"rollback failed: {exc}"
                                self._record_exception(exc)
                                self.logger.exception("rollback failed")
                            result = "reverted"
                            reverted = True
                        if not reverted:
                            syn_roi, syn_eff, *_ = self._recent_synergy_metrics(tracker)
                            score, _, _ = self._composite_score(
                                coverage_delta,
                                error_delta,
                                roi_delta,
                                flakiness,
                                runtime_delta,
                                complexity,
                                entropy_delta,
                                synergy_roi=syn_roi,
                                synergy_efficiency=syn_eff,
                                filename=str(root_test),
                                tracker=tracker,
                            )
                        else:
                            score = float("-inf")
                    except asyncio.CancelledError:
                        reason = "evaluation cancelled"
                        self.logger.error("candidate eval cancelled")
                        return None
                    except RuntimeError as exc:
                        reason = f"sandbox tests failed: {exc}"
                        self._record_exception(exc)
                        self.logger.error("sandbox tests failed", exc_info=exc)
                        score = float("-inf")
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        self._record_exception(exc)
                        self.logger.exception("patch failed")
                        score = float("-inf")
                    finally:
                        self._log_patch(
                            "auto_debug_candidate",
                            result,
                            before_cov,
                            after_cov,
                            coverage_delta=coverage_delta,
                            error_delta=error_delta,
                            roi_delta=roi_delta,
                            score=score,
                            reason=reason,
                            flakiness=flakiness,
                            runtime_impact=runtime_delta,
                            complexity=complexity,
                            synergy_roi=syn_roi if "syn_roi" in locals() else 0.0,
                            synergy_efficiency=(
                                syn_eff if "syn_eff" in locals() else 0.0
                            ),
                            log_path=(
                                str(self._last_test_log)
                                if self._last_test_log
                                else None
                            ),
                        )
                        self.logger.info(
                            "candidate evaluation",
                            extra=log_record(
                                idx=idx,
                                coverage_delta=coverage_delta,
                                roi_delta=roi_delta,
                                score=score,
                                result=result,
                            ),
                        )
                        try:
                            metrics = {
                                "success": result != "reverted",
                                "entropy_delta": entropy_delta,
                                "runtime": runtime_delta,
                                "error": reason,
                                "coverage": {
                                    "before": before_cov,
                                    "after": after_cov,
                                },
                            }
                            record_run(
                                SimpleNamespace(
                                    success=metrics.get("success"),
                                    duration=metrics.get("runtime"),
                                    failure=metrics.get("error"),
                                ),
                                metrics,
                            )
                        except Exception:
                            pass
                        if progress_cb:
                            try:
                                progress_cb(idx + 1, total_candidates)
                            except Exception:
                                self.logger.exception("progress callback failed")
                        if (
                            pid is not None
                            and result != "reverted"
                            and hasattr(self.engine, "rollback_patch")
                        ):
                            try:
                                self.logger.info(
                                    "rolling back patch",
                                    extra=log_record(
                                        patch_id=pid, module=str(root_test)
                                    ),
                                )
                                self.engine.rollback_patch(str(pid))
                            except Exception as exc:
                                self._record_exception(exc)
                                self.logger.exception(
                                    "candidate rollback failed"
                                )
                        root_test.unlink(missing_ok=True)

                    return {"score": score, "code": code}

                async def _eval_all() -> list[dict[str, object] | None]:
                    tasks = [
                        asyncio.create_task(_eval_candidate(i, c))
                        for i, c in enumerate(tests)
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    cleaned: list[dict[str, object] | None] = []
                    for res in results:
                        if isinstance(res, Exception):
                            self._record_exception(res)
                            self.logger.error("candidate eval failed", exc_info=res)
                            continue
                        cleaned.append(res)
                    return cleaned

                try:
                    results = asyncio.run(_eval_all())
                except Exception as exc:
                    self._record_exception(exc)
                    self.logger.error("candidate evaluation failed", exc_info=exc)
                    results = []

                for res in results:
                    if not res:
                        continue
                    if res["score"] > (best["score"] if best else float("-inf")):
                        best = res

                if not best:
                    break

                code = best["code"]
                root_dir = resolve_path(self._settings.sandbox_repo_path or ".")
                root_test = root_dir / f"test_auto{os.extsep}py"
                root_test.write_text(code)
                root_test = resolve_path(root_test)
                code_hash: str | None = None
                try:
                    with open(root_test, "rb") as fh:
                        code_hash = _hash_code(fh.read())
                except Exception:
                    code_hash = None
                result = "failed"
                before_cov = after_cov = None
                before_runtime = after_runtime = 0.0
                coverage_delta = 0.0
                error_delta = 0.0
                roi_delta = 0.0
                runtime_delta = 0.0
                reason = None
                try:
                    res = self._run_tests(root_test)
                    before_cov, before_runtime = res[:2]
                    if len(res) > 2:
                        return None
                    before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                    pid, reverted, roi_delta = self.engine.apply_patch(
                        root_test,
                        "auto_debug",
                        reason="auto_debug",
                        trigger="self_debugger_sandbox",
                    )
                    try:
                        post_round_orphan_scan(
                            Path.cwd(), logger=self.logger, router=router
                        )
                    except Exception:
                        self.logger.exception(
                            "post_round_orphan_scan after apply_patch failed"
                        )
                    if self.policy:
                        try:
                            state = self.state_getter() if self.state_getter else ()
                            self.policy.update(state, roi_delta)
                        except Exception as exc:
                            self.logger.exception("policy patch update failed", exc)
                    res = self._run_tests(root_test)
                    after_cov, after_runtime = res[:2]
                    if len(res) > 2:
                        return None
                    after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                    coverage_delta = (
                        (after_cov - before_cov)
                        if after_cov is not None and before_cov is not None
                        else 0.0
                    )
                    error_delta = before_err - after_err
                    flakiness = self._test_flakiness(root_test, runs=self.flakiness_runs)
                    runtime_delta = after_runtime - before_runtime
                    code_div, complexity, _ = compute_entropy_metrics([root_test])
                    entropy_delta, _ = compute_entropy_delta(code_div, complexity)
                    syn_roi, syn_eff, *_ = self._recent_synergy_metrics(tracker)
                    patch_score, moving_avg, _ = self._composite_score(
                        coverage_delta,
                        error_delta,
                        roi_delta,
                        flakiness,
                        runtime_delta,
                        complexity,
                        entropy_delta,
                        synergy_roi=syn_roi,
                        synergy_efficiency=syn_eff,
                        filename=str(root_test),
                        tracker=tracker,
                    )
                    delta = patch_score - moving_avg
                    result = "reverted" if reverted else "success"
                    if delta < self.delta_margin:
                        if pid is not None and getattr(self.engine, "rollback_mgr", None):
                            try:
                                self.engine.rollback_mgr.rollback(str(pid))
                            except Exception as exc:
                                reason = f"rollback failed: {exc}"
                                self._record_exception(exc)
                                self.logger.exception("rollback failed")
                        result = "reverted"
                        reason = (
                            f"delta {delta:.3f} below margin {self.delta_margin:.3f}"
                        )
                    elif (
                        not reverted
                        and pid is not None
                        and getattr(self.engine, "rollback_mgr", None)
                        and (coverage_delta < 0 or error_delta < 0)
                    ):
                        try:
                            self.engine.rollback_mgr.rollback(str(pid))
                        except Exception as exc:
                            reason = f"rollback failed: {exc}"
                            self._record_exception(exc)
                            self.logger.exception("rollback failed")
                        result = "reverted"
                    else:
                        patched = not reverted and delta >= self.delta_margin
                except RuntimeError as exc:
                    reason = f"sandbox tests failed: {exc}"
                    failure = parse_failure(str(exc))
                    region = extract_target_region(failure.trace)
                    if region:
                        failure.target_region = region
                        level, _ = self._attempt_tracker.level_for(region, region)
                        self._attempt_tracker.record_failure(level, region, region)
                        failure.attempts = self._attempt_tracker.attempts_for(region)
                        self._last_region = region
                    self._record_failed_strategy(failure)
                    self._record_exception(exc)
                    try:
                        self.error_logger.log(
                            TelemetryEvent(
                                stack_trace=failure.trace,
                                root_cause=",".join(failure.tags),
                            )
                        )
                    except Exception:
                        self.logger.exception("failed to log parsed failure")
                    self.logger.error("sandbox tests failed", exc_info=exc)
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    failure = parse_failure(str(exc))
                    region = extract_target_region(failure.trace)
                    if region:
                        failure.target_region = region
                        level, _ = self._attempt_tracker.level_for(region, region)
                        self._attempt_tracker.record_failure(level, region, region)
                        failure.attempts = self._attempt_tracker.attempts_for(region)
                        self._last_region = region
                    self._record_failed_strategy(failure)
                    self._record_exception(exc)
                    try:
                        self.error_logger.log(
                            TelemetryEvent(
                                stack_trace=failure.trace,
                                root_cause=",".join(failure.tags),
                            )
                        )
                    except Exception:
                        self.logger.exception("failed to log parsed failure")
                    self.logger.exception("patch failed")
                finally:
                    self._log_patch(
                        "auto_debug",
                        result,
                        before_cov,
                        after_cov,
                        coverage_delta=coverage_delta,
                        error_delta=error_delta,
                        roi_delta=roi_delta,
                        score=patch_score if "patch_score" in locals() else None,
                        flakiness=flakiness if "flakiness" in locals() else None,
                        runtime_impact=(
                            runtime_delta if "runtime_delta" in locals() else None
                        ),
                        complexity=complexity if "complexity" in locals() else None,
                        synergy_roi=syn_roi if "syn_roi" in locals() else 0.0,
                        synergy_efficiency=syn_eff if "syn_eff" in locals() else 0.0,
                        log_path=str(self._last_test_log) if self._last_test_log else None,
                        reason=reason,
                    )
                    try:
                        metrics = {
                            "success": result != "reverted",
                            "entropy_delta": entropy_delta if "entropy_delta" in locals() else 0.0,
                            "runtime": runtime_delta if "runtime_delta" in locals() else 0.0,
                            "error": reason,
                            "coverage": {
                                "before": before_cov,
                                "after": after_cov,
                            },
                        }
                        record_run(
                            SimpleNamespace(
                                success=metrics.get("success"),
                                duration=metrics.get("runtime"),
                                failure=metrics.get("error"),
                            ),
                            metrics,
                        )
                    except Exception:
                        pass
                    root_test.unlink(missing_ok=True)
                if patched and pid is not None:
                    try:
                        from .patch_branch_manager import finalize_patch_branch

                        finalize_patch_branch(
                            str(pid),
                            patch_score,
                            self.merge_threshold,
                            audit_trail=self.audit_trail,
                        )
                    except Exception:
                        self.logger.exception("patch branch finalization failed")
                if not patched:
                    if code_hash and patch_db:
                        try:
                            with self._db_lock:
                                patch_db.record_failed_strategy(code_hash)
                        except Exception:
                            self.logger.exception("record failed strategy failed")
                    if failure:
                        tests.extend(self._context_feedback(failure))
                    attempt += 1
                else:
                    if self._last_region is not None:
                        self._attempt_tracker.reset(self._last_region)
                        self._last_region = None
                    break
            if patched:
                break


__all__ = ["SelfDebuggerSandbox"]
