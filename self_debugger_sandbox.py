from __future__ import annotations

"""Self-debugging workflow with sandboxed patch testing."""

import logging
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
from statistics import pstdev, fmean
from coverage import Coverage

from .automated_debugger import AutomatedDebugger
from .self_coding_engine import SelfCodingEngine
from .audit_trail import AuditTrail
from .code_database import PatchHistoryDB, _hash_code
from .self_improvement_policy import SelfImprovementPolicy
from typing import Callable


class SelfDebuggerSandbox(AutomatedDebugger):
    """Extend AutomatedDebugger with sandbox verification.

    Parameters
    ----------
    flakiness_runs:
        Number of test executions used when estimating flakiness.
    """

    def __init__(
        self,
        telemetry_db: object,
        engine: SelfCodingEngine,
        audit_trail: AuditTrail | None = None,
        policy: SelfImprovementPolicy | None = None,
        state_getter: Callable[[], tuple[int, ...]] | None = None,
        *,
        score_threshold: float = 0.5,
        score_weights: tuple[float, float, float, float] | None = None,
        flakiness_runs: int = 5,
    ) -> None:
        super().__init__(telemetry_db, engine)
        self.audit_trail = audit_trail or getattr(engine, "audit_trail", None)
        self.policy = policy
        self.state_getter = state_getter
        self._bad_hashes: set[str] = set()
        self.score_threshold = score_threshold
        self.score_weights = score_weights or (1.0, 1.0, 1.0, 1.0)
        self.flakiness_runs = max(1, int(flakiness_runs))
        self._score_db: PatchHistoryDB | None = None
        self._metric_stats: dict[str, tuple[float, float]] = {
            "coverage": (0.0, 1.0),
            "error": (0.0, 1.0),
            "roi": (0.0, 1.0),
            "complexity": (0.0, 1.0),
        }

    # ------------------------------------------------------------------
    def _coverage_percent(self, paths: list[Path], env: dict[str, str] | None = None) -> float:
        """Run tests for *paths* under coverage using parallel workers asynchronously."""

        async def run_sets(path_sets: list[list[Path]]) -> float:
            tmp_dir = Path(tempfile.mkdtemp())
            coverage_files: list[Path] = []
            procs = []
            for idx, set_paths in enumerate(path_sets):
                data_file = tmp_dir / f".coverage.{idx}"
                coverage_files.append(data_file)
                p_env = dict(env or os.environ)
                p_env["COVERAGE_FILE"] = str(data_file)
                cmd = [
                    sys.executable,
                    "-m",
                    "coverage",
                    "run",
                    "--parallel-mode",
                    "-m",
                    "pytest",
                    "-q",
                    "-n",
                    "auto",
                    *[str(p) for p in set_paths],
                ]
                proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        env=p_env,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                )
                procs.append((proc, cmd))

            for proc, cmd in procs:
                await proc.wait()
                if proc.returncode != 0:
                    self.logger.error("test run failed", extra={"cmd": cmd, "rc": proc.returncode})

            cov = Coverage(data_file=str(tmp_dir / ".coverage"))
            cov.combine([str(f) for f in coverage_files])
            buf = io.StringIO()
            xml_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
            xml_tmp.close()
            percent = 0.0
            try:
                cov.xml_report(
                    outfile=xml_tmp.name,
                    include=[str(p) for paths in path_sets for p in paths],
                )
                percent = cov.report(
                    include=[str(p) for paths in path_sets for p in paths],
                    file=buf,
                )
            except Exception:
                self.logger.exception("coverage generation failed")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                try:
                    os.unlink(xml_tmp.name)
                except Exception:
                    self.logger.exception("coverage cleanup failed")
            return float(percent or 0.0)

        # Support a list of paths treated as individual test sets
        if paths and isinstance(paths[0], (list, tuple)):
            sets = [list(p) for p in paths]  # type: ignore[list-item]
        else:
            sets = [[p] for p in paths]

        return asyncio.run(run_sets(sets))

    # ------------------------------------------------------------------
    def _run_tests(self, path: Path, env: dict[str, str] | None = None) -> tuple[float, float]:
        """Return coverage percentage and runtime for tests at *path* with telemetry tests."""
        test_paths = [path]
        tmp: Path | None = None
        try:
            logs = list(self._recent_logs())
            tests = self._generate_tests(logs)
            if tests:
                tf = tempfile.NamedTemporaryFile(delete=False, suffix="_telemetry.py")
                tf.write("\n\n".join(tests).encode("utf-8"))
                tf.close()
                tmp = Path(tf.name)
                test_paths.append(tmp)
        except Exception:
            self.logger.exception("failed to create telemetry tests")
        start = time.perf_counter()
        cov = self._coverage_percent(test_paths, env)
        runtime = time.perf_counter() - start
        if tmp:
            tmp.unlink(missing_ok=True)
        return cov, runtime

    # ------------------------------------------------------------------
    def _test_flakiness(
        self,
        path: Path,
        env: dict[str, str] | None = None,
        *,
        runs: int | None = None,
    ) -> float:
        """Return the standard deviation of coverage across multiple test runs."""
        n = runs if runs is not None else self.flakiness_runs
        coverages = []
        for _ in range(max(1, int(n))):
            cov, _ = self._run_tests(path, env)
            coverages.append(cov)
        return pstdev(coverages) if len(coverages) > 1 else 0.0

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
    def _update_score_weights(
        self, patch_db: PatchHistoryDB | None = None, limit: int = 50
    ) -> None:
        """Adjust ``score_weights`` using recent patch metrics and update rolling statistics."""

        records: list[tuple[float, float, float, float]] = []

        if patch_db:
            try:
                patches = patch_db.filter()
                patches.sort(key=lambda r: r.ts)
                for rec in patches[-limit:]:
                    records.append(
                        (
                            0.0,
                            float(rec.errors_before - rec.errors_after),
                            float(rec.roi_delta),
                            float(rec.complexity_delta),
                        )
                    )
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
                            float(rec.get("complexity") or rec.get("complexity_delta") or 0.0),
                        )
                    )
            except Exception:
                self.logger.exception("score weight update failed from audit trail")

        if not records:
            return

        cov = [r[0] for r in records][-limit:]
        err = [r[1] for r in records][-limit:]
        roi = [r[2] for r in records][-limit:]
        comp = [r[3] for r in records][-limit:]

        means = {
            "coverage": fmean(abs(v) for v in cov) if cov else 0.0,
            "error": fmean(abs(v) for v in err) if err else 0.0,
            "roi": fmean(abs(v) for v in roi) if roi else 0.0,
            "complexity": fmean(abs(v) for v in comp) if comp else 0.0,
        }
        stds = {
            "coverage": pstdev(cov) if len(cov) > 1 else 0.0,
            "error": pstdev(err) if len(err) > 1 else 0.0,
            "roi": pstdev(roi) if len(roi) > 1 else 0.0,
            "complexity": pstdev(comp) if len(comp) > 1 else 0.0,
        }
        self._metric_stats = {k: (means[k], stds[k]) for k in means}

        weights = [
            means["coverage"] / (stds["coverage"] + 1e-6),
            means["error"] / (stds["error"] + 1e-6),
            means["roi"] / (stds["roi"] + 1e-6),
            means["complexity"] / (stds["complexity"] + 1e-6),
        ]
        total = sum(weights)
        if total > 0:
            self.score_weights = tuple(w / total * 4.0 for w in weights)

    # ------------------------------------------------------------------
    def _composite_score(
        self,
        coverage_delta: float,
        error_delta: float,
        roi_delta: float,
        flakiness: float,
        runtime_delta: float,
        complexity: float,
    ) -> float:
        """Return a composite score from multiple metrics."""
        self._update_score_weights(self._score_db)
        mc = {k: s for k, s in self._metric_stats.items()}
        cov = coverage_delta / (mc.get("coverage", (0.0, 1.0))[1] + 1e-6)
        err = error_delta / (mc.get("error", (0.0, 1.0))[1] + 1e-6)
        roi = roi_delta / (mc.get("roi", (0.0, 1.0))[1] + 1e-6)
        comp = complexity / (mc.get("complexity", (0.0, 1.0))[1] + 1e-6)
        x = (
            self.score_weights[0] * cov
            + self.score_weights[1] * err
            + self.score_weights[2] * roi
            - self.score_weights[3] * comp
            - flakiness
            - runtime_delta
        )
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except Exception:
            return 0.0

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
                    "score": score,
                    "reason": reason,
                    "flakiness": flakiness,
                    "runtime_impact": runtime_impact,
                    "complexity": complexity,
                },
                sort_keys=True,
            )
            self.audit_trail.record(payload)
        except Exception:
            self.logger.exception("audit trail logging failed")

    # ------------------------------------------------------------------
    def analyse_and_fix(
        self, patch_db: PatchHistoryDB | None = None, limit: int = 1
    ) -> None:  # type: ignore[override]
        """Analyse telemetry and attempt fixes with retries."""

        self._score_db = patch_db
        for _ in range(max(1, int(limit))):
            logs = list(self._recent_logs())
            if not logs:
                return
            tests = self._generate_tests(logs)
            best: dict[str, object] | None = None
            for code in tests:
                with tempfile.TemporaryDirectory() as tmp:
                    tdir = Path(tmp)
                    repo = tdir / "repo"
                    shutil.copytree(Path("."), repo, dirs_exist_ok=True)
                    test_path = repo / "test_auto.py"
                    test_path.write_text(code)
                    env = os.environ.copy()
                    env["PYTHONPATH"] = str(repo)
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
                                    "skipping known bad patch", extra={"hash": code_hash}
                                )
                                continue
                            if patch_db:
                                try:
                                    records = patch_db.by_hash(code_hash)
                                except Exception:
                                    records = []
                                if any(r.reverted or r.roi_delta <= 0 for r in records):
                                    self.logger.info(
                                        "skipping patch due to negative history",
                                        extra={"hash": code_hash},
                                    )
                                    self._bad_hashes.add(code_hash)
                                    continue

                        subprocess.run(["pytest", "-q"], cwd=str(repo), check=True, env=env)
                    except Exception:
                        self.logger.exception("sandbox tests failed")
                        continue

                    root_test = Path("test_auto.py")
                    root_test.write_text(code)
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
                        before_cov, before_runtime = self._run_tests(root_test)
                        before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        pid, reverted, roi_delta = self.engine.apply_patch(root_test, "auto_debug")
                        after_cov, after_runtime = self._run_tests(root_test)
                        after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        coverage_delta = (after_cov - before_cov) if after_cov is not None and before_cov is not None else 0.0
                        error_delta = before_err - after_err
                        flakiness = self._test_flakiness(root_test, runs=self.flakiness_runs)
                        complexity = self._code_complexity(root_test)
                        runtime_delta = after_runtime - before_runtime
                        result = "reverted" if reverted else "success"
                        if (
                            not reverted
                            and pid is not None
                            and getattr(self.engine, "rollback_mgr", None)
                            and (coverage_delta < 0 or error_delta < 0)
                        ):
                            try:
                                self.engine.rollback_mgr.rollback(str(pid))
                            except Exception:
                                self.logger.exception("rollback failed")
                            result = "reverted"
                            reverted = True
                        if not reverted:
                            score = self._composite_score(
                                coverage_delta,
                                error_delta,
                                roi_delta,
                                flakiness,
                                runtime_delta,
                                complexity,
                            )
                        else:
                            score = float("-inf")
                    except Exception:
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
                            flakiness=flakiness,
                            runtime_impact=runtime_delta,
                            complexity=complexity,
                        )
                        if pid is not None and result != "reverted":
                            try:
                                self.engine.rollback_patch(str(pid))
                            except Exception:
                                self.logger.exception("candidate rollback failed")
                        root_test.unlink(missing_ok=True)

                if score > (best["score"] if best else float("-inf")):
                    best = {"score": score, "code": code}

            if not best:
                continue

            code = best["code"]
            root_test = Path("test_auto.py")
            root_test.write_text(code)
            result = "failed"
            before_cov = after_cov = None
            before_runtime = after_runtime = 0.0
            coverage_delta = 0.0
            error_delta = 0.0
            roi_delta = 0.0
            patched = False
            runtime_delta = 0.0
            reason = None
            try:
                before_cov, before_runtime = self._run_tests(root_test)
                before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                pid, reverted, roi_delta = self.engine.apply_patch(root_test, "auto_debug")
                if self.policy:
                    try:
                        state = self.state_getter() if self.state_getter else ()
                        self.policy.update(state, roi_delta)
                    except Exception as exc:
                        self.logger.exception("policy patch update failed", exc)
                after_cov, after_runtime = self._run_tests(root_test)
                after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                coverage_delta = (after_cov - before_cov) if after_cov is not None and before_cov is not None else 0.0
                error_delta = before_err - after_err
                flakiness = self._test_flakiness(root_test, runs=self.flakiness_runs)
                complexity = self._code_complexity(root_test)
                runtime_delta = after_runtime - before_runtime
                patch_score = self._composite_score(
                    coverage_delta,
                    error_delta,
                    roi_delta,
                    flakiness,
                    runtime_delta,
                    complexity,
                )
                result = "reverted" if reverted else "success"
                reason = None
                if patch_score < self.score_threshold:
                    if pid is not None and getattr(self.engine, "rollback_mgr", None):
                        try:
                            self.engine.rollback_mgr.rollback(str(pid))
                        except Exception:
                            self.logger.exception("rollback failed")
                    result = "reverted"
                    reason = f"score {patch_score:.3f} below threshold {self.score_threshold:.3f}"
                elif (
                    not reverted
                    and pid is not None
                    and getattr(self.engine, "rollback_mgr", None)
                    and (coverage_delta < 0 or error_delta < 0)
                ):
                    try:
                        self.engine.rollback_mgr.rollback(str(pid))
                    except Exception:
                        self.logger.exception("rollback failed")
                    result = "reverted"
                else:
                    patched = not reverted and patch_score >= self.score_threshold
            except Exception:
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
                    score=patch_score if 'patch_score' in locals() else None,
                    flakiness=flakiness if 'flakiness' in locals() else None,
                    runtime_impact=runtime_delta if 'runtime_delta' in locals() else None,
                    complexity=complexity if 'complexity' in locals() else None,
                    reason=reason,
                )
                root_test.unlink(missing_ok=True)
            if patched:
                break


__all__ = ["SelfDebuggerSandbox"]
