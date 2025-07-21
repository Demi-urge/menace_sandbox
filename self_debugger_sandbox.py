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
from statistics import pstdev
from coverage import Coverage

from .automated_debugger import AutomatedDebugger
from .self_coding_engine import SelfCodingEngine
from .audit_trail import AuditTrail
from .code_database import PatchHistoryDB, _hash_code
from .self_improvement_policy import SelfImprovementPolicy
from .roi_tracker import ROITracker
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
        score_weights: tuple[float, float, float, float, float, float] | None = None,
        flakiness_runs: int = 5,
        smoothing_factor: float = 0.5,
    ) -> None:
        super().__init__(telemetry_db, engine)
        self.audit_trail = audit_trail or getattr(engine, "audit_trail", None)
        self.policy = policy
        self.state_getter = state_getter
        self._bad_hashes: set[str] = set()
        self.score_threshold = score_threshold
        self.score_weights = score_weights or (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        self.flakiness_runs = max(1, int(flakiness_runs))
        self.smoothing_factor = max(0.0, min(1.0, float(smoothing_factor))) or 0.5
        self._score_db: PatchHistoryDB | None = None
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
        self._last_test_log: Path | None = None

    # ------------------------------------------------------------------
    async def _coverage_percent(self, paths: list[Path], env: dict[str, str] | None = None) -> float:
        """Run tests for *paths* under coverage using parallel workers asynchronously."""

        async def run_one(idx: int, set_paths: list[Path]) -> tuple[Path, int]:
            data_file = tmp_dir / f".coverage.{idx}"
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
            await proc.wait()
            if proc.returncode != 0:
                self.logger.error(
                    "test run failed", extra={"cmd": cmd, "rc": proc.returncode}
                )
            return data_file, int(proc.returncode)

        # Support a list of paths treated as individual test sets
        if paths and isinstance(paths[0], (list, tuple)):
            sets = [list(p) for p in paths]  # type: ignore[list-item]
        else:
            sets = [[p] for p in paths]

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            results = await asyncio.gather(
                *(run_one(idx, sp) for idx, sp in enumerate(sets))
            )
            coverage_files = [r[0] for r in results]
            returncodes = [r[1] for r in results]
            if any(rc != 0 for rc in returncodes):
                raise RuntimeError("test subprocess failed")
            cov = Coverage(data_file=str(tmp_dir / ".coverage"))
            cov.combine([str(f) for f in coverage_files])
            buf = io.StringIO()
            xml_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
            xml_tmp.close()
            percent = 0.0
            try:
                cov.xml_report(
                    outfile=xml_tmp.name,
                    include=[str(p) for paths in sets for p in paths],
                )
                percent = cov.report(
                    include=[str(p) for paths in sets for p in paths],
                    file=buf,
                )
            except Exception:
                self.logger.exception("coverage generation failed")
            finally:
                try:
                    os.unlink(xml_tmp.name)
                except Exception:
                    self.logger.exception("coverage cleanup failed")
            return float(percent or 0.0)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    def _run_tests(self, path: Path, env: dict[str, str] | None = None) -> tuple[float, float]:
        """Return coverage percentage and runtime for tests at *path* with telemetry tests."""
        test_paths = [path]
        tmp: Path | None = None
        self._last_test_log = None
        sandbox_dir = Path(os.getenv("SANDBOX_DATA_DIR", "sandbox_data"))
        sandbox_dir.mkdir(parents=True, exist_ok=True)
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

        cov_file = tempfile.NamedTemporaryFile(delete=False, dir=sandbox_dir, suffix=".cov")
        cov_file.close()
        p_env = dict(env or os.environ)
        p_env["COVERAGE_FILE"] = cov_file.name
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
            *[str(p) for p in test_paths],
        ]

        start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, env=p_env)
            output = proc.stdout + proc.stderr
            rc = proc.returncode
        except Exception:
            rc = 1
            output = ""
            self.logger.exception("test execution failed")
        runtime = time.perf_counter() - start

        if rc != 0:
            with tempfile.NamedTemporaryFile(delete=False, dir=sandbox_dir, suffix=".log", mode="w", encoding="utf-8") as lf:
                lf.write(output)
                self._last_test_log = Path(lf.name)

        percent = 0.0
        try:
            cov = Coverage(data_file=cov_file.name)
            cov.load()
            buf = io.StringIO()
            xml_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
            xml_tmp.close()
            try:
                cov.xml_report(outfile=xml_tmp.name, include=[str(p) for p in test_paths])
                percent = cov.report(include=[str(p) for p in test_paths], file=buf)
            except Exception:
                self.logger.exception("coverage generation failed")
            finally:
                try:
                    os.unlink(xml_tmp.name)
                except Exception:
                    self.logger.exception("coverage cleanup failed")
        finally:
            try:
                os.unlink(cov_file.name)
            except Exception:
                self.logger.exception("coverage cleanup failed")

        if tmp:
            tmp.unlink(missing_ok=True)

        return float(percent or 0.0), runtime

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

        async def _run_all() -> list[float]:
            tasks = [
                asyncio.to_thread(self._run_tests, path, env) for _ in range(n)
            ]
            results = await asyncio.gather(*tasks)
            return [cov for cov, _ in results]

        coverages = asyncio.run(_run_all())
        if len(coverages) <= 1:
            return 0.0
        stdev = pstdev(coverages)
        return stdev / math.sqrt(len(coverages))

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
    def _update_score_weights(
        self, patch_db: PatchHistoryDB | None = None, limit: int = 50
    ) -> None:
        """Adjust ``score_weights`` using patch history statistics."""

        records: list[tuple[float, float, float, float, float, float]] = []
        weights_path: Path | None = None

        if patch_db:
            weights_path = Path(getattr(patch_db, "path", "weights.db")).with_suffix(
                ".weights.json"
            )
            try:
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
                            float(rec.get("complexity") or rec.get("complexity_delta") or 0.0),
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

        self._metric_stats = {
            k: (means[k], math.sqrt(vars_[k])) for k in data.keys()
        }

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

        weights = [1.0 / (vars_["coverage"] + 1e-6),
                   1.0 / (vars_["error"] + 1e-6),
                   1.0 / (vars_["roi"] + 1e-6),
                   1.0 / (vars_["complexity"] + 1e-6),
                   1.0 / (vars_["synergy_roi"] + 1e-6),
                   1.0 / (vars_["synergy_efficiency"] + 1e-6)]

        keys = ["coverage", "error", "roi", "complexity", "synergy_roi", "synergy_efficiency"]
        for i, key in enumerate(keys):
            if key in correlations:
                weights[i] *= 1.0 + max(0.0, correlations[key])
        total = sum(weights)
        if total > 0:
            self.score_weights = tuple(w / total * 6.0 for w in weights)
        if patch_db:
            try:
                patch_db.store_weights(self.score_weights)
            except Exception:
                self.logger.exception("score weight DB persistence failed")
        if weights_path:
            try:
                with open(weights_path, "w", encoding="utf-8") as fh:
                    json.dump({"weights": self.score_weights, "stats": self._metric_stats}, fh)
            except Exception:
                self.logger.exception("score weight persistence failed")

    # ------------------------------------------------------------------
    def _composite_score(
        self,
        coverage_delta: float,
        error_delta: float,
        roi_delta: float,
        flakiness: float,
        runtime_delta: float,
        complexity: float,
        synergy_roi: float | None = None,
        synergy_efficiency: float | None = None,
        synergy_resilience: float | None = None,
        synergy_antifragility: float | None = None,
        *,
        tracker: ROITracker | None = None,
    ) -> float:
        """Return a composite score from multiple metrics."""
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

        self._update_score_weights(self._score_db)
        mc = {k: s for k, s in self._metric_stats.items()}
        cov = coverage_delta / (mc.get("coverage", (0.0, 1.0))[1] + 1e-6)
        err = error_delta / (mc.get("error", (0.0, 1.0))[1] + 1e-6)
        roi = roi_delta / (mc.get("roi", (0.0, 1.0))[1] + 1e-6)
        comp = complexity / (mc.get("complexity", (0.0, 1.0))[1] + 1e-6)
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
        x = (
            self.score_weights[0] * cov
            + self.score_weights[1] * err
            + self.score_weights[2] * roi
            - self.score_weights[3] * comp
            + self.score_weights[4] * syn_r
            + self.score_weights[5] * syn_e
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
        except Exception:
            self.logger.exception("audit trail logging failed")

    # ------------------------------------------------------------------
    def analyse_and_fix(
        self,
        patch_db: PatchHistoryDB | None = None,
        limit: int = 1,
        tracker: ROITracker | None = None,
    ) -> None:  # type: ignore[override]
        """Analyse telemetry and attempt fixes with retries."""

        self._score_db = patch_db
        for _ in range(max(1, int(limit))):
            logs = list(self._recent_logs())
            if not logs:
                return
            tests = self._generate_tests(logs)
            best: dict[str, object] | None = None

            async def _eval_candidate(idx: int, code: str) -> dict[str, object] | None:
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
                                return None
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
                                    return None

                        subprocess.run(["pytest", "-q"], cwd=str(repo), check=True, env=env)
                    except Exception:
                        self.logger.exception("sandbox tests failed")
                        return None

                    root_test = Path(f"test_auto_{idx}.py")
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
                        before_cov, before_runtime = await asyncio.to_thread(self._run_tests, root_test)
                        before_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        pid, reverted, roi_delta = await asyncio.to_thread(
                            self.engine.apply_patch, root_test, "auto_debug"
                        )
                        after_cov, after_runtime = await asyncio.to_thread(self._run_tests, root_test)
                        after_err = getattr(self.engine, "_current_errors", lambda: 0)()
                        coverage_delta = (after_cov - before_cov) if after_cov is not None and before_cov is not None else 0.0
                        error_delta = before_err - after_err
                        flakiness = await asyncio.to_thread(
                            self._test_flakiness, root_test, runs=self.flakiness_runs
                        )
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
                            syn_roi, syn_eff = self._recent_synergy_metrics(tracker)
                            score = self._composite_score(
                                coverage_delta,
                                error_delta,
                                roi_delta,
                                flakiness,
                                runtime_delta,
                                complexity,
                                synergy_roi=syn_roi,
                                synergy_efficiency=syn_eff,
                                tracker=tracker,
                            )
                        else:
                            score = float("-inf")
                    except RuntimeError as exc:
                        self.logger.error("sandbox tests failed", exc_info=exc)
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
                            synergy_roi=syn_roi if 'syn_roi' in locals() else 0.0,
                            synergy_efficiency=syn_eff if 'syn_eff' in locals() else 0.0,
                            log_path=str(self._last_test_log) if self._last_test_log else None,
                        )
                        if pid is not None and result != "reverted":
                            try:
                                self.engine.rollback_patch(str(pid))
                            except Exception:
                                self.logger.exception("candidate rollback failed")
                        root_test.unlink(missing_ok=True)

                    return {"score": score, "code": code}

            async def _eval_all() -> list[dict[str, object] | None]:
                tasks = [_eval_candidate(i, c) for i, c in enumerate(tests)]
                return await asyncio.gather(*tasks)

            results = asyncio.run(_eval_all())

            for res in results:
                if not res:
                    continue
                if res["score"] > (best["score"] if best else float("-inf")):
                    best = res

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
                syn_roi, syn_eff = self._recent_synergy_metrics(tracker)
                patch_score = self._composite_score(
                    coverage_delta,
                    error_delta,
                    roi_delta,
                    flakiness,
                    runtime_delta,
                    complexity,
                    synergy_roi=syn_roi,
                    synergy_efficiency=syn_eff,
                    tracker=tracker,
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
            except RuntimeError as exc:
                self.logger.error("sandbox tests failed", exc_info=exc)
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
                    synergy_roi=syn_roi if 'syn_roi' in locals() else 0.0,
                    synergy_efficiency=syn_eff if 'syn_eff' in locals() else 0.0,
                    log_path=str(self._last_test_log) if self._last_test_log else None,
                    reason=reason,
                )
                root_test.unlink(missing_ok=True)
            if patched:
                break


__all__ = ["SelfDebuggerSandbox"]
