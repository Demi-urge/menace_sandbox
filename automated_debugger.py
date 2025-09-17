from __future__ import annotations

"""Automatic debugging service that patches errors without manual help."""

from pathlib import Path
import logging
import re
import tempfile
import traceback
from typing import Any, Iterable

from .self_coding_engine import SelfCodingEngine
try:  # pragma: no cover - optional self-coding dependency
    from .self_coding_manager import SelfCodingManager
except ImportError:  # pragma: no cover - self-coding unavailable
    SelfCodingManager = Any  # type: ignore
from .retry_utils import retry
from .self_improvement.target_region import TargetRegion, extract_target_region
from .patch_attempt_tracker import PatchAttemptTracker
from .vector_service.context_builder import ContextBuilder
from .bot_registry import BotRegistry
from .data_bot import DataBot
from .coding_bot_interface import self_coding_managed


_FRAME_RE = re.compile(r"File \"([^\"]+)\", line (\d+), in ([^\n]+)")

registry = BotRegistry()
data_bot = DataBot(start_server=False)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class AutomatedDebugger:
    """Analyse telemetry logs and trigger self-coding fixes."""

    manager: SelfCodingManager

    def __init__(
        self,
        telemetry_db: object,
        context_builder: ContextBuilder,
        engine: SelfCodingEngine | None = None,
        *,
        manager: SelfCodingManager,
    ) -> None:
        if context_builder is None or not isinstance(context_builder, ContextBuilder):
            raise TypeError("context_builder must be a ContextBuilder instance")
        context_builder.refresh_db_weights()
        self.telemetry_db = telemetry_db
        self.engine = engine
        self.manager = manager
        self.context_builder = context_builder
        self.logger = logging.getLogger("AutomatedDebugger")
        self._tracker = PatchAttemptTracker()
        try:
            name = getattr(self, "name", getattr(self, "bot_name", self.__class__.__name__))
            self.manager.register_bot(name)
            orch = getattr(self.manager, "evolution_orchestrator", None)
            if orch:
                orch.register_bot(name)
        except Exception:
            self.logger.exception("bot registration failed")

    # ------------------------------------------------------------------
    def _recent_logs(self, limit: int = 5) -> Iterable[str]:
        try:
            return self.telemetry_db.recent_errors(limit=limit)
        except Exception:
            return []

    def _generate_tests(self, logs: Iterable[str]) -> list[str]:
        """Create simple tests derived from stack traces."""
        tests: list[str] = []

        def _parse(log: str) -> tuple[str, str] | None:
            try:
                frames: list[traceback.FrameSummary] = []
                for line in log.splitlines():
                    line_stripped = line.lstrip()
                    if not line_stripped.startswith("File"):
                        continue
                    rest = line_stripped[4:].lstrip()
                    if rest.startswith("\"") or rest.startswith("'"):
                        q = rest[0]
                        end = rest.find(q, 1)
                        if end == -1:
                            continue
                        file = rest[1:end]
                        rest = rest[end + 1:].lstrip(", ")
                    else:
                        parts = rest.split(",", 1)
                        file = parts[0].strip()
                        rest = parts[1] if len(parts) > 1 else ""
                    if rest.startswith("line"):
                        rest = rest[4:].lstrip()
                    lineno_part, _, func_part = rest.partition(", in ")
                    try:
                        lineno = int(lineno_part.strip())
                    except Exception:
                        continue
                    func = func_part.strip() or "<module>"
                    frames.append(traceback.FrameSummary(file, lineno, func))
                if not frames:
                    return None
                for fr in reversed(frames):
                    if Path(fr.filename).is_file():
                        return fr.filename, fr.name
                fr = frames[-1]
                return fr.filename, fr.name
            except Exception:
                return None

        pattern = re.compile(r"File '([^']+)', line (\d+)(?:, in ([^\s]+))?")
        for i, log in enumerate(logs):
            try:
                self.context_builder.build_context(str(log))
            except Exception:
                self.logger.exception("context build failed")
            parsed = _parse(log)
            if not parsed:
                matches = pattern.findall(log)
                if matches:
                    file, _line, func = matches[-1]
                else:
                    tests.append(f"def test_auto_{i}():\n    raise Exception({log!r})\n")
                    continue
            else:
                file, func = parsed
                if func == "<module>":
                    func = ""

            path = Path(file)
            if path.is_file():
                try:
                    rel = path.resolve().relative_to(Path.cwd())
                    modname = rel.with_suffix("").as_posix().replace("/", ".")
                except Exception:
                    modname = path.stem
            else:
                modname = path.stem

            lines = [
                f"def test_auto_{i}():",
                "    import importlib, inspect, pytest",
                "    try:",
                f"        mod = importlib.import_module('{modname}')",
                "    except Exception as exc:",
                "        print(f'cannot import {modname}: {exc}')",
                "        pytest.skip(f'cannot import {modname}: {exc}')",
            ]
            if func:
                lines.append(f"    fn = getattr(mod, '{func}', None)")
                lines.append("    assert fn is not None")
                lines.append("    sig = inspect.signature(fn)")
                lines.append(
                    "    req = [p for p in sig.parameters.values() if "
                    "p.default is inspect._empty and "
                    "p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]"
                )
                lines.append("    if req:")
                lines.append(f"        pytest.skip('function {func} requires args')")
            else:
                lines.append("    src = inspect.getsource(mod)")
                lines.append("    assert src")
            ematch = re.search(r'([A-Za-z_]*Error)', log)
            etype = ematch.group(1) if ematch else "Exception"
            lines.append("    try:")
            if func:
                lines.append("        fn()")
            else:
                lines.append("        importlib.reload(mod)")
            lines.append(f"    except {etype} as exc:")
            lines.append("        pytest.fail(str(exc))")
            tests.append("\n".join(lines) + "\n")

        return tests

    # ------------------------------------------------------------------
    def analyse_and_fix(self) -> None:
        logs = list(self._recent_logs())
        if not logs:
            return
        tests = self._generate_tests(logs)
        for log, code in zip(logs, tests):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                test_path = Path(f.name)

            line_region: TargetRegion | None = None
            func_region: TargetRegion | None = None
            m = _FRAME_RE.findall(log)
            if m:
                filename, lineno, func = m[-1]
                line_region = TargetRegion(
                    filename=filename,
                    start_line=int(lineno),
                    end_line=int(lineno),
                    function=func.strip(),
                )
                func_region = extract_target_region(log)

            if line_region and func_region:
                level, target = self._tracker.level_for(line_region, func_region)
                module_path = Path(func_region.filename)
            else:
                level, target = "module", None
                module_path = test_path

            try:
                retrieval_context = self.context_builder.build_context(str(log))
            except Exception:
                self.logger.exception("context build failed")
                retrieval_context = None

            @retry(Exception, attempts=3)
            def _apply(path: Path, region: TargetRegion | None) -> dict[str, Any] | None:
                kwargs: dict[str, Any] = {}
                if retrieval_context is not None:
                    kwargs["context_meta"] = {"retrieval_context": retrieval_context}
                return self.manager.auto_run_patch(path, "auto_debug", **kwargs)

            summary: dict[str, Any] | None = None
            try:
                summary = _apply(module_path, target)
                try:
                    from sandbox_runner import integrate_new_orphans  # type: ignore

                    integrate_new_orphans(
                        Path.cwd(), context_builder=self.context_builder
                    )
                except Exception:
                    self.logger.exception(
                        "integrate_new_orphans after apply_patch failed",
                    )
                failed_tests = int(summary.get("self_tests", {}).get("failed", 0)) if summary else 0
                if failed_tests:
                    raise RuntimeError(f"self tests failed ({failed_tests})")
                if summary is None:
                    raise RuntimeError("post validation summary unavailable")
                self.logger.info("patch succeeded for %s", test_path.name)
                if line_region:
                    self._tracker.reset(line_region)
            except Exception:
                self.logger.exception("patch failed")
                if line_region and func_region:
                    self._tracker.record_failure(level, line_region, func_region)
            finally:
                test_path.unlink(missing_ok=True)


__all__ = ["AutomatedDebugger"]
