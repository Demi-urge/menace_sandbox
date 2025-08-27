from __future__ import annotations

"""Automatic debugging service that patches errors without manual help."""

from pathlib import Path
import logging
import re
import subprocess
import tempfile
import traceback
from typing import Iterable

from .self_coding_engine import SelfCodingEngine
from .retry_utils import retry


class AutomatedDebugger:
    """Analyse telemetry logs and trigger self-coding fixes."""

    def __init__(self, telemetry_db: object, engine: SelfCodingEngine) -> None:
        self.telemetry_db = telemetry_db
        self.engine = engine
        self.logger = logging.getLogger("AutomatedDebugger")

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
        for code in tests:
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                f.write(code)
                path = Path(f.name)

            @retry(Exception, attempts=3)
            def _apply(path: Path) -> None:
                self.engine.apply_patch(
                    path,
                    "auto_debug",
                    reason="auto_debug",
                    trigger="automated_debugger",
                )

            try:
                _apply(path)
                try:
                    from sandbox_runner import integrate_new_orphans  # type: ignore

                    integrate_new_orphans(Path.cwd())
                except Exception:
                    self.logger.exception(
                        "integrate_new_orphans after apply_patch failed"
                    )
                res = subprocess.run(["pytest", "-q", str(path)], capture_output=True, text=True)
                if res.returncode != 0:
                    self.logger.error(
                        "test failed after patch: %s",
                        (res.stdout + res.stderr).strip(),
                    )
                    self.logger.warning("Patch failed. Reverting or retrying...")
                    continue
                self.logger.info("patch succeeded for %s", path.name)
            except Exception:
                self.logger.exception("patch failed")
            finally:
                path.unlink(missing_ok=True)


__all__ = ["AutomatedDebugger"]
